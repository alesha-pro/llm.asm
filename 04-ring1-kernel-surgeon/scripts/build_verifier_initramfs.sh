#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KO_PATH="${1:-$PROJ/guest/bench_mod/surgeon_bench.ko}"
OUT_CPIO="$(realpath -m "${2:-$PROJ/build/verifier-initramfs.cpio.gz}")"

WORK="$(mktemp -d "$PROJ/build/ver-initrd.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK"/{bin,sbin,proc,sys,dev,tmp,lib/modules}
cp /usr/bin/busybox "$WORK/bin/busybox"
chmod 755 "$WORK/bin/busybox"
for cmd in sh ash cat echo mount insmod uptime poweroff sleep grep cut; do
    ln -s busybox "$WORK/bin/$cmd"
done
cp "$KO_PATH" "$WORK/lib/modules/surgeon_bench.ko"

cat > "$WORK/init" << 'INITEOF'
#!/bin/sh
export PATH=/bin:/sbin
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
echo 4 > /proc/sys/kernel/printk 2>/dev/null || true
insmod /lib/modules/surgeon_bench.ko
BOOT_ID="$(cat /proc/sys/kernel/random/boot_id)"
echo "VERIFIER_READY boot_id=${BOOT_ID} uptime=$(cut -d" " -f1 /proc/uptime)"
while IFS= read -r line; do
    case "$line" in
        PING)
            echo "PONG boot_id=$(cat /proc/sys/kernel/random/boot_id) uptime=$(cut -d" " -f1 /proc/uptime)"
            ;;
        QUIT)
            echo "BYE"
            poweroff -f
            ;;
        *)
            echo "$line" > /proc/surgeon_bench
            RES="$(cat /proc/surgeon_bench)"
            echo "${RES} boot_id=$(cat /proc/sys/kernel/random/boot_id) uptime=$(cut -d" " -f1 /proc/uptime)"
            ;;
    esac
done
INITEOF
chmod 755 "$WORK/init"

(cd "$WORK" && find . -print0 | cpio --null -ov --format=newc 2>/dev/null | gzip -9 > "$OUT_CPIO")
ls -lh "$OUT_CPIO"
