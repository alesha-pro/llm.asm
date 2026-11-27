#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KO_PATH="${1:?usage: build_initramfs.sh <surgeon_crash.ko> <out.cpio.gz>}"
OUT_CPIO="$(realpath -m "${2:?missing output path}")"

WORK="$(mktemp -d "$PROJ/build/initramfs.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK"/{bin,sbin,etc,proc,sys,dev,tmp,root,lib/modules}
cp /usr/bin/busybox "$WORK/bin/busybox"
chmod 755 "$WORK/bin/busybox"

for cmd in sh ash ls cat echo mount umount mkdir ps uptime dmesg uname insmod rmmod lsmod sleep clear id head tail grep awk sed cut tr wc top poweroff reboot; do
    ln -s busybox "$WORK/bin/$cmd"
done

cp "$KO_PATH" "$WORK/lib/modules/surgeon_crash.ko"

cat > "$WORK/init" << 'INITEOF'
#!/bin/sh
export PATH=/bin:/sbin

mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true

echo 7 > /proc/sys/kernel/printk 2>/dev/null || true
echo 1 > /proc/sys/kernel/sysrq 2>/dev/null || true

insmod /lib/modules/surgeon_crash.ko || true

banner() {
    echo ""
    echo "================================================================"
    echo "  LINUX 5.15 GUEST — RING -1 KERNEL SURGEON (MiniCPM5-2B Q4_K_M)"
    echo "================================================================"
    uname -a
    echo "Uptime: $(cat /proc/uptime) | Boot ID: $(cat /proc/sys/kernel/random/boot_id)"
    echo "Framebuffer: $(cat /sys/class/graphics/fb0/name 2>/dev/null || echo none) ($(cat /sys/class/graphics/fb0/virtual_size 2>/dev/null || echo n/a))"
    echo ""
    echo "Live Ring 0 Crash Tests (Bare-Metal MiniCPM5-2B inside Ring -1):"
    echo "  echo pf      > /proc/surgeon_crash   # Ring 0 NULL deref (#PF -> 11x NOP)"
    echo "  echo de      > /proc/surgeon_crash   # Ring 0 div/0 (#DE -> 31 c0 xor eax,eax)"
    echo "  echo ud      > /proc/surgeon_crash   # Ring 0 UD2 trap (#UD -> 90 90)"
    echo "  echo extable > /proc/surgeon_crash   # Benign Linux extable (PASS_TO_LINUX)"
    echo "  echo danger  > /proc/surgeon_crash   # Spinlock-held #PF (HALT_FOR_DUMP)"
    echo "  echo status  > /proc/surgeon_crash   # Query Ring -1 VMMCALL"
    echo "================================================================"
    echo ""
}

banner
if [ -c /dev/tty0 ]; then
    banner > /dev/tty0 2>/dev/null || true
fi
if [ -c /dev/tty0 ]; then
    setsid sh -c 'exec /bin/sh </dev/tty0 >/dev/tty0 2>&1' &
fi
exec /bin/sh
INITEOF
chmod 755 "$WORK/init"

(cd "$WORK" && find . -print0 | cpio --null -ov --format=newc 2>/dev/null | gzip -9 > "$OUT_CPIO")
ls -lh "$OUT_CPIO"
