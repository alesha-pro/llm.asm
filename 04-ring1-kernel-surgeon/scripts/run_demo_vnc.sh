#!/usr/bin/env bash
# ============================================================================
# Ring -1 Kernel Surgeon — One-Take VNC Video Demo Launcher
# Usage:
#   ./scripts/run_demo_vnc.sh vanilla   # Unprotected Linux -> Kernel panic on echo pf
#   ./scripts/run_demo_vnc.sh surgeon   # Ring -1 + Bare-Metal MiniCPM5-2B Q4_K_M (default)
# Connect with a VNC client to: vnc://127.0.0.1:5901 (or set VNC_BIND=0.0.0.0)
# ============================================================================
set -euo pipefail

MODE="${1:-surgeon}"
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISO="${ISO:-$PROJ/build/ring1-surgeon.iso}"
VMLINUZ="${VMLINUZ:-$PROJ/guest/vmlinuz}"
INITRD="${INITRD:-$PROJ/build/initramfs.cpio.gz}"
MODEL_R1M="${MODEL_R1M:-$PROJ/models/minicpm5-2b-surgeon.r1m}"
OVMF_CODE="${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE.fd}"
OVMF_VARS="${OVMF_VARS:-$PROJ/build/OVMF_VARS.fd}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
VNC_BIND="${VNC_BIND:-127.0.0.1}"
SMP="${SMP:-24}"
SERIAL_LOG="${SERIAL_LOG:-$PROJ/build/demo_${MODE}_serial.log}"

if [[ ! -f "$OVMF_VARS" ]]; then
    cp /usr/share/OVMF/OVMF_VARS.fd "$OVMF_VARS"
fi

VNC_NUM="${VNC_DISPLAY#:}"
VNC_PORT="$((5900 + VNC_NUM))"

echo "======================================================================"
echo " RING -1 KERNEL SURGEON — LIVE VNC DEMO (MODE: ${MODE^^})"
echo "======================================================================"
echo " VNC URL    : vnc://${VNC_BIND}:${VNC_PORT}  (${VNC_BIND}${VNC_DISPLAY})"
echo " Serial log : ${SERIAL_LOG}"
if [[ "$MODE" == "vanilla" ]]; then
    echo " Mode       : UNPROTECTED LINUX 5.15 (expect Kernel panic on echo pf!)"
    echo "======================================================================"
    exec qemu-system-x86_64 \
        -machine q35,accel=kvm \
        -cpu host -smp 1 -m 2G \
        -kernel "$VMLINUZ" \
        -initrd "$INITRD" \
        -append "console=tty0 console=ttyS0,115200 nokaslr panic_on_oops=1 rdinit=/init vga=0x344" \
        -device VGA,vgamem_mb=32,xres=1280,yres=720 \
        -vnc "${VNC_BIND}${VNC_DISPLAY}" \
        -chardev stdio,id=ser0,mux=on,logfile="$SERIAL_LOG",signal=off \
        -serial chardev:ser0
else
    if [[ ! -f "$MODEL_R1M" ]]; then
        echo "ERROR: Missing $MODEL_R1M" >&2; exit 1
    fi
    echo " Mode       : RING -1 AMD SVM + BARE-METAL MiniCPM5-2B Q4_K_M (${SMP} L1 Cores)"
    echo " Model NVDIMM: $MODEL_R1M (2 GiB @ 0x100000000, UNMAPPED in AMD NPT)"
    echo "======================================================================"
    exec qemu-system-x86_64 \
        -machine q35,accel=kvm,nvdimm=on \
        -cpu EPYC,+svm \
        -smp "$SMP" \
        -m 2G,slots=2,maxmem=6G \
        -object memory-backend-file,id=r1model,mem-path="$MODEL_R1M",size=2G,readonly=on \
        -device nvdimm,id=nv1,memdev=r1model,unarmed=on \
        -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
        -drive if=pflash,format=raw,file="$OVMF_VARS" \
        -cdrom "$ISO" \
        -boot d \
        -device VGA,vgamem_mb=32,xres=1280,yres=720 \
        -vnc "${VNC_BIND}${VNC_DISPLAY}" \
        -chardev stdio,id=ser0,mux=on,logfile="$SERIAL_LOG",signal=off \
        -serial chardev:ser0
fi
