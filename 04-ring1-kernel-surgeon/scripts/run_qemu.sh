#!/usr/bin/env bash
# Launch Ring -1 Kernel Surgeon inside QEMU Nested AMD SVM + OVMF UEFI + VNC
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISO="${ISO:-$PROJ/build/ring1-surgeon.iso}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
VNC_BIND="${VNC_BIND:-127.0.0.1}"
RAM="${RAM:-2G}"
SMP="${SMP:-1}"
SERIAL_LOG="${SERIAL_LOG:-$PROJ/build/serial.log}"
MONITOR_SOCK="${MONITOR_SOCK:-$PROJ/build/qemu-monitor.sock}"
OVMF_CODE="${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE.fd}"
OVMF_VARS_SRC="${OVMF_VARS_SRC:-/usr/share/OVMF/OVMF_VARS.fd}"
OVMF_VARS="$PROJ/build/OVMF_VARS.fd"

if [[ ! -f "$ISO" ]]; then
    echo "ISO not found: $ISO — run 'make all' first." >&2
    exit 1
fi

if [[ ! -f "$OVMF_VARS" ]]; then
    cp "$OVMF_VARS_SRC" "$OVMF_VARS"
fi

# Derive TCP port for user-friendly printout
VNC_NUM="${VNC_DISPLAY#:}"
VNC_PORT="$((5900 + VNC_NUM))"

echo "================================================================"
echo " Ring -1 Kernel Surgeon — QEMU Nested AMD SVM + VNC Harness"
echo "================================================================"
echo " ISO         : $ISO"
echo " CPU         : EPYC,+svm (KVM nested=$(cat /sys/module/kvm_amd/parameters/nested))"
echo " RAM / SMP   : $RAM / $SMP"
echo " VNC         : vnc://${VNC_BIND}:${VNC_PORT}  (${VNC_BIND}${VNC_DISPLAY})"
echo " Serial log  : $SERIAL_LOG"
echo " QMP/Monitor : $MONITOR_SOCK"
echo "================================================================"

rm -f "$MONITOR_SOCK"

exec qemu-system-x86_64 \
    -machine q35,accel=kvm \
    -cpu EPYC,+svm \
    -smp "$SMP" \
    -m "$RAM" \
    -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
    -drive if=pflash,format=raw,file="$OVMF_VARS" \
    -cdrom "$ISO" \
    -boot d \
    -device VGA,vgamem_mb=32,xres=1280,yres=720 \
    -vnc "${VNC_BIND}${VNC_DISPLAY}" \
    -chardev stdio,id=ser0,mux=on,logfile="$SERIAL_LOG",signal=off \
    -serial chardev:ser0 \
    -monitor unix:"$MONITOR_SOCK",server,nowait \
    "$@"
