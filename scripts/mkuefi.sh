#!/usr/bin/env bash
# ============================================================================
# scripts/mkuefi.sh — Build the UEFI boot disk (BOOTX64.EFI + kernel + chunked
# model on FAT32) and optionally boot it in QEMU/OVMF.
#
#   scripts/mkuefi.sh <model.asmbin> <out.img> [cmdline]      # build
#   RUN=1 RAM=14G SMP=16 scripts/mkuefi.sh ...                 # build + boot
#
# Independent of the Makefile: the kernel ELF64 must already be built
# (`make os` -> build/os/asm-llm-os.elf64). The GRUB/Multiboot path
# (mkiso.sh, make os-run) is unaffected.
# Put <out.img> on a native Linux filesystem: the image holds the full model.
# ============================================================================
set -euo pipefail

MODEL="${1:?usage: mkuefi.sh <model.asmbin> <out.img> [cmdline]}"
OUT_IMG="${2:?missing output disk image path}"
CMDLINE="${3:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL="$ROOT/build/os/asm-llm-os.elf64"
LOADER="$ROOT/build/uefi/BOOTX64.EFI"
OVMF="${OVMF:-/usr/share/ovmf/OVMF.fd}"

[[ -f "$KERNEL" ]] || { echo "error: $KERNEL missing (run: make os)" >&2; exit 1; }

mkdir -p "$(dirname "$LOADER")"
nasm -f bin -w+all -I"$ROOT/02-baremetal-os/" -I"$ROOT/shared/" \
    "$ROOT/02-baremetal-os/uefi_loader.asm" -o "$LOADER"

python3 "$ROOT/scripts/mkuefi_img.py" --loader "$LOADER" --kernel "$KERNEL" \
    --model "$MODEL" --out "$OUT_IMG" --cmdline "$CMDLINE"

if [[ "${RUN:-0}" == "1" ]]; then
    ACCEL=(-cpu max)
    [[ -w /dev/kvm ]] && ACCEL=(-enable-kvm -cpu host)
    exec qemu-system-x86_64 -machine q35 "${ACCEL[@]}" -m "${RAM:-14G}" -smp "${SMP:-4}" \
        -bios "$OVMF" -drive file="$OUT_IMG",format=raw,if=virtio \
        -nographic -no-reboot
fi
