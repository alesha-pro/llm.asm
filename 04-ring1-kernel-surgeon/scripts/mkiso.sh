#!/usr/bin/env bash
# Build UEFI GRUB2 Multiboot2 bootable ISO containing:
#   - /boot/surgeon.elf       (Ring -1 AMD SVM Hypervisor)
#   - /boot/vmlinuz           (Unmodified Linux guest bzImage)
#   - /boot/initramfs.cpio.gz (BusyBox + surgeon_crash.ko)
set -euo pipefail

HV_ELF="${1:?usage: mkiso.sh <surgeon.elf> <vmlinuz> <initramfs.cpio.gz> <out.iso>}"
VMLINUZ="${2:?missing vmlinuz}"
INITRD="${3:?missing initramfs}"
OUT_ISO="${4:?missing out.iso}"

OUT_DIR="$(dirname "$OUT_ISO")"
mkdir -p "$OUT_DIR"
WORK="$(mktemp -d "$OUT_DIR/iso-work.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

ISO_ROOT="$WORK/iso"
mkdir -p "$ISO_ROOT/boot/grub"
cp "$HV_ELF"  "$ISO_ROOT/boot/surgeon.elf"
cp "$VMLINUZ" "$ISO_ROOT/boot/vmlinuz"
cp "$INITRD"  "$ISO_ROOT/boot/initramfs.cpio.gz"

cat > "$ISO_ROOT/boot/grub/grub.cfg" << 'GRUBEOF'
serial --unit=0 --speed=115200 --word=8 --parity=no --stop=1
terminal_input console serial
terminal_output console serial
set timeout=1
set default=0

insmod all_video
insmod gfxterm
set gfxmode=1280x720x32,1024x768x32,auto
set gfxpayload=keep

menuentry "Ring -1 Kernel Surgeon (AMD SVM Hypervisor + Linux Guest)" {
    multiboot2 /boot/surgeon.elf
    module2 /boot/vmlinuz vmlinuz
    module2 /boot/initramfs.cpio.gz initramfs
    boot
}
GRUBEOF

mkdir -p "$WORK/efi_tree/EFI/BOOT"
cat > "$WORK/embed.cfg" << 'EMBEOF'
search --no-floppy --file --set=root /boot/surgeon.elf
set prefix=($root)/boot/grub
configfile ($root)/boot/grub/grub.cfg
EMBEOF

grub-mkstandalone \
    -O x86_64-efi \
    -o "$WORK/efi_tree/EFI/BOOT/BOOTX64.EFI" \
    --modules="part_gpt part_msdos iso9660 fat multiboot2 normal serial terminal search search_fs_file configfile all_video efi_gop efi_uga video_bochs video_cirrus gfxterm font" \
    "boot/grub/grub.cfg=$WORK/embed.cfg"

# Build FAT16 EFI System Partition image (zero root)
EFI_IMG="$ISO_ROOT/boot/efiboot.img"
python3 - "$WORK/efi_tree/EFI/BOOT/BOOTX64.EFI" "$EFI_IMG" << 'PYEOF'
import math, struct, sys

efi_src, out_img = sys.argv[1], sys.argv[2]
data = open(efi_src, "rb").read()

BPS = 512
SPC = 4
CLUST_BYTES = BPS * SPC
RESV = 4
NFATS = 2
ROOT_ENTRIES = 512
ROOT_SECTORS = (ROOT_ENTRIES * 32) // BPS
data_clusters = max(16, math.ceil(len(data) / CLUST_BYTES) + 32)
total_clusters = max(4120, data_clusters)
fat_sectors = math.ceil((total_clusters + 2) * 2 / BPS)
data_start_sec = RESV + NFATS * fat_sectors + ROOT_SECTORS
total_sectors = data_start_sec + total_clusters * SPC

img = bytearray(total_sectors * BPS)
bpb = struct.pack(
    "<3s8sHBHBHHBHHHIIBBBI11s8s",
    b"\xEB\x3C\x90", b"RING1SRG",
    BPS, SPC, RESV, NFATS, ROOT_ENTRIES,
    total_sectors if total_sectors < 65536 else 0,
    0xF8, fat_sectors, 32, 64, 0,
    0 if total_sectors < 65536 else total_sectors,
    0x80, 0, 0x29, 0x20261127, b"RING1_EFI  ", b"FAT16   ",
)
img[0:len(bpb)] = bpb
img[510:512] = b"\x55\xAA"

fat = bytearray(fat_sectors * BPS)
struct.pack_into("<HH", fat, 0, 0xFFF8, 0xFFFF)

def clust_off(c):
    return (data_start_sec + (c - 2) * SPC) * BPS

def dir_entry(name83, attr, first_clust, size):
    return struct.pack("<11sBBBHHHHHHHL", name83, attr, 0, 0, 0, 0, 0, 0, 0, 0, first_clust, size)

struct.pack_into("<HH", fat, 2 * 2, 0xFFFF, 0xFFFF)
n_file_clust = math.ceil(len(data) / CLUST_BYTES)
for i in range(n_file_clust):
    c = 4 + i
    nxt = (c + 1) if (i + 1 < n_file_clust) else 0xFFFF
    struct.pack_into("<H", fat, c * 2, nxt)

for f in range(NFATS):
    off = (RESV + f * fat_sectors) * BPS
    img[off:off + len(fat)] = fat

root_off = (RESV + NFATS * fat_sectors) * BPS
img[root_off:root_off + 32] = dir_entry(b"EFI        ", 0x10, 2, 0)

c2 = clust_off(2)
img[c2 + 0:c2 + 32]  = dir_entry(b".          ", 0x10, 2, 0)
img[c2 + 32:c2 + 64] = dir_entry(b"..         ", 0x10, 0, 0)
img[c2 + 64:c2 + 96] = dir_entry(b"BOOT       ", 0x10, 3, 0)

c3 = clust_off(3)
img[c3 + 0:c3 + 32]  = dir_entry(b".          ", 0x10, 3, 0)
img[c3 + 32:c3 + 64] = dir_entry(b"..         ", 0x10, 2, 0)
img[c3 + 64:c3 + 96] = dir_entry(b"BOOTX64 EFI", 0x20, 4, len(data))

c4 = clust_off(4)
img[c4:c4 + len(data)] = data

with open(out_img, "wb") as f:
    f.write(img)
PYEOF

xorriso -as mkisofs \
    -R -J -V "RING1_SURGEON" \
    -e boot/efiboot.img \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    -o "$OUT_ISO" \
    "$ISO_ROOT"

ls -lh "$OUT_ISO"
