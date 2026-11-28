#!/usr/bin/env python3
"""
scripts/mkuefi_img.py — Build a raw UEFI boot disk (MBR + FAT32 ESP) for the
asm-llm UEFI loader path. Pure Python, no mtools / root / loop mounts
(same policy as mkiso.sh Path B).

Layout written to the volume:
    \\EFI\\BOOT\\BOOTX64.EFI   uefi_loader.asm output
    \\LLM\\KERNEL.ELF          build/os/asm-llm-os.elf64 (unchanged kernel)
    \\LLM\\CMDLINE.TXT         kernel cmdline (optional)
    \\LLM\\MODEL.000 ...       .asmbin split into chunks (FAT32 max file size
                             is 4 GiB - 1; Qwen3-8B Q8_0 is ~8.7 GB)

The model is never copied to temporary chunk files: chunks are streamed from
the source .asmbin straight into their clusters. All-zero 1 MiB blocks are
skipped, so the image stays sparse where the source is sparse.

Usage:
    mkuefi_img.py --loader BOOTX64.EFI --kernel asm-llm-os.elf64 \\
                  --model qwen3-8b-q8_0.asmbin --out disk.img [--cmdline "..."]
"""
import argparse
import math
import os
import struct
import sys

SECTOR = 512
PART_LBA = 2048                  # 1 MiB aligned partition start
SPC = 64                         # 32 KiB clusters
CLUSTER = SECTOR * SPC
RESERVED = 32
NFATS = 2
MIN_FAT32_CLUSTERS = 65525 + 64  # FAT32 needs >= 65525 clusters
EOC = 0x0FFFFFFF
DEFAULT_CHUNK = 2 << 30          # 2 GiB: far from any signed-32-bit edge
FAT32_MAX_FILE = (4 << 30) - 1
ZERO_BLOCK = 1 << 20

# Fixed timestamp 2026-01-01 00:00:00 (reproducible images)
FAT_DATE = ((2026 - 1980) << 9) | (1 << 5) | 1
FAT_TIME = 0


def name83(name: str) -> bytes:
    base, _, ext = name.upper().partition(".")
    assert 1 <= len(base) <= 8 and len(ext) <= 3, name
    return (base.ljust(8) + ext.ljust(3)).encode("ascii")


def dirent(name: bytes, attr: int, clus: int, size: int) -> bytes:
    return struct.pack(
        "<11sBBBHHHHHHHI",
        name, attr, 0, 0, FAT_TIME, FAT_DATE, FAT_DATE,
        (clus >> 16) & 0xFFFF, FAT_TIME, FAT_DATE, clus & 0xFFFF, size,
    )


class Node:
    def __init__(self, name, is_dir, src=None, offset=0, size=0, data=None):
        self.name, self.is_dir = name, is_dir
        self.src, self.offset, self.size, self.data = src, offset, size, data
        self.children = []
        self.clus = 0
        self.nclus = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmdline", default="")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    args = ap.parse_args()

    if not (CLUSTER <= args.chunk_size <= FAT32_MAX_FILE):
        sys.exit(f"--chunk-size must be in [{CLUSTER}, {FAT32_MAX_FILE}]")

    model_size = os.path.getsize(args.model)
    with open(args.model, "rb") as f:
        hdr = f.read(128)
    if hdr[:4] != b"ASML":
        sys.exit(f"{args.model}: not an .asmbin (bad magic)")
    hdr_file_size = struct.unpack_from("<Q", hdr, 104)[0]
    if hdr_file_size != model_size:
        sys.exit(f"{args.model}: header file_size {hdr_file_size} != actual {model_size}")

    # ---- Tree -------------------------------------------------------------
    root = Node("", True)
    efi = Node("EFI", True); boot = Node("BOOT", True); llm = Node("LLM", True)
    root.children = [efi, llm]
    efi.children = [boot]
    boot.children = [Node("BOOTX64.EFI", False, src=args.loader,
                          size=os.path.getsize(args.loader))]
    llm.children = [Node("KERNEL.ELF", False, src=args.kernel,
                         size=os.path.getsize(args.kernel))]
    if args.cmdline:
        cmd = args.cmdline.encode("ascii")
        if len(cmd) > 1023:
            sys.exit("cmdline longer than 1023 bytes")
        llm.children.append(Node("CMDLINE.TXT", False, data=cmd, size=len(cmd)))
    nchunks = math.ceil(model_size / args.chunk_size)
    if nchunks > 64:
        sys.exit("more than 64 model chunks (loader MAX_CHUNKS); raise --chunk-size")
    for i in range(nchunks):
        off = i * args.chunk_size
        llm.children.append(Node(f"MODEL.{i:03d}", False, src=args.model, offset=off,
                                 size=min(args.chunk_size, model_size - off)))

    # ---- Cluster allocation (contiguous, depth-first) -----------------------
    order = []
    nxt = 2

    def alloc(n):
        nonlocal nxt
        n.nclus = 1 if n.is_dir else max(1, math.ceil(n.size / CLUSTER)) if n.size else 0
        if n.is_dir:
            assert (len(n.children) + 2) * 32 <= CLUSTER
        if n.nclus:
            n.clus, nxt = nxt, nxt + n.nclus
        order.append(n)
        for c in n.children:
            alloc(c)

    alloc(root)
    used = nxt - 2
    nclusters = max(MIN_FAT32_CLUSTERS, used + 64)
    fat_sectors = math.ceil((nclusters + 2) * 4 / SECTOR)
    data_lba = RESERVED + NFATS * fat_sectors          # relative to partition
    part_sectors = data_lba + nclusters * SPC
    disk_sectors = PART_LBA + part_sectors + 2048

    def clus_off(c):
        return (PART_LBA + data_lba + (c - 2) * SPC) * SECTOR

    # ---- Write ------------------------------------------------------------
    with open(args.out, "wb") as out:
        out.truncate(disk_sectors * SECTOR)

        # MBR: one partition, type 0xEF (EFI System)
        mbr = bytearray(SECTOR)
        struct.pack_into("<B3sB3sII", mbr, 446, 0x00, b"\xFE\xFF\xFF", 0xEF,
                         b"\xFE\xFF\xFF", PART_LBA, part_sectors)
        mbr[510:512] = b"\x55\xAA"
        out.seek(0); out.write(mbr)

        # FAT32 boot sector
        bs = bytearray(SECTOR)
        struct.pack_into(
            "<3s8sHBHBHHBHHHII", bs, 0, b"\xEB\x58\x90", b"ASMLLMOS",
            SECTOR, SPC, RESERVED, NFATS, 0, 0, 0xF8, 0, 63, 255, PART_LBA, part_sectors)
        struct.pack_into(
            "<IHHIHH12sBBBI11s8s", bs, 36, fat_sectors, 0, 0, 2, 1, 6, b"\x00" * 12,
            0x80, 0, 0x29, 0x20261128, b"ASMLLM_UEFI", b"FAT32   ")
        bs[510:512] = b"\x55\xAA"
        free = nclusters - used
        fsinfo = bytearray(SECTOR)
        struct.pack_into("<I", fsinfo, 0, 0x41615252)
        struct.pack_into("<III", fsinfo, 484, 0x61417272, free, nxt)
        struct.pack_into("<I", fsinfo, 508, 0xAA550000)
        for base in (0, 6):
            out.seek((PART_LBA + base) * SECTOR); out.write(bs)
            out.seek((PART_LBA + base + 1) * SECTOR); out.write(fsinfo)

        # FAT
        fat = bytearray(fat_sectors * SECTOR)
        struct.pack_into("<II", fat, 0, 0x0FFFFFF8, EOC)
        for n in order:
            for k in range(n.nclus):
                c = n.clus + k
                struct.pack_into("<I", fat, c * 4, EOC if k == n.nclus - 1 else c + 1)
        for i in range(NFATS):
            out.seek((PART_LBA + RESERVED + i * fat_sectors) * SECTOR)
            out.write(fat)

        # Directories
        def parent_of(target, node=root, parent=None):
            if node is target:
                return parent
            for c in node.children:
                p = parent_of(target, c, node)
                if p is not None:
                    return p
            return None

        for n in order:
            if not n.is_dir:
                continue
            ents = bytearray()
            if n is root:
                ents += dirent(b"ASMLLM_UEFI", 0x08, 0, 0)
            else:
                p = parent_of(n)
                ents += dirent(b".          ", 0x10, n.clus, 0)
                ents += dirent(b"..         ", 0x10, 0 if p is root else p.clus, 0)
            for c in n.children:
                ents += dirent(name83(c.name), 0x10 if c.is_dir else 0x20,
                               c.clus, 0 if c.is_dir else c.size)
            out.seek(clus_off(n.clus)); out.write(ents)

        # File data
        for n in order:
            if n.is_dir or not n.size:
                continue
            dst = clus_off(n.clus)
            if n.data is not None:
                out.seek(dst); out.write(n.data)
                continue
            with open(n.src, "rb") as f:
                f.seek(n.offset)
                left = n.size
                while left:
                    blk = f.read(min(ZERO_BLOCK, left))
                    if not blk:
                        sys.exit(f"{n.src}: short read")
                    if blk.count(0) != len(blk):
                        out.seek(dst); out.write(blk)
                    dst += len(blk); left -= len(blk)
            if n.name.startswith("MODEL."):
                print(f"  {n.name}: {n.size:,} B")

    print(f"[mkuefi_img] {args.out}: {disk_sectors * SECTOR / 2**30:.2f} GiB, "
          f"FAT32 {nclusters} clusters x {CLUSTER // 1024} KiB, "
          f"model {model_size:,} B in {nchunks} chunk(s)")


if __name__ == "__main__":
    main()
