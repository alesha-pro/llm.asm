NASM    := nasm
CC      := gcc
LD      := ld

BUILD   := build
HV_DIR  := hv
HV_BLD  := $(BUILD)/hv

CFLAGS_NOFPU := -m64 -ffreestanding -fno-stack-protector -fno-pic -fno-pie \
                -mcmodel=small -mno-red-zone -mno-mmx -mno-sse -mno-sse2 -mno-avx \
                -O2 -Wall -Wextra -Werror -std=gnu11 -I$(HV_DIR)

CFLAGS_AVX2  := -m64 -ffreestanding -fno-stack-protector -fno-pic -fno-pie \
                -mcmodel=small -mno-red-zone -mavx2 -mfma -mf16c \
                -O3 -Wall -Wextra -Werror -std=gnu11 -I$(HV_DIR)

AFLAGS  := -f elf64
LDFLAGS := -m elf_x86_64 -nostdlib -T $(HV_DIR)/linker.ld -z max-page-size=0x1000

HV_ELF  := $(BUILD)/surgeon.elf
KO      := guest/crash_mod/surgeon_crash.ko
INITRD  := $(BUILD)/initramfs.cpio.gz
VMLINUZ := guest/vmlinuz
ISO     := $(BUILD)/ring1-surgeon.iso

HV_OBJS := \
	$(HV_BLD)/boot.o \
	$(HV_BLD)/svm_asm.o \
	$(HV_BLD)/ap_trampoline.o \
	$(HV_BLD)/serial.o \
	$(HV_BLD)/fb.o \
	$(HV_BLD)/npt.o \
	$(HV_BLD)/smp.o \
	$(HV_BLD)/x86_decode.o \
	$(HV_BLD)/linux_boot.o \
	$(HV_BLD)/llm_engine.o \
	$(HV_BLD)/surgeon.o \
	$(HV_BLD)/main.o

.PHONY: all hv crash-mod initramfs iso clean

all: iso

$(HV_BLD):
	mkdir -p $(HV_BLD)

$(HV_BLD)/%.o: $(HV_DIR)/%.asm | $(HV_BLD)
	$(NASM) $(AFLAGS) $< -o $@

$(HV_BLD)/llm_engine.o: $(HV_DIR)/llm_engine.c $(HV_DIR)/*.h | $(HV_BLD)
	$(CC) $(CFLAGS_AVX2) -c $< -o $@

$(HV_BLD)/%.o: $(HV_DIR)/%.c $(HV_DIR)/*.h | $(HV_BLD)
	$(CC) $(CFLAGS_NOFPU) -c $< -o $@

$(HV_ELF): $(HV_OBJS) $(HV_DIR)/linker.ld
	$(LD) $(LDFLAGS) $(HV_OBJS) -o $@
	@ls -lh $@

hv: $(HV_ELF)

$(KO): guest/crash_mod/surgeon_crash.c guest/crash_mod/Makefile
	$(MAKE) -C guest/crash_mod

crash-mod: $(KO)

$(INITRD): $(KO) scripts/build_initramfs.sh | $(HV_BLD)
	bash scripts/build_initramfs.sh $(KO) $(INITRD)

initramfs: $(INITRD)

$(ISO): $(HV_ELF) $(VMLINUZ) $(INITRD) scripts/mkiso.sh
	bash scripts/mkiso.sh $(HV_ELF) $(VMLINUZ) $(INITRD) $(ISO)

iso: $(ISO)

clean:
	rm -rf $(BUILD)/hv $(HV_ELF) $(INITRD) $(ISO)
	$(MAKE) -C guest/crash_mod clean || true
