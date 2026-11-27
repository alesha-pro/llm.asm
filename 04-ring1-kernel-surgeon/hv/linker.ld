ENTRY(_start)

SECTIONS
{
    . = 0x00200000;
    __hv_start = .;

    .text ALIGN(4096) : {
        KEEP(*(.multiboot))
        *(.text .text.*)
    }

    .rodata ALIGN(4096) : {
        *(.rodata .rodata.*)
    }

    .data ALIGN(4096) : {
        *(.data .data.*)
    }

    . = ALIGN(4096);
    __bss_vma = .;
    .bss : {
        __bss_start = .;
        *(COMMON)
        *(.bss .bss.*)
        . = ALIGN(4096);
        __bss_end = .;
        . = 0x00400000 - __bss_vma;
    }

    __hv_end = .;

    ASSERT(__hv_end == 0x00400000, "Hypervisor arena must be exact 2 MiB (0x00200000..0x00400000)")

    /DISCARD/ : {
        *(.comment)
        *(.note.*)
        *(.eh_frame*)
    }
}
