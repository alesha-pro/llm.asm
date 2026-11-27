#include "svm.h"

// Compact x86-64 instruction length decoder (Intel/AMD 64-bit mode)
// Handles legacy prefixes, REX, 1-byte / 0F 2-byte / 0F 38 / 0F 3A opcodes,
// ModRM, SIB, disp8/disp32, imm8/imm16/imm32/imm64, and Group 3 (F6/F7).

static int decode_modrm(const uint8_t *code, int pos, int max_len, bool addr32) {
    if (pos >= max_len) return -1;
    uint8_t modrm = code[pos++];
    uint8_t mod = (modrm >> 6) & 3;
    uint8_t rm  = modrm & 7;
    (void)addr32;

    if (mod == 3) return pos;

    bool has_sib = (rm == 4);
    uint8_t sib_base = 0;
    if (has_sib) {
        if (pos >= max_len) return -1;
        sib_base = code[pos++] & 7;
    }

    if (mod == 0) {
        if (rm == 5) {
            pos += 4; // RIP-relative disp32
        } else if (has_sib && sib_base == 5) {
            pos += 4; // [scaled index + disp32] or [disp32]
        }
    } else if (mod == 1) {
        pos += 1; // disp8
    } else if (mod == 2) {
        pos += 4; // disp32
    }

    return (pos <= max_len) ? pos : -1;
}

int x86_decode_len(const uint8_t *code, int max_len) {
    if (max_len <= 0) return -1;
    if (max_len > 15) max_len = 15;

    int pos = 0;
    bool pfx_66 = false;
    bool pfx_67 = false;
    uint8_t rex = 0;

    // 1. Legacy prefixes
    while (pos < max_len) {
        uint8_t b = code[pos];
        if (b == 0x66) { pfx_66 = true; pos++; }
        else if (b == 0x67) { pfx_67 = true; pos++; }
        else if (b == 0xF0 || b == 0xF2 || b == 0xF3 ||
                 b == 0x2E || b == 0x36 || b == 0x3E ||
                 b == 0x26 || b == 0x64 || b == 0x65) {
            pos++;
        } else {
            break;
        }
    }

    // 2. REX prefix (0x40..0x4F)
    if (pos < max_len && (code[pos] & 0xF0) == 0x40) {
        rex = code[pos++];
    }
    bool rex_w = (rex & 0x08) != 0;

    if (pos >= max_len) return -1;
    uint8_t op = code[pos++];

    int imm_size = 0;
    bool has_modrm = false;
    int op_imm_word = pfx_66 ? 2 : 4; // operand-size immediate (in 64-bit mode REX.W still uses imm32 except B8..BF!)

    // 3. Two-byte (0F xx) and three-byte (0F 38 / 0F 3A) opcodes
    if (op == 0x0F) {
        if (pos >= max_len) return -1;
        uint8_t op2 = code[pos++];

        if (op2 == 0x38) {
            // 0F 38 xx ModRM
            if (pos >= max_len) return -1;
            pos++;
            has_modrm = true;
        } else if (op2 == 0x3A) {
            // 0F 3A xx ModRM imm8
            if (pos >= max_len) return -1;
            pos++;
            has_modrm = true;
            imm_size = 1;
        } else if ((op2 >= 0x80 && op2 <= 0x8F)) {
            // Jcc rel32
            imm_size = 4;
        } else if (op2 == 0x0B || op2 == 0x05 || op2 == 0x07 || op2 == 0x30 ||
                   op2 == 0x31 || op2 == 0x32 || op2 == 0x33 || op2 == 0xA2 ||
                   op2 == 0x08 || op2 == 0x09 || op2 == 0xA0 || op2 == 0xA1 ||
                   op2 == 0xA8 || op2 == 0xA9 || (op2 >= 0xC8 && op2 <= 0xCF)) {
            // UD2 (0F 0B), SYSCALL, SYSRET, WRMSR, RDTSC, RDMSR, CPUID, BSWAP...
            has_modrm = false;
        } else if (op2 == 0xBA || op2 == 0xA4 || op2 == 0xAC || op2 == 0x70 ||
                   op2 == 0x71 || op2 == 0x72 || op2 == 0x73 || op2 == 0xC2 ||
                   op2 == 0xC4 || op2 == 0xC5 || op2 == 0xC6) {
            has_modrm = true;
            imm_size = 1;
        } else {
            // Most 0F xx instructions (MOVZX/MOVSX B6/B7/BE/BF, CMOVcc 40..4F, SETcc 90..9F,
            // IMUL AF, BT/BTS/BTR/BTC, CMPXCHG B0/B1, XADD C0/C1, SSE/MOVUPS/etc.) take ModRM
            has_modrm = true;
        }
    } else {
        // 1-byte opcodes
        switch (op) {
            // ALU r/m, r and r, r/m
            case 0x00: case 0x01: case 0x02: case 0x03:
            case 0x08: case 0x09: case 0x0A: case 0x0B:
            case 0x10: case 0x11: case 0x12: case 0x13:
            case 0x18: case 0x19: case 0x1A: case 0x1B:
            case 0x20: case 0x21: case 0x22: case 0x23:
            case 0x28: case 0x29: case 0x2A: case 0x2B:
            case 0x30: case 0x31: case 0x32: case 0x33:
            case 0x38: case 0x39: case 0x3A: case 0x3B:
            case 0x63: // MOVSXD
            case 0x84: case 0x85: case 0x86: case 0x87: // TEST / XCHG
            case 0x88: case 0x89: case 0x8A: case 0x8B: // MOV
            case 0x8C: case 0x8D: case 0x8E: case 0x8F: // MOV seg / LEA / POP r/m
            case 0xD0: case 0xD1: case 0xD2: case 0xD3: // shifts by 1 / CL
            case 0xFE: case 0xFF:                       // INC/DEC/CALL/JMP/PUSH r/m
                has_modrm = true;
                break;

            // ALU AL/eAX, imm
            case 0x04: case 0x0C: case 0x14: case 0x1C:
            case 0x24: case 0x2C: case 0x34: case 0x3C:
            case 0xA8: case 0x6A: case 0xCD: case 0xE4: case 0xE5: case 0xE6: case 0xE7:
            case 0xEB: case 0xE0: case 0xE1: case 0xE2: case 0xE3:
            case 0x70: case 0x71: case 0x72: case 0x73:
            case 0x74: case 0x75: case 0x76: case 0x77:
            case 0x78: case 0x79: case 0x7A: case 0x7B:
            case 0x7C: case 0x7D: case 0x7E: case 0x7F:
                imm_size = 1;
                break;

            case 0x05: case 0x0D: case 0x15: case 0x1D:
            case 0x25: case 0x2D: case 0x35: case 0x3D:
            case 0xA9: case 0x68:
                imm_size = op_imm_word;
                break;

            case 0xE8: case 0xE9: // CALL / JMP rel32
                imm_size = 4;
                break;

            case 0x69: // IMUL r, r/m, imm32/16
                has_modrm = true;
                imm_size = op_imm_word;
                break;

            case 0x6B: // IMUL r, r/m, imm8
            case 0x80: case 0x82: case 0x83: // Group 1 r/m, imm8
            case 0xC0: case 0xC1: // shifts by imm8
            case 0xC6: // MOV r/m8, imm8
                has_modrm = true;
                imm_size = 1;
                break;

            case 0x81: // Group 1 r/m, imm32/16
            case 0xC7: // MOV r/m32/64, imm32 (or imm16 with 66)
                has_modrm = true;
                imm_size = op_imm_word;
                break;

            case 0xF6: // Group 3 byte (reg 0/1 = TEST has imm8)
                has_modrm = true;
                if (pos < max_len && (((code[pos] >> 3) & 7) <= 1)) imm_size = 1;
                break;

            case 0xF7: // Group 3 word/dword/qword (reg 0/1 = TEST has imm32/16)
                has_modrm = true;
                if (pos < max_len && (((code[pos] >> 3) & 7) <= 1)) imm_size = op_imm_word;
                break;

            case 0xA0: case 0xA1: case 0xA2: case 0xA3: // MOV moffs (64-bit addr unless 67)
                imm_size = pfx_67 ? 4 : 8;
                break;

            case 0xB0: case 0xB1: case 0xB2: case 0xB3:
            case 0xB4: case 0xB5: case 0xB6: case 0xB7:
                imm_size = 1;
                break;

            case 0xB8: case 0xB9: case 0xBA: case 0xBB:
            case 0xBC: case 0xBD: case 0xBE: case 0xBF:
                imm_size = rex_w ? 8 : op_imm_word;
                break;

            case 0xC2: case 0xCA: // RET imm16
                imm_size = 2;
                break;

            case 0xC8: // ENTER imm16, imm8
                imm_size = 3;
                break;

            default:
                // 1-byte ops without ModRM/imm: PUSH/POP reg (50..5F), NOP (90),
                // XCHG (91..97), CBW/CDQ (98/99), PUSHF/POPF (9C/9D), MOVS/STOS/LODS (A4..AF),
                // RET (C3), LEAVE (C9), INT3 (CC), IRET (CF), CLI/STI/CLD/STD (FA..FD), HLT (F4)
                has_modrm = false;
                break;
        }
    }

    if (has_modrm) {
        pos = decode_modrm(code, pos, max_len, pfx_67);
        if (pos < 0) return -1;
    }

    pos += imm_size;
    if (pos > max_len) return -1;
    return pos;
}
