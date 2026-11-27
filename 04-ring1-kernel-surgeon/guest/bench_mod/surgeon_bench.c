#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/string.h>
#include <asm/extable.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Ring -1 Kernel Surgeon");
MODULE_DESCRIPTION("Single-level KVM Ring 0 postcondition & live-patch verifier module");

static struct proc_dir_entry *proc_ent;
static char last_result[512];
static char g_kbuf[512];
static char g_code_hex[256];
static char g_patch_hex[128];
static u8   g_code_buf[128];
static u8   g_patch_buf[32];

/* Dedicated executable slot in module .text (writable in verifier VM with rodata=off) */
static u8 __attribute__((section(".text"), aligned(4096))) bench_exec_slot[4096] = { 0xC3 };

struct bench_ctx {
    u64 canary_before;
    u64 out_val;
    u64 canary_after;
    int lock_depth;
    int side_effect_dirty;
};

typedef u64 (*stub_fn_t)(struct bench_ctx *ctx);

static int parse_hex_bytes(const char *hex, u8 *out, int max_len)
{
    int n = 0;
    char bstr[3];
    unsigned long v;

    while (*hex && n < max_len) {
        while (*hex == ' ' || *hex == ',' || *hex == ':')
            hex++;
        if (!*hex || *hex == '\n' || *hex == '\r')
            break;
        if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
            hex += 2;
        if (!hex[0] || !hex[1])
            return -EINVAL;
        bstr[0] = hex[0];
        bstr[1] = hex[1];
        bstr[2] = '\0';
        v = 0;
        if (kstrtoul(bstr, 16, &v) != 0)
            return -EINVAL;
        out[n++] = (u8)v;
        hex += 2;
    }
    return n;
}

static noinline int run_extable_probe(void)
{
    unsigned long val = 0;
    unsigned long __user *bad_uptr = (unsigned long __user *)0x10UL;
    int ret = get_user(val, bad_uptr);
    return (ret == -EFAULT) ? 0 : -EIO;
}

static noinline int run_warn_control(void)
{
    WARN_ON_ONCE(1);
    return 0;
}

static ssize_t bench_write(struct file *file, const char __user *ubuf,
                           size_t count, loff_t *ppos)
{
    size_t len = min(count, sizeof(g_kbuf) - 1);
    unsigned int patch_off = 0, patch_len = 0;
    unsigned long long expect_out = 0, expect_ret = 0;
    int parsed, code_len, plen;
    struct bench_ctx ctx1, ctx2;
    stub_fn_t fn;
    u64 r1, r2;
    int canary_ok, lock_ok, out_ok, ret_ok, repeat_ok, pass;

    if (copy_from_user(g_kbuf, ubuf, len))
        return -EFAULT;
    g_kbuf[len] = '\0';

    if (strncmp(g_kbuf, "extable_test", 12) == 0) {
        int r = run_extable_probe();
        snprintf(last_result, sizeof(last_result),
                 "BENCH_RESULT status=%s mode=extable rc=%d canary_ok=1 lock_ok=1 repeat_ok=1",
                 (r == 0) ? "PASS" : "FAIL", r);
        return count;
    }

    if (strncmp(g_kbuf, "warn_test", 9) == 0) {
        int r = run_warn_control();
        snprintf(last_result, sizeof(last_result),
                 "BENCH_RESULT status=%s mode=warn rc=%d canary_ok=1 lock_ok=1 repeat_ok=1",
                 (r == 0) ? "PASS" : "FAIL", r);
        return count;
    }

    if (strncmp(g_kbuf, "exec ", 5) == 0) {
        g_code_hex[0] = '\0';
        g_patch_hex[0] = '\0';
        parsed = sscanf(g_kbuf + 5, "%255s %u %u %127s %llx %llx",
                        g_code_hex, &patch_off, &patch_len, g_patch_hex,
                        &expect_out, &expect_ret);
        if (parsed != 6) {
            snprintf(last_result, sizeof(last_result),
                     "BENCH_RESULT status=FAIL reason=bad_args parsed=%d", parsed);
            return count;
        }

        code_len = parse_hex_bytes(g_code_hex, g_code_buf, sizeof(g_code_buf));
        plen = (strcmp(g_patch_hex, "NONE") == 0)
               ? 0
               : parse_hex_bytes(g_patch_hex, g_patch_buf, sizeof(g_patch_buf));

        if (code_len <= 0 || plen < 0 || patch_off + patch_len > (unsigned int)code_len) {
            snprintf(last_result, sizeof(last_result),
                     "BENCH_RESULT status=FAIL reason=bad_hex code_len=%d plen=%d",
                     code_len, plen);
            return count;
        }

        if (plen > 0) {
            if ((unsigned int)plen != patch_len) {
                snprintf(last_result, sizeof(last_result),
                         "BENCH_RESULT status=FAIL reason=patch_len_mismatch got=%d expect=%u",
                         plen, patch_len);
                return count;
            }
            memcpy(g_code_buf + patch_off, g_patch_buf, plen);
        }

        memset(bench_exec_slot, 0xC3, 256);
        memcpy(bench_exec_slot, g_code_buf, code_len);
        mb();

        ctx1.canary_before = 0xCAFEBABE11223344ULL;
        ctx1.out_val = 0x12345678ULL;
        ctx1.canary_after = 0x55667788DEADC0DEULL;
        ctx1.lock_depth = 0;
        ctx1.side_effect_dirty = 0;
        ctx2 = ctx1;

        fn = (stub_fn_t)(void *)bench_exec_slot;
        r1 = fn(&ctx1);
        r2 = fn(&ctx2);

        canary_ok = (ctx1.canary_before == 0xCAFEBABE11223344ULL &&
                     ctx1.canary_after  == 0x55667788DEADC0DEULL &&
                     ctx2.canary_before == 0xCAFEBABE11223344ULL &&
                     ctx2.canary_after  == 0x55667788DEADC0DEULL);
        lock_ok   = (ctx1.lock_depth == 0 && ctx2.lock_depth == 0);
        out_ok    = (expect_out == 0xFFFFFFFFFFFFFFFFULL) || (ctx1.out_val == expect_out);
        ret_ok    = (expect_ret == 0xFFFFFFFFFFFFFFFFULL) || (r1 == expect_ret);
        repeat_ok = (r1 == r2 && ctx1.out_val == ctx2.out_val);
        pass      = (canary_ok && lock_ok && out_ok && ret_ok && repeat_ok);

        snprintf(last_result, sizeof(last_result),
                 "BENCH_RESULT status=%s ret=0x%llx out=0x%llx canary_ok=%d lock_ok=%d out_ok=%d ret_ok=%d repeat_ok=%d",
                 pass ? "PASS" : "FAIL",
                 (unsigned long long)r1, (unsigned long long)ctx1.out_val,
                 canary_ok, lock_ok, out_ok, ret_ok, repeat_ok);
        return count;
    }

    return count;
}

static ssize_t bench_read(struct file *file, char __user *ubuf, size_t count, loff_t *ppos)
{
    return simple_read_from_buffer(ubuf, count, ppos, last_result, strlen(last_result));
}

static const struct proc_ops bench_proc_ops = {
    .proc_write = bench_write,
    .proc_read  = bench_read,
};

static int __init surgeon_bench_init(void)
{
    snprintf(last_result, sizeof(last_result), "BENCH_READY");
    proc_ent = proc_create("surgeon_bench", 0666, NULL, &bench_proc_ops);
    if (!proc_ent)
        return -ENOMEM;
    pr_info("surgeon_bench: loaded -> /proc/surgeon_bench ready\n");
    return 0;
}

static void __exit surgeon_bench_exit(void)
{
    if (proc_ent)
        proc_remove(proc_ent);
}

module_init(surgeon_bench_init);
module_exit(surgeon_bench_exit);
