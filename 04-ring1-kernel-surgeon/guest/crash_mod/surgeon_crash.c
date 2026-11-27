#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/spinlock.h>
#include <linux/string.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Ring -1 Kernel Surgeon");
MODULE_DESCRIPTION("Stage 3 Ring 0 #PF / #DE / #UD / extable / danger live demo harness");

static struct proc_dir_entry *proc_ent;
static DEFINE_SPINLOCK(demo_spinlock);

struct r1_ctx_hint {
    u32  magic;
    u32  lock_held;
    u32  irq_ctx;
    u32  side_effect;
    u32  extable;
    u32  bug_warn;
    u32  stack_ok;
    u32  obs_complete;
    char sym[48];
    char contract[192];
    char log_tail[96];
} __packed;

static struct r1_ctx_hint g_hint;

static bool ring1_present(void)
{
    unsigned int a = 0x40000000U, b = 0, c = 0, d = 0;
    asm volatile("cpuid" : "+a"(a), "=b"(b), "=c"(c), "=d"(d));
    return (b == 0x676E6952U && c == 0x72755331U && d == 0x6E6F6567U);
}

static noinline void redraw_ring1_hud(void);
static void send_hint(const struct r1_ctx_hint *h)
{
    unsigned long rax = 0x53555249UL;
    unsigned long rbx = (unsigned long)h;
    if (!ring1_present())
        return;
    asm volatile("vmmcall" : "+a"(rax) : "b"(rbx) : "memory");
}

static noinline void trigger_ring0_null_deref(void)
{
    volatile unsigned int *null_ptr = (volatile unsigned int *)0x0;
    memset(&g_hint, 0, sizeof(g_hint));
    g_hint.magic = 0x52314854U;
    g_hint.stack_ok = 1;
    g_hint.obs_complete = 1;
    strscpy(g_hint.sym, "surgeon_crash+0x0180", sizeof(g_hint.sym));
    strscpy(g_hint.contract,
            "Optional telemetry store to unmapped pointer before commit; skipping store preserves out=0x100 and returns 0.",
            sizeof(g_hint.contract));
    strscpy(g_hint.log_tail, "surgeon_crash: stage=pre_commit opt_ptr=NULL", sizeof(g_hint.log_tail));
    send_hint(&g_hint);

    pr_info("surgeon_crash: [RING 0] Executing NULL-pointer write *(u32 *)0 = 0xDEADBEEF ...\n");
    *null_ptr = 0xDEADBEEFU;
    pr_info("surgeon_crash: [RING 0] SURVIVED NULL-DEREF! Execution resumed cleanly in Ring 0!\n");
    redraw_ring1_hud();
}

static noinline void trigger_ring0_div_zero(void)
{
    unsigned int num = 0x2a, den = 0, quot = 0xdead;
    memset(&g_hint, 0, sizeof(g_hint));
    g_hint.magic = 0x52314854U;
    g_hint.stack_ok = 1;
    g_hint.obs_complete = 1;
    strscpy(g_hint.sym, "surgeon_crash+0x0210", sizeof(g_hint.sym));
    strscpy(g_hint.contract,
            "Integer division quotient in RAX/EAX is stored to out_val and returned; numerator=0x2a, divisor=0, safe fallback requires quotient=0 (out=0, ret=0).",
            sizeof(g_hint.contract));
    strscpy(g_hint.log_tail, "surgeon_crash: denom=0 fallback_quotient=0", sizeof(g_hint.log_tail));
    send_hint(&g_hint);

    pr_info("surgeon_crash: [RING 0] Executing divide-by-zero (0x2a / 0 via div ecx) ...\n");
    asm volatile(
        "xor %%edx, %%edx\n\t"
        "div %%ecx\n\t"
        : "=a"(quot)
        : "a"(num), "c"(den)
        : "edx", "cc", "memory"
    );
    pr_info("surgeon_crash: [RING 0] SURVIVED #DE DIV-BY-ZERO! Safe fallback quotient=%u (expected 0)\n", quot);
    redraw_ring1_hud();
}

static noinline void trigger_ring0_ud2(void)
{
    memset(&g_hint, 0, sizeof(g_hint));
    g_hint.magic = 0x52314854U;
    g_hint.stack_ok = 1;
    g_hint.obs_complete = 1;
    strscpy(g_hint.sym, "surgeon_crash+0x01ab", sizeof(g_hint.sym));
    strscpy(g_hint.contract,
            "Non-fatal diagnostic trap with EAX=0 and out_val=0x42 already prepared; skipping trap completes cleanly with ret=0.",
            sizeof(g_hint.contract));
    strscpy(g_hint.log_tail, "surgeon_crash: diag_trap state=clean pre_commit=1", sizeof(g_hint.log_tail));
    send_hint(&g_hint);

    pr_info("surgeon_crash: [RING 0] Executing UD2 invalid opcode in Ring 0 ...\n");
    asm volatile("xor %%eax, %%eax\n\t.byte 0x0f, 0x0b" ::: "eax", "memory");
    pr_info("surgeon_crash: [RING 0] SURVIVED UD2 TRAP! Execution resumed cleanly in Ring 0!\n");
    redraw_ring1_hud();
}

static noinline void trigger_extable_control(void)
{
    unsigned long val = 0;
    unsigned long __user *bad = (unsigned long __user *)0x10UL;
    int rc;
    memset(&g_hint, 0, sizeof(g_hint));
    g_hint.magic = 0x52314854U;
    g_hint.extable = 1;
    g_hint.stack_ok = 1;
    g_hint.obs_complete = 1;
    strscpy(g_hint.sym, "__get_user_8+0x0014", sizeof(g_hint.sym));
    strscpy(g_hint.contract,
            "User-pointer read covered by Linux _ASM_EXTABLE fixup (returns -EFAULT).",
            sizeof(g_hint.contract));
    strscpy(g_hint.log_tail, "linux: standard extable get_user probe on 0x10", sizeof(g_hint.log_tail));
    send_hint(&g_hint);

    pr_info("surgeon_crash: [RING 0] Executing benign Linux _ASM_EXTABLE get_user(0x10) ...\n");
    rc = get_user(val, bad);
    pr_info("surgeon_crash: [RING 0] EXTABLE CONTROL OK! Linux handled fault natively -> rc=%d (expected -14)\n", rc);
    redraw_ring1_hud();
}

static noinline void trigger_danger_lock_held(void)
{
    volatile unsigned int *null_ptr = (volatile unsigned int *)0x0;
    unsigned long flags;
    memset(&g_hint, 0, sizeof(g_hint));
    g_hint.magic = 0x52314854U;
    g_hint.lock_held = 1;
    g_hint.stack_ok = 1;
    g_hint.obs_complete = 1;
    strscpy(g_hint.sym, "kdrv_tx+0x0180", sizeof(g_hint.sym));
    strscpy(g_hint.contract,
            "Ring 0 NULL write while holding spinlock (lock_held=1).",
            sizeof(g_hint.contract));
    strscpy(g_hint.log_tail, "kdrv_tx: spin_lock held; hazard lock=1", sizeof(g_hint.log_tail));
    send_hint(&g_hint);

    pr_info("surgeon_crash: [RING 0] Acquiring spinlock and triggering UNSAFE NULL deref (expect HALT_FOR_DUMP) ...\n");
    spin_lock_irqsave(&demo_spinlock, flags);
    *null_ptr = 0xDEADBEEFU;
    spin_unlock_irqrestore(&demo_spinlock, flags);
    pr_info("surgeon_crash: [RING 0] WARNING: returned from unsafe fault!\n");
}

static noinline void query_ring1_vmmcall(void)
{
    unsigned long rax = 0x53555247UL;
    unsigned long rbx = 0, rcx = 0, rdx = 0;
    if (!ring1_present()) {
        pr_info("surgeon_crash: [STATUS] Running WITHOUT Ring -1 hypervisor (vanilla Linux)\n");
        return;
    }
    asm volatile("vmmcall" : "+a"(rax), "=b"(rbx), "=c"(rcx), "=d"(rdx) :: "memory");
    pr_info("surgeon_crash: [VMMCALL] Ring -1 sig=0x%016lx heals=%lu l1_cores=%lu llm_ready=%lu\n",
            rax, rbx, rcx, rdx);
}

static noinline void redraw_ring1_hud(void)
{
    unsigned long rax = 0x53555248UL;
    if (!ring1_present()) return;
    asm volatile("vmmcall" : "+a"(rax) :: "memory");
}

static ssize_t surgeon_write(struct file *file, const char __user *ubuf,
                             size_t count, loff_t *ppos)
{
    char kbuf[32];
    size_t len = min(count, sizeof(kbuf) - 1);
    if (copy_from_user(kbuf, ubuf, len))
        return -EFAULT;
    kbuf[len] = '\0';

    if (strncmp(kbuf, "pf", 2) == 0) trigger_ring0_null_deref();
    else if (strncmp(kbuf, "de", 2) == 0) trigger_ring0_div_zero();
    else if (strncmp(kbuf, "ud", 2) == 0) trigger_ring0_ud2();
    else if (strncmp(kbuf, "extable", 7) == 0) trigger_extable_control();
    else if (strncmp(kbuf, "danger", 6) == 0) trigger_danger_lock_held();
    else if (strncmp(kbuf, "status", 6) == 0) query_ring1_vmmcall();
    else if (strncmp(kbuf, "hud", 3) == 0) redraw_ring1_hud();
    else pr_info("surgeon_crash: commands: pf | de | ud | extable | danger | status | hud\n");
    return count;
}

static const struct proc_ops surgeon_proc_ops = {
    .proc_write = surgeon_write,
};

static int __init surgeon_crash_init(void)
{
    proc_ent = proc_create("surgeon_crash", 0222, NULL, &surgeon_proc_ops);
    if (!proc_ent) return -ENOMEM;
    pr_info("surgeon_crash: loaded -> /proc/surgeon_crash (pf | de | ud | extable | danger | status | hud)\n");
    return 0;
}

static void __exit surgeon_crash_exit(void)
{
    if (proc_ent) proc_remove(proc_ent);
}

module_init(surgeon_crash_init);
module_exit(surgeon_crash_exit);
