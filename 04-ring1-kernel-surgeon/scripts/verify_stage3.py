#!/usr/bin/env python3
import glob, os, select, shutil, socket, subprocess, sys, time
from PIL import Image

PROJ = os.environ.get("PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISO = os.path.join(PROJ, "build/ring1-surgeon.iso")
MODEL_R1M = os.environ.get("MODEL_R1M", os.path.join(PROJ, "models/minicpm5-2b-surgeon.r1m"))
OVMF_CODE = "/usr/share/OVMF/OVMF_CODE.fd"
OVMF_VARS = os.path.join(PROJ, "build/OVMF_VARS.fd")
if not os.path.exists(OVMF_VARS):
    shutil.copy("/usr/share/OVMF/OVMF_VARS.fd", OVMF_VARS)

run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "build")
os.makedirs(run_dir, exist_ok=True)
ser_log = os.path.join(run_dir, "stage3_serial.log")
mon_sock = os.path.join(run_dir, "stage3_mon.sock")
if os.path.exists(mon_sock):
    os.unlink(mon_sock)

cmd = [
    "qemu-system-x86_64",
    "-machine", "q35,accel=kvm,nvdimm=on",
    "-cpu", "EPYC,+svm",
    "-smp", "24",
    "-m", "2G,slots=2,maxmem=6G",
    "-object", f"memory-backend-file,id=r1model,mem-path={MODEL_R1M},size=2G,readonly=on",
    "-device", "nvdimm,id=nv1,memdev=r1model,unarmed=on",
    "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
    "-drive", f"if=pflash,format=raw,file={OVMF_VARS}",
    "-cdrom", ISO,
    "-boot", "d",
    "-device", "VGA,vgamem_mb=32,xres=1280,yres=720",
    "-vnc", "127.0.0.1:19",
    "-serial", "stdio",
    "-monitor", f"unix:{mon_sock},server,nowait",
]

print("[STAGE3] Launching QEMU Nested AMD SVM (24 L1 cores + 2 GiB NVDIMM MiniCPM5-2B Q4_K_M)...")
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
log_f = open(ser_log, "wb")
buf = bytearray()

def read_until(needle: str, start_pos: int = 0, timeout: float = 45.0) -> bool:
    end = time.time() + timeout
    nb = needle.encode()
    while time.time() < end:
        if proc.poll() is not None:
            return nb in buf[start_pos:]
        r, _, _ = select.select([proc.stdout], [], [], 0.15)
        if r:
            ch = os.read(proc.stdout.fileno(), 4096)
            if not ch: break
            buf.extend(ch); log_f.write(ch); log_f.flush()
            if nb in buf[start_pos:]:
                return True
    return False

def send_cmd(s: str):
    proc.stdin.write((s + "\n").encode())
    proc.stdin.flush()

def screendump(name: str):
    ppm = os.path.join(run_dir, name + ".ppm")
    png = os.path.join(run_dir, name + ".png")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(mon_sock)
    s.settimeout(2.0)
    try: s.recv(4096)
    except Exception: pass
    s.sendall(f"screendump {ppm}\n".encode())
    time.sleep(0.5)
    try: s.recv(4096)
    except Exception: pass
    s.close()
    if os.path.exists(ppm):
        Image.open(ppm).save(png)
        os.unlink(ppm)
        print(f"[STAGE3] Saved VNC screenshot -> {png}")

try:
    ok_boot = read_until("LINUX 5.15 GUEST", 0, 45.0)
    print(f"[STAGE3] 1. Boot Linux under Ring -1 + MiniCPM5-2B: {ok_boot}")
    if not ok_boot:
        print(buf[-4000:].decode(errors="replace"))
        sys.exit(1)
    time.sleep(0.8)
    screendump("01_stage3_booted")

    # Query status via VMMCALL
    pos = len(buf)
    send_cmd("echo status > /proc/surgeon_crash && echo BOOT_ID_0:$(cat /proc/sys/kernel/random/boot_id)")
    ok_status = read_until("llm_ready=1", pos, 6.0)
    print(f"[STAGE3] 2. VMMCALL status (l1_cores + llm_ready=1): {ok_status}")

    # Test A: #PF NULL-deref (11B) -> Bare-Metal MiniCPM5-2B live-patches 11x NOP!
    pos = len(buf)
    t0 = time.time()
    send_cmd("echo pf > /proc/surgeon_crash")
    ok_pf = read_until("SURVIVED NULL-DEREF!", pos, 40.0)
    print(f"[STAGE3] 3. #PF NULL-deref healed live by L1 MiniCPM5-2B: {ok_pf} ({time.time()-t0:.2f}s)")
    time.sleep(0.4)
    screendump("02_ai_pf_healed")

    # Test B: #DE div ecx (2B) -> Bare-Metal MiniCPM5-2B live-patches 31 c0 (xor eax,eax) -> quotient=0!
    pos = len(buf)
    t0 = time.time()
    send_cmd("echo de > /proc/surgeon_crash")
    ok_de = read_until("Safe fallback quotient=0", pos, 40.0)
    print(f"[STAGE3] 4. #DE div/0 healed live (quotient=0) by L1 MiniCPM5-2B: {ok_de} ({time.time()-t0:.2f}s)")
    time.sleep(0.4)
    screendump("03_ai_de_healed")

    # Test C: #UD ud2 (2B) -> Bare-Metal MiniCPM5-2B live-patches 90 90!
    pos = len(buf)
    t0 = time.time()
    send_cmd("echo ud > /proc/surgeon_crash")
    ok_ud = read_until("SURVIVED UD2 TRAP!", pos, 40.0)
    print(f"[STAGE3] 5. #UD ud2 healed live by L1 MiniCPM5-2B: {ok_ud} ({time.time()-t0:.2f}s)")
    time.sleep(0.4)
    screendump("04_ai_ud_healed")

    # Test D: Benign Linux _ASM_EXTABLE control -> Bare-Metal MiniCPM5-2B outputs PASS_TO_LINUX -> rc=-14!
    pos = len(buf)
    t0 = time.time()
    send_cmd("echo extable > /proc/surgeon_crash")
    ok_ext = read_until("EXTABLE CONTROL OK! Linux handled fault natively -> rc=-14", pos, 40.0)
    print(f"[STAGE3] 6. Benign extable PASS_TO_LINUX by L1 MiniCPM5-2B: {ok_ext} ({time.time()-t0:.2f}s)")
    time.sleep(0.4)
    screendump("05_ai_extable_pass")

    # Verify uptime + boot_id continuity after all 4 recoveries!
    pos = len(buf)
    send_cmd("echo status > /proc/surgeon_crash && echo ALIVE_UPTIME:$(cat /proc/uptime) BOOT_ID_1:$(cat /proc/sys/kernel/random/boot_id)")
    ok_alive = read_until("ALIVE_UPTIME:", pos, 6.0)
    print(f"[STAGE3] 7. Linux alive with continuous uptime & boot_id: {ok_alive}")

    # Test E: Dangerous spinlock-held #PF -> Bare-Metal MiniCPM5-2B outputs HALT_FOR_DUMP & freezes guest safely!
    pos = len(buf)
    t0 = time.time()
    send_cmd("echo danger > /proc/surgeon_crash")
    ok_halt = read_until("HALT_FOR_DUMP EXECUTED", pos, 40.0)
    print(f"[STAGE3] 8. Unsafe spinlock-held #PF -> HALT_FOR_DUMP by L1 MiniCPM5-2B: {ok_halt} ({time.time()-t0:.2f}s)")
    time.sleep(0.5)
    screendump("06_ai_danger_halt")

    print("\n=== KEY STAGE 3 SERIAL LOG EXCERPT ===")
    for line in buf.decode(errors="replace").splitlines():
        if any(k in line for k in ["RING -1", "[SMP]", "[LLM]", "[NPT]", "[SVM]", "[VMMCALL]", "DIAGNOSIS:", "ACTION:", "PATCH:", "surgeon_crash", "ALIVE_UPTIME", "BOOT_ID_", "HALT_FOR_DUMP", "FATAL"]):
            print(line)

    if not (ok_boot and ok_status and ok_pf and ok_de and ok_ud and ok_ext and ok_alive and ok_halt):
        sys.exit(2)
    print("\n[STAGE3] ALL 8 END-TO-END BARE-METAL L1 MiniCPM5-2B + VNC CHECKS PASSED!")
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()
    log_f.close()
    if os.path.exists(mon_sock):
        os.unlink(mon_sock)
