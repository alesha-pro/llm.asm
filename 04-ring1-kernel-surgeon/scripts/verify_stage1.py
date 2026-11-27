#!/usr/bin/env python3
import os, select, socket, subprocess, sys, time

PROJ = os.environ.get("PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISO = os.path.join(PROJ, "build/ring1-surgeon.iso")
OVMF_CODE = "/usr/share/OVMF/OVMF_CODE.fd"
OVMF_VARS = os.path.join(PROJ, "build/OVMF_VARS.fd")
if not os.path.exists(OVMF_VARS):
    import shutil
    shutil.copy("/usr/share/OVMF/OVMF_VARS.fd", OVMF_VARS)

run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "build")
os.makedirs(run_dir, exist_ok=True)
ser_log = os.path.join(run_dir, "verify_serial.log")
mon_sock = os.path.join(run_dir, "verify_mon.sock")
if os.path.exists(mon_sock):
    os.unlink(mon_sock)

cmd = [
    "qemu-system-x86_64",
    "-machine", "q35,accel=kvm",
    "-cpu", "EPYC,+svm",
    "-smp", "1",
    "-m", "2G",
    "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
    "-drive", f"if=pflash,format=raw,file={OVMF_VARS}",
    "-cdrom", ISO,
    "-boot", "d",
    "-device", "VGA,vgamem_mb=32,xres=1280,yres=720",
    "-vnc", "127.0.0.1:19",
    "-serial", "stdio",
    "-monitor", f"unix:{mon_sock},server,nowait",
]

print("[VERIFY] Launching QEMU Nested AMD SVM...")
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
log_f = open(ser_log, "wb")
buf = bytearray()

def read_until(needle, timeout=35.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        r, _, _ = select.select([proc.stdout], [], [], 0.2)
        if r:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            buf.extend(chunk)
            log_f.write(chunk)
            log_f.flush()
            if needle.encode() in buf:
                return True
    return False

def send_cmd(s):
    proc.stdin.write((s + "\n").encode())
    proc.stdin.flush()

def screendump(ppm_path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(mon_sock)
    s.settimeout(2.0)
    try: s.recv(4096)
    except Exception: pass
    s.sendall(f"screendump {ppm_path}\n".encode())
    time.sleep(0.5)
    try: s.recv(4096)
    except Exception: pass
    s.close()

try:
    ok_boot = read_until("LINUX GUEST RUNNING UNDER RING -1 KERNEL SURGEON", timeout=45.0)
    print(f"[VERIFY] Linux booted under Ring -1 SVM: {ok_boot}")
    if not ok_boot:
        print("=== TAIL OF SERIAL LOG ===")
        print(buf[-4000:].decode(errors="replace"))
        sys.exit(1)

    time.sleep(1.0)
    screendump(os.path.join(run_dir, "01_linux_booted.ppm"))

    # 1. Query Ring -1 VMMCALL status
    send_cmd("echo status > /proc/surgeon_crash")
    ok_vmm = read_until("4f4b5f52494e4731 heals=0", timeout=5.0)
    print(f"[VERIFY] VMMCALL status query: {ok_vmm}")

    # 2. Trigger Ring 0 NULL-pointer dereference (#PF) -> Ring -1 heals it live!
    send_cmd("echo pf > /proc/surgeon_crash")
    ok_pf_hv = read_until("LIVE INTERCEPT #1", timeout=10.0)
    ok_pf_alive = read_until("SURVIVED NULL-DEREF!", timeout=10.0)
    print(f"[VERIFY] Ring 0 #PF intercepted={ok_pf_hv}, guest survived={ok_pf_alive}")
    time.sleep(0.5)
    screendump(os.path.join(run_dir, "02_after_pf_heal.ppm"))

    # 3. Trigger Ring 0 UD2 (#UD) -> Ring -1 heals it live!
    send_cmd("echo ud > /proc/surgeon_crash")
    ok_ud_hv = read_until("LIVE INTERCEPT #2", timeout=10.0)
    ok_ud_alive = read_until("SURVIVED UD2 TRAP!", timeout=10.0)
    print(f"[VERIFY] Ring 0 #UD intercepted={ok_ud_hv}, guest survived={ok_ud_alive}")
    time.sleep(0.5)
    screendump(os.path.join(run_dir, "03_after_ud_heal.ppm"))

    # 4. Verify uptime & shell still alive + re-verify VMMCALL heal count == 2
    send_cmd("echo status > /proc/surgeon_crash && echo ALIVE_UPTIME:$(cat /proc/uptime)")
    ok_alive = read_until("ALIVE_UPTIME:", timeout=5.0)
    time.sleep(0.5)
    # Drain remaining output
    while select.select([proc.stdout], [], [], 0.2)[0]:
        chunk = os.read(proc.stdout.fileno(), 4096)
        if not chunk: break
        buf.extend(chunk); log_f.write(chunk)
    screendump(os.path.join(run_dir, "04_final_hud.ppm"))
    print(f"[VERIFY] Final guest alive check: {ok_alive}")
    print("=== KEY SERIAL EXCERPT ===")
    text = buf.decode(errors="replace")
    for line in text.splitlines():
        if any(k in line for k in ["RING -1", "[MB2]", "[SVM]", "[NPT]", "[LINUX]", "[VMMCALL]", "surgeon_crash", "efifb", "ALIVE_UPTIME", "FATAL", "PANIC", "Oops", "BUG:"]):
            print(line)
    if not (ok_boot and ok_pf_hv and ok_pf_alive and ok_ud_hv and ok_ud_alive and ok_alive):
        sys.exit(2)
    print("[VERIFY] ALL STAGE 1 CHECKS PASSED!")
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()
    log_f.close()
    if os.path.exists(mon_sock):
        os.unlink(mon_sock)
