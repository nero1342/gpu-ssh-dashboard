#!/usr/bin/env python3
"""GPU/CPU monitor for lab servers over SSH. Stdlib only.

Usage: python3 server.py              (hosts/groups from local/hosts.json; see hosts.example.json)
       python3 server.py host1 host2 …  (one ad-hoc group)
       python3 server.py --demo         (fictional servers and data, no SSH)
Then open http://127.0.0.1:8765

If a host fails and ~/.ssh/config has an "r<host>" alias (e.g. a ProxyJump through a gateway),
that alias is tried automatically.

Env: GPUMON_PORT (8765), GPUMON_INTERVAL seconds (3), GPUMON_HISTORY seconds kept (3600)
"""
import json
import re
import os
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))


DEMO = "--demo" in sys.argv[1:]


def load_groups():
    if DEMO:
        import demo
        return demo.GROUPS
    if sys.argv[1:]:
        return [{"name": "hosts", "hosts": sys.argv[1:]}]
    path = os.environ.get("GPUMON_HOSTS") or os.path.join(HERE, "local", "hosts.json")
    if not os.path.exists(path):
        sys.exit(
            f"gpumon: no host list found at {path}\n"
            "  Create it from the example:\n"
            "    mkdir -p local && cp hosts.example.json local/hosts.json\n"
            "  then put your own ~/.ssh/config host names in it,\n"
            "  or pass hosts directly:  python3 server.py host1 host2"
        )
    with open(path) as f:
        cfg = json.load(f)
    groups = [{"name": k, "hosts": v} for k, v in cfg.items() if not k.startswith("_") and v]
    if not groups:
        sys.exit(f"gpumon: {path} has no hosts")
    return groups


GROUPS = load_groups()
HOSTS = [h for g in GROUPS for h in g["hosts"]]
GROUP_OF = {h: g["name"] for g in GROUPS for h in g["hosts"]}
PORT = int(os.environ.get("GPUMON_PORT", "8765"))
# 127.0.0.1 = this machine only. Set e.g. a VPN address to share it; there is no login, and the page
# shows every user's process command lines, so avoid 0.0.0.0 on public networks.
BIND = os.environ.get("GPUMON_BIND", "127.0.0.1")
INTERVAL = float(os.environ.get("GPUMON_INTERVAL", "3"))
HISTORY_SECS = float(os.environ.get("GPUMON_HISTORY", "3600"))
# SSH connection-sharing sockets. Kept on local /tmp: some network/ACL home directories refuse the
# hard link ssh uses to create them ("muxserver_listen: link mux listener … Operation not permitted"),
# and socket paths must stay short (~104 chars on macOS).
CTL_DIR = os.environ.get("GPUMON_CTL_DIR") or f"/tmp/gpumon-{os.getuid()}"


def ensure_ctl_dir():
    os.makedirs(CTL_DIR, mode=0o700, exist_ok=True)
    st = os.stat(CTL_DIR)
    if st.st_uid != os.getuid():
        sys.exit(f"gpumon: {CTL_DIR} belongs to another user; set GPUMON_CTL_DIR to a private directory")
    os.chmod(CTL_DIR, 0o700)

GPU_FIELDS = ("index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
              "power.draw,power.limit,fan.speed,clocks.sm,clocks.max.sm")

REMOTE = r"""
echo '##CPU'; head -1 /proc/stat; sleep 0.5; head -1 /proc/stat
echo '##NPROC'; nproc
echo '##MEM'; grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
echo '##LOAD'; cat /proc/loadavg
echo '##UPTIME'; cut -d' ' -f1 /proc/uptime
echo '##GPU'; nvidia-smi --query-gpu=%s --format=csv,noheader,nounits 2>&1
echo '##APPS'; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null
echo '##PS'; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do ps -o pid=,user:32=,etime=,args= -p "$p" 2>/dev/null; done
echo '##TOP'; ps -eo pid=,user:32=,pcpu=,pmem=,etime=,args= --sort=-pcpu 2>/dev/null | head -12
echo '##DISK'; df -P -B1 / /home /data /scratch 2>/dev/null | tail -n +2
echo '##WHO'; who 2>/dev/null | awk '{print $1}' | sort | uniq -c
echo '##END'
""" % GPU_FIELDS

STATIC = r"""
echo '##OS'; (. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME")
echo '##KERNEL'; uname -r
echo '##CPUMODEL'; lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1
echo '##DRIVER'; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
echo '##CUDA'; nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}'
echo '##END'
"""

state = {h: {"host": h, "group": GROUP_OF[h], "ok": False, "error": "connecting…", "updated": None, "info": {}, "via": None}
         for h in HOSTS}
history = {h: deque(maxlen=max(10, int(HISTORY_SECS / INTERVAL))) for h in HOSTS}
lock = threading.Lock()


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def sections(out):
    sec, cur = {}, None
    for line in out.splitlines():
        if line.startswith("##"):
            cur = line[2:].strip()
            sec[cur] = []
        elif cur:
            sec[cur].append(line)
    if "END" not in sec:
        raise RuntimeError("incomplete output")
    return sec


def first(sec, key, default=""):
    v = sec.get(key) or [default]
    return v[0].strip()


def parse(out):
    sec = sections(out)

    cpu = None
    s = [list(map(int, l.split()[1:])) for l in sec.get("CPU", []) if l.startswith("cpu ")]
    if len(s) == 2:
        idle = lambda v: v[3] + (v[4] if len(v) > 4 else 0)
        dt = sum(s[1]) - sum(s[0])
        di = idle(s[1]) - idle(s[0])
        cpu = round(100.0 * (dt - di) / dt, 1) if dt > 0 else 0.0

    mem = {}
    for l in sec.get("MEM", []):
        k, v = l.split(":", 1)
        mem[k] = int(v.split()[0]) * 1024
    load = [num(x) for x in first(sec, "LOAD", "0 0 0").split()[:3]]

    procs = {}
    for l in sec.get("PS", []):
        parts = l.split(None, 3)
        if len(parts) >= 3:
            procs[parts[0]] = {"user": parts[1], "etime": parts[2], "cmd": parts[3] if len(parts) > 3 else ""}

    gpus, by_uuid = [], {}
    faults = []
    for l in sec.get("GPU", []):
        m = re.search(r"for GPU ?(\d+): (?:[0-9A-Fa-f:.]+: )?(.*)", l)
        if m:  # e.g. "Unable to determine the device handle for GPU4: 0000:3D:00.0: Unknown Error"
            faults.append(int(m.group(1)))
            gpus.append({"index": int(m.group(1)), "uuid": "", "name": "unavailable", "fault": m.group(2).strip() or "error",
                         "util": None, "mem_used": None, "mem_total": None, "temp": None, "power": None,
                         "power_limit": None, "fan": None, "clock": None, "clock_max": None, "procs": []})
            continue
        f = [x.strip() for x in l.split(",")]
        if len(f) < 9 or not f[0].isdigit():
            continue
        f += [""] * (12 - len(f))
        g = {
            "index": int(f[0]), "uuid": f[1], "name": f[2].replace("NVIDIA ", "").replace("GeForce ", ""),
            "util": num(f[3]), "mem_used": num(f[4]), "mem_total": num(f[5]),
            "temp": num(f[6]), "power": num(f[7]), "power_limit": num(f[8]),
            "fan": num(f[9]), "clock": num(f[10]), "clock_max": num(f[11]), "procs": [],
        }
        gpus.append(g)
        by_uuid[g["uuid"]] = g
    for l in sec.get("APPS", []):
        f = [x.strip() for x in l.split(",")]
        if len(f) >= 3 and f[0] in by_uuid:
            p = {"pid": f[1], "mem": num(f[2])}
            p.update(procs.get(f[1], {"user": "?", "etime": "", "cmd": ""}))
            by_uuid[f[0]]["procs"].append(p)

    top = []
    for l in sec.get("TOP", []):
        parts = l.split(None, 5)
        if len(parts) >= 5 and not (len(parts) > 5 and parts[5].startswith("ps -eo")):
            top.append({"pid": parts[0], "user": parts[1], "cpu": num(parts[2]), "mem": num(parts[3]),
                        "etime": parts[4], "cmd": parts[5] if len(parts) > 5 else ""})

    disks, seen = [], set()
    for l in sec.get("DISK", []):
        parts = l.split()
        if len(parts) >= 6 and parts[0] not in seen:
            seen.add(parts[0])
            disks.append({"fs": parts[0], "mount": parts[5], "total": num(parts[1]), "used": num(parts[2])})

    who = []
    for l in sec.get("WHO", []):
        parts = l.split()
        if len(parts) == 2:
            who.append({"user": parts[1], "sessions": int(parts[0])})

    gpus.sort(key=lambda g: g["index"])
    if faults and gpus:  # give faulty GPUs the model name of their siblings
        name = next((g["name"] for g in gpus if not g.get("fault")), "unavailable")
        for g in gpus:
            if g.get("fault"):
                g["name"] = name
    return {
        "ok": True, "error": None,
        "cpu": cpu, "nproc": int(first(sec, "NPROC", "0") or 0),
        "mem_total": mem.get("MemTotal"), "mem_avail": mem.get("MemAvailable"),
        "swap_total": mem.get("SwapTotal"), "swap_free": mem.get("SwapFree"),
        "load": load, "uptime": num(first(sec, "UPTIME", "")),
        "gpus": gpus, "top": top, "disks": disks, "who": who,
    }


def sample(d):
    mem_pct = None
    if d.get("mem_total"):
        mem_pct = round(100.0 * (1 - d["mem_avail"] / d["mem_total"]), 1)
    return {
        "t": round(d["updated"], 1), "cpu": d["cpu"], "mem": mem_pct, "load": d["load"][0],
        "g": [[g["util"], g["mem_used"], g["temp"], g["power"]] for g in d["gpus"]],
    }


def run_ssh(host, script, timeout):
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-o", "ControlMaster=auto", "-o", f"ControlPath={CTL_DIR}/%C",
        "-o", "ControlPersist=120", host, "bash -s",
    ]
    r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and "##END" not in r.stdout:
        raise RuntimeError((r.stderr.strip().splitlines() or ["ssh exit %d" % r.returncode])[-1])
    return r.stdout


def ssh_routes(host):
    """The host itself, plus its r<host> jump alias if ~/.ssh/config defines one."""
    routes = [host]
    alt = "r" + host
    try:
        out = subprocess.run(["ssh", "-G", alt], capture_output=True, text=True, timeout=5).stdout
        hn = next((l.split()[1] for l in out.splitlines() if l.startswith("hostname ")), alt)
        if hn != alt:
            routes.append(alt)
    except Exception:
        pass
    return routes


def poll(host):
    have_info = False
    routes, ri = ssh_routes(host), 0
    while True:
        t0 = time.time()
        route = routes[ri]
        try:
            data = parse(run_ssh(route, REMOTE, 25))
            data["via"] = None if route == host else route
            data["latency_ms"] = int((time.time() - t0) * 1000)
            data["updated"] = time.time()
            with lock:
                state[host].update(data)
                history[host].append(sample(data))
            if not have_info:
                sec = sections(run_ssh(route, STATIC, 25))
                info = {k.lower(): first(sec, k) for k in ("OS", "KERNEL", "CPUMODEL", "DRIVER", "CUDA")}
                with lock:
                    state[host]["info"] = info
                have_info = True
        except Exception as e:  # keep last good data, flag the error
            with lock:
                state[host].update({"ok": False, "error": str(e)[:200]})
            ri = (ri + 1) % len(routes)  # try the other route next time
        time.sleep(max(0.5, INTERVAL - (time.time() - t0)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/stats":
            with lock:
                self._json({"hosts": [state[h] for h in HOSTS], "groups": GROUPS, "interval": INTERVAL,
                            "history_secs": HISTORY_SECS, "now": time.time()})
        elif u.path == "/api/history":
            want = parse_qs(u.query).get("host") or HOSTS
            with lock:
                self._json({h: list(history[h]) for h in want if h in history})
        elif u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    if DEMO:
        import demo
        threading.Thread(target=demo.run, args=(state, history, lock, sample, INTERVAL, HISTORY_SECS), daemon=True).start()
        print("gpumon: demo mode, synthetic data, no SSH", flush=True)
    else:
        ensure_ctl_dir()
        for h in HOSTS:
            threading.Thread(target=poll, args=(h,), daemon=True).start()
    for g in GROUPS:
        print(f"gpumon [{g['name']}]: {', '.join(g['hosts'])}", flush=True)
    print(f"gpumon: http://{BIND}:{PORT}", flush=True)
    try:
        ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
