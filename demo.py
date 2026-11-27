"""Synthetic data for `python3 server.py --demo`.

Fictional servers, users and jobs: lets you try the dashboard (or take screenshots) without
touching real machines. Nothing here talks to the network.
"""
import random
import time

GROUPS = [
    {"name": "vision-lab", "hosts": ["atlas", "borealis", "cosmos", "dune"]},
    {"name": "robotics", "hosts": ["echo", "falcon", "gemini", "helios", "ion"]},
]

# host: (gpu count, model, MiB per GPU, power limit W, CPU threads, RAM GiB, CPU model)
SPECS = {
    "atlas":    (4, "RTX A5000", 24564, 230, 64, 128, "AMD Ryzen Threadripper PRO 3975WX 32-Cores"),
    "borealis": (4, "RTX A5000", 24564, 230, 64, 256, "AMD Ryzen Threadripper PRO 3975WX 32-Cores"),
    "cosmos":   (4, "RTX A6000", 49140, 300, 64, 512, "AMD EPYC 7313 16-Core Processor"),
    "dune":     (2, "RTX 3090 Ti", 24564, 450, 64, 96, "AMD Ryzen Threadripper 3970X 32-Core Processor"),
    "echo":     (8, "A100-SXM4-80GB", 81920, 400, 128, 1024, "AMD EPYC 7763 64-Core Processor"),
    "falcon":   (8, "H100 80GB HBM3", 81559, 700, 192, 2048, "Intel(R) Xeon(R) Platinum 8468"),
    "gemini":   (8, "TITAN RTX", 24576, 280, 64, 512, "Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz"),
    "helios":   (4, "RTX 2080 Ti", 11264, 250, 48, 256, "Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz"),
}
OFFLINE = {"ion": "ssh: connect to host ion port 22: Connection timed out"}
FAULTY = {("gemini", 5)}  # shows how a GPU that nvidia-smi can't reach looks

USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"]
JOBS = [
    ("python train.py --config configs/vit_large.yaml --amp", 0.85),
    ("torchrun --nproc_per_node=4 train_ddp.py --batch-size 256", 0.97),
    ("python finetune.py --model llama-3-8b --lora-rank 16", 0.92),
    ("python eval.py --ckpt runs/exp42/best.pt --split val", 0.55),
    ("python -m diffusion.sample --steps 50 --guidance 7.5", 0.75),
    ("python rl/ppo.py --env Humanoid-v4 --num-envs 4096", 0.45),
    ("python nerf/train.py --scene lego --iters 200000", 0.88),
]


def _etime(secs):
    secs = int(secs)
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Host:
    def __init__(self, name, t0):
        self.name = name
        self.rng = random.Random(sum(map(ord, name)) * 7919)
        n, model, mem, plimit, threads, ram, cpu = SPECS[name]
        self.model, self.mem_total, self.plimit, self.threads, self.ram, self.cpumodel = model, mem, plimit, threads, ram, cpu
        self.uptime0 = t0 - self.rng.uniform(3, 60) * 86400
        self.pid = self.rng.randint(100000, 900000)
        self.gpus = [{"index": i, "job": None, "util": 0.0} for i in range(n)]
        TiB = 1024 ** 4
        self.disks = [{"fs": "/dev/nvme0n1p2", "type": "ext4", "mount": "/", "label": "/", "system": True,
                       "total": 1.7 * TiB, "used": 1.7 * TiB * self.rng.uniform(0.3, 0.7)}]
        for k in range(self.rng.choice([1, 2, 3, 4])):
            size = self.rng.choice([3.5, 7.3, 7.3, 14.0]) * TiB
            self.disks.append({"fs": f"/dev/sd{chr(98 + k)}", "type": "ext4", "mount": f"/data/add_disk{k}",
                               "label": f"/data/add_disk{k}", "system": False, "total": size,
                               "used": size * min(0.995, self.rng.uniform(0.45, 1.05))})
        for d in self.disks:
            d["avail"] = max(0.0, d["total"] * 0.98 - d["used"])
        for g in self.gpus:
            if self.rng.random() < 0.6:
                self._start_job(g, t0 - self.rng.uniform(600, 3 * 86400))

    def _start_job(self, g, t):
        cmd, intensity = self.rng.choice(JOBS)
        self.pid += self.rng.randint(1, 500)
        g["job"] = {"user": self.rng.choice(USERS), "cmd": cmd, "pid": str(self.pid), "start": t,
                    "intensity": intensity, "mem": self.mem_total * self.rng.uniform(0.35, 0.95)}

    def step(self, t):
        for g in self.gpus:
            if (self.name, g["index"]) in FAULTY:
                continue
            j = g["job"]
            if j and self.rng.random() < 0.0015:
                g["job"] = j = None
            elif not j and self.rng.random() < 0.002:
                self._start_job(g, t)
                j = g["job"]
            if j:
                target = 100 * j["intensity"] + self.rng.gauss(0, 6)
                if self.rng.random() < 0.08:  # data-loading stalls / eval phases
                    target *= self.rng.uniform(0.1, 0.6)
            else:
                target = 0
            g["util"] = max(0.0, min(100.0, 0.5 * g["util"] + 0.5 * target))

    def snapshot(self, t):
        gpus, top, who = [], [], {}
        busy = 0.0
        for g in self.gpus:
            i = g["index"]
            if (self.name, i) in FAULTY:
                gpus.append({"index": i, "uuid": "", "name": self.model, "fault": "Unknown Error", "util": None,
                             "mem_used": None, "mem_total": None, "temp": None, "power": None, "power_limit": None,
                             "fan": None, "clock": None, "clock_max": None, "procs": []})
                continue
            j, u = g["job"], round(g["util"])
            procs = []
            if j:
                procs.append({"pid": j["pid"], "mem": round(j["mem"]), "user": j["user"],
                              "etime": _etime(t - j["start"]), "cmd": j["cmd"]})
                top.append({"pid": j["pid"], "user": j["user"], "cpu": round(80 + u * 1.5 + self.rng.uniform(0, 40), 1),
                            "mem": round(self.rng.uniform(0.5, 4), 1), "etime": _etime(t - j["start"]), "cmd": j["cmd"]})
                who[j["user"]] = who.get(j["user"], 0) + 1
            busy += u
            temp = round(30 + u * 0.5 + self.rng.uniform(-1, 1))
            gpus.append({
                "index": i, "uuid": f"GPU-{self.name}-{i}", "name": self.model, "util": float(u),
                "mem_used": float(round(j["mem"]) if j else self.rng.choice([1, 4, 17])), "mem_total": float(self.mem_total),
                "temp": float(temp), "power": round(self.plimit * (0.07 + 0.85 * u / 100) + self.rng.uniform(-3, 3), 2),
                "power_limit": float(self.plimit), "fan": float(max(30, min(100, temp - 10))),
                "clock": float(210 if u < 3 else self.rng.randint(1500, 1900)), "clock_max": 2100.0, "procs": procs,
            })
        top += [{"pid": "1123", "user": "root", "cpu": 1.2, "mem": 0.1, "etime": _etime(t - self.uptime0), "cmd": "/usr/bin/containerd"},
                {"pid": "2841", "user": "root", "cpu": 0.6, "mem": 0.0, "etime": _etime(t - self.uptime0), "cmd": "[kworker/u256:2-events]"}]
        top.sort(key=lambda p: -p["cpu"])
        n_busy = sum(1 for g in self.gpus if g["job"])
        cpu = round(min(100, 1 + n_busy * 2.4 * 64 / self.threads + self.rng.uniform(0, 2)), 1)
        load1 = n_busy * 1.6 + self.rng.uniform(0.1, 0.8)
        ram = self.ram * 1073741824
        used = ram * min(0.9, 0.06 + n_busy * 0.06 + self.rng.uniform(0, 0.01))
        return {
            "ok": True, "error": None, "via": None, "cpu": cpu, "nproc": self.threads,
            "mem_total": ram, "mem_avail": ram - used, "swap_total": 8 * 1073741824, "swap_free": 7.2 * 1073741824,
            "load": [round(load1, 2), round(load1 * 0.95, 2), round(load1 * 0.9, 2)], "uptime": t - self.uptime0,
            "gpus": gpus, "top": top[:12],
            "disks": self.disks,
            "who": [{"user": u, "sessions": c} for u, c in sorted(who.items())],
            "info": {"os": "Ubuntu 24.04.1 LTS", "kernel": "6.8.0-45-generic", "cpumodel": self.cpumodel,
                     "driver": "550.120", "cuda": "12.4"},
            "latency_ms": self.rng.randint(40, 140), "updated": t,
        }


def run(state, history, lock, sample, interval, history_secs):
    """Pre-fill history_secs of samples, then keep producing live data forever."""
    now = time.time()
    hosts = {h: Host(h, now - history_secs) for h in SPECS}
    with lock:
        for h, err in OFFLINE.items():
            state[h].update({"ok": False, "error": err})
    t = now - history_secs
    while t < now:
        for h, host in hosts.items():
            host.step(t)
            history[h].append(sample(host.snapshot(t)))
        t += interval
    while True:
        t = time.time()
        with lock:
            for h, host in hosts.items():
                host.step(t)
                snap = host.snapshot(t)
                state[h].update(snap)
                history[h].append(sample(snap))
        time.sleep(interval)
