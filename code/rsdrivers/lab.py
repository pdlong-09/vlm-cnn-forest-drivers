"""Minimal ssh/scp to the lab server for this project (own copy, so this project imports nothing from src/).

The password comes from the LAB_PW environment variable (or a prompt), is passed to ssh only through the child
process environment via SSH_ASKPASS, and is never written to disk; the askpass helper holds no secret.
"""
from __future__ import annotations

import getpass
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HOST = "lab"
GIT_BIN = Path(os.environ.get("GIT_USR_BIN", r"C:\Program Files\Git\usr\bin"))


class Lab:
    def __init__(self, host: str = HOST):
        """host "lab" (server1): password from LAB_PW; any other ssh-config host (e.g. "server2", design notes): key only."""
        self.host = host
        self.tmp = Path(tempfile.mkdtemp(prefix="labA_"))
        if host != HOST:
            self.env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
            return
        pw = os.environ.get("LAB_PW") or getpass.getpass("server1 password: ")
        ask = self.tmp / "askpass.sh"
        ask.write_text('#!/bin/sh\nprintf "%s\\n" "$LAB_PW"\n', encoding="utf-8")
        self.env = {**os.environ, "LAB_PW": pw, "SSH_ASKPASS": str(ask).replace("\\", "/"),
                    "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": "dummy", "MSYS_NO_PATHCONV": "1"}

    def _run(self, args, timeout):
        try:
            r = subprocess.run(args, env=self.env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"{args[0]} timed out after {timeout} s") from e
        if r.returncode != 0:
            raise RuntimeError(f"{Path(args[0]).name} failed ({r.returncode}): {r.stderr.strip()[-400:]}")
        return r.stdout

    def retry(self, fn, what, tries=5):
        for k in range(tries):
            try:
                return fn()
            except RuntimeError as e:
                if k == tries - 1 or "No such file" in str(e):
                    raise
                print(f"{what}: attempt {k + 1} failed ({str(e)[:80]}); retrying", flush=True)
                time.sleep(min(15 * (k + 1), 120))  # capped (24/10/2026: long bore outages)

    def ssh(self, cmd, timeout=300):
        return self._run([str(GIT_BIN / "ssh.exe"), "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", self.host, cmd],
                         timeout)

    def put(self, local: Path, remote: str, timeout=900, tries=5):
        self.retry(lambda: self._run([str(GIT_BIN / "scp.exe"), "-q", str(local).replace("\\", "/"), f"{self.host}:{remote}"],
                                     timeout), f"put {Path(local).name}", tries)

    def get(self, remote: str, local: Path, timeout=900):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.retry(lambda: self._run([str(GIT_BIN / "scp.exe"), "-q", f"{self.host}:{remote}", str(local).replace("\\", "/")],
                                     timeout), f"get {remote.rsplit('/', 1)[-1]}")

    def put_chunked(self, local: Path, remote: str, chunk_mb=4, workers=4):
        """Upload in parts with retries, join on the server and verify the SHA-256 (the tunnel sometimes drops).

        Parts go up `workers` at a time (22/10/2026: one scp over bore reached only ~0.2 MB/s). Transport only: the
        joined file is checked against the local SHA-256 and the part count before anything uses it.
        Streamed (23/10/2026, local RAM care): the file is never held in memory, only `workers` parts at a time."""
        from concurrent.futures import ThreadPoolExecutor
        step = chunk_mb * 2 ** 20
        size = Path(local).stat().st_size
        h = hashlib.sha256()
        with open(local, "rb") as fh:
            for block in iter(lambda: fh.read(8 * 2 ** 20), b""):
                h.update(block)
        digest = h.hexdigest()
        offsets = list(range(0, size, step))

        def send(n):
            part = self.tmp / f"{Path(local).name}.part{n:04d}"
            with open(local, "rb") as fh:
                fh.seek(offsets[n])
                part.write_bytes(fh.read(step))
            try:
                self.put(part, f"{remote}.part{n:04d}", timeout=600, tries=200)  # 24/10/2026: bore outages of over 45 min
            finally:
                part.unlink(missing_ok=True)
            return f"{remote}.part{n:04d}"

        with ThreadPoolExecutor(max(1, workers)) as pool:
            parts = list(pool.map(send, range(len(offsets))))
        # join with a glob: listing hundreds of part names made the command too long (truncated, 22/10/2026)
        n_expected = len(parts)
        got = self.retry(lambda: self.ssh(
            f"n=$(ls {remote}.part* | wc -l); [ $n -eq {n_expected} ] || {{ echo parts:$n; exit 3; }}; "
            f"cat {remote}.part* > {remote} && sha256sum {remote} | cut -d' ' -f1"), "join").strip()
        if got != digest:
            raise RuntimeError(f"checksum mismatch after upload of {Path(local).name}: {got[:40]}")
        self.ssh(f"rm -f {remote}.part*")

    def get_chunked(self, remote: str, local: Path, chunk_mb=8, workers=4):
        """Download in parts with retries, join locally and verify the SHA-256 (24/10/2026: a single scp of a large
        result file starts again from zero whenever bore drops, and the tunnel was down for over 45 min that evening)."""
        from concurrent.futures import ThreadPoolExecutor
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        out = self.retry(lambda: self.ssh(f"rm -f {remote}.gpart*; split -b {chunk_mb}M -d -a 4 {remote} {remote}.gpart && "
                                          f"sha256sum {remote} | cut -d' ' -f1 && ls {remote}.gpart* | wc -l", timeout=900),
                         "split").split()
        digest, n = out[0], int(out[1])
        parts = self.tmp / f"{local.name}.parts"
        parts.mkdir(exist_ok=True)

        def fetch(i):
            dst = parts / f"{i:04d}"
            self.retry(lambda: self._run([str(GIT_BIN / "scp.exe"), "-q", f"{self.host}:{remote}.gpart{i:04d}",
                                          str(dst).replace("\\", "/")], 900), f"get part {i:04d}", tries=200)
            return dst
        with ThreadPoolExecutor(max(1, workers)) as pool:
            files = list(pool.map(fetch, range(n)))
        h = hashlib.sha256()
        with open(local, "wb") as fo:
            for f in files:
                with open(f, "rb") as fh:
                    for block in iter(lambda: fh.read(8 * 2 ** 20), b""):
                        h.update(block)
                        fo.write(block)
                f.unlink()
        if h.hexdigest() != digest:
            raise RuntimeError(f"checksum mismatch after download of {remote}")
        self.ssh(f"rm -f {remote}.gpart*")

    def run_detached(self, workdir: str, cmd: str, marker: str, poll_s: int = 60, timeout_h: float = 12) -> int:
        """Start `cmd` on the server with nohup (so a dropped bore connection does not kill it) and poll the exit
        code written to `marker`. 22/10/2026: a 4-hour ssh session over bore was cut and killed the M3b run."""
        if "'" in cmd:
            raise ValueError("cmd must not contain single quotes")
        pidfile = marker + ".pid"
        # idempotent start (22/10/2026): an ssh drop after the job had started made a blind retry launch it twice;
        # the second copy hit OOM, wrote its exit code, and the cleanup then deleted the inputs of the first
        start = (f"cd {workdir} && if [ -e {pidfile} ]; then echo ALREADY; else rm -f {marker}; "
                 f"nohup bash -c 'echo $$ > {pidfile}; {cmd}; echo $? > {marker}' > /dev/null 2>&1 & echo STARTED; fi")
        for attempt in range(6):
            try:
                print("start:", self.ssh(start, timeout=120).strip(), flush=True)
                break
            except RuntimeError as e:
                time.sleep(15 * (attempt + 1))
                try:  # did the dropped call start the job anyway?
                    if self.ssh(f"test -e {pidfile} && echo yes || echo no", timeout=60).strip() == "yes":
                        print("start: job was started before the connection dropped", flush=True)
                        break
                except RuntimeError:
                    pass
                print(f"start attempt {attempt + 1} failed ({str(e)[:60]}); retrying", flush=True)
        else:
            raise RuntimeError("could not start the detached job")
        t0 = time.time()
        while time.time() - t0 < timeout_h * 3600:
            time.sleep(poll_s)
            try:
                out = self.ssh(f"cat {marker} 2>/dev/null || echo RUNNING", timeout=120).strip()
            except RuntimeError as e:  # connection dropped while polling: the job keeps running
                print(f"poll failed ({str(e)[:60]}); retrying", flush=True)
                continue
            if out != "RUNNING":
                return int(out.splitlines()[-1])
        raise RuntimeError(f"detached job did not finish within {timeout_h} h")

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def pick_gpu(lab: Lab, max_used_mib: int = 2000, prefer=(0,)) -> int:
    """GPU 0 by default, if less than max_used_mib is in use.
    Never touches other users' processes: if both are busy, stop."""
    out = lab.ssh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits")
    used = {int(i): int(m) for i, m in (l.split(",") for l in out.strip().splitlines())}
    print("GPU memory used (MiB):", used, flush=True)
    for g in prefer:
        if used.get(g, 10 ** 9) < max_used_mib:
            return g
    raise SystemExit(f"GPU {prefer} busy; not starting")


def wait_gpu(lab: Lab, hours: float = 12, every_s: int = 300, max_used_mib: int = 2000) -> int:
    """pick_gpu, but wait (polling) while GPU 0 is used by others, up to `hours`."""
    t0 = time.time()
    while True:
        try:
            return pick_gpu(lab, max_used_mib)
        except SystemExit:
            if time.time() - t0 > hours * 3600:
                raise
            print(f"GPU 0 busy; waiting {every_s} s", flush=True)
            time.sleep(every_s)
