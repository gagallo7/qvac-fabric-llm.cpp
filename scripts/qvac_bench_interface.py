"""Interface between qvac-bench.py and the qvac-bench-monitor.py sidecar.

Hosts the sidecar client, the monitored subprocess runner and the capture of
run metadata (driver invocation, executed command lines) so the bench driver
only deals with benchmark logic.
"""

import datetime
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OptionsType = dict[str, Any]

MONITOR_SCRIPT: Path = Path(__file__).resolve().with_name("qvac-bench-monitor.py")


def log(label: str, msg: str) -> None:
    logger.info("[%s] %s", label, msg)


class MonitorClient:
    """Drives the qvac-bench-monitor.py sidecar over its stdin command protocol.

    Every method is best-effort: a missing or crashed monitor must never fail the sweep.
    """

    def __init__(self, enabled: bool, interval: float, gpu_source: str = "auto") -> None:
        self.enabled = enabled
        self.interval = interval
        self.gpu_source = gpu_source
        self.proc: subprocess.Popen | None = None
        self.dead = False

    @property
    def active(self) -> bool:
        return self.proc is not None and not self.dead

    @staticmethod
    def _gpus_cli(gpus: Any) -> list[str]:
        if gpus is None:
            return []
        text = str(gpus).strip()
        if not text:
            return []
        return ["--gpus", text]

    def start(self, path: Path, phase: str, gpus: Any = None, meta: "OptionsType | None" = None) -> None:
        if not self.enabled:
            return
        try:
            self.proc = subprocess.Popen(
                [sys.executable, str(MONITOR_SCRIPT),
                 "--output", str(path), "--interval", str(self.interval),
                 "--gpu-source", self.gpu_source,
                 *self._gpus_cli(gpus),
                 *(["--meta", json.dumps(meta)] if meta else []),
                 "--control-stdin", "--phase", phase],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            log("monitor", f"monitoring CPU/GPU every {self.interval}s into {path.name}")
        except OSError as e:
            self.proc = None
            log("monitor", f"failed to start monitor, continuing without it: {e}")

    def rotate(self, path: Path, run: "OptionsType | None", phase: str,
               events_path: "Path | None" = None, gpus: Any = None) -> None:
        payload: OptionsType = {
            "cmd": "rotate",
            "path": str(path),
            "events_path": str(events_path) if events_path is not None else None,
            "phase": phase,
            "run": run,
        }
        if gpus is not None:
            payload["gpus"] = gpus
        self._send(payload)

    def phase(self, label: str) -> None:
        self._send({"cmd": "phase", "phase": label})

    def run_start(self) -> None:
        self._send_timed("run_start")

    def run_end(self) -> None:
        self._send_timed("run_end")

    def watch(self, pid: int | None, argv: "Sequence[str] | None" = None) -> None:
        """Point the monitor's per-process (pidstat-style) sampling at pid."""
        payload: OptionsType = {"cmd": "watch", "pid": pid}
        if pid is not None and argv is not None:
            payload["argv"] = list(argv)
        self._send(payload)

    def _send_timed(self, command: str) -> None:
        now = datetime.datetime.now(datetime.timezone.utc)
        self._send({"cmd": command, "ts": now.isoformat(), "t": now.timestamp()})

    def _send(self, obj: OptionsType) -> None:
        if self.proc is None or self.dead:
            return
        if self.proc.poll() is not None:
            self.dead = True
            log("monitor", "monitor process exited unexpectedly, continuing without monitoring")
            return
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            self.dead = True
            log("monitor", f"failed to send command to monitor, continuing without monitoring: {e}")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        except OSError as e:
            log("monitor", f"failed to stop monitor: {e}")
        finally:
            self.proc = None


MONITOR: "MonitorClient | None" = None
RUN_COMMANDS: "list[list[str]] | None" = None


def set_monitor(monitor: "MonitorClient | None") -> None:
    """Set the monitor client that monitored_run() reports child processes to."""
    global MONITOR
    MONITOR = monitor


def set_run_commands(commands: "list[list[str]] | None") -> None:
    """Set the list monitored_run() appends each executed argv to (None disables)."""
    global RUN_COMMANDS
    RUN_COMMANDS = commands


def build_invocation(args, raw_config: "OptionsType | None", config: OptionsType) -> OptionsType:
    """Describe how the driver was invoked: python argv, config file path and
    contents, and the merged effective config."""
    return {
        "argv": sys.argv,
        "config_path": args.config,
        "raw_config": raw_config if args.config else None,
        "config": config,
    }


def monitored_run(args, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run(check=True) replacement that reports the child PID and
    command line to the monitor sidecar so it can sample per-process
    (pidstat-style) statistics and log what was executed."""
    check = kwargs.pop("check", False)
    argv = [str(a) for a in args]
    if RUN_COMMANDS is not None:
        RUN_COMMANDS.append(argv)
    with subprocess.Popen(args, **kwargs) as proc:
        if MONITOR is not None:
            MONITOR.watch(proc.pid, argv)
        try:
            returncode = proc.wait()
        except BaseException:
            proc.kill()
            raise
        finally:
            if MONITOR is not None:
                MONITOR.watch(None)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, args)
    return subprocess.CompletedProcess(args, returncode)
