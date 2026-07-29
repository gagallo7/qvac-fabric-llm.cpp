import argparse
import ctypes
import datetime
import glob
import json
import logging
import os
import platform
import re
import selectors
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any, TextIO

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_warned: set[str] = set()


def log(msg: str) -> None:
    logger.info("[monitor] %s", msg)


def warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        log(msg)


def timestamps() -> dict[str, Any]:
    now = datetime.datetime.now(datetime.timezone.utc)
    return {"ts": now.isoformat(), "t": now.timestamp()}


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        # amd-smi wraps values as {"value": 63, "unit": "C"} or {"clk": {...}}
        for key in ("value", "clk", "avg_clk"):
            if key in value:
                return _num(value[key])
        return None
    if isinstance(value, str):
        m = _NUM_RE.search(value)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def parse_gpu_indices(value: Any) -> list[int] | None:
    """Parse ggml_vk_visible_devices-style GPU indices. None means all GPUs."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, float) and value == int(value):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            parsed = parse_gpu_indices(item)
            if parsed:
                out.extend(parsed)
        return out
    if isinstance(value, str):
        if not value.strip():
            return None
        out = []
        for part in value.split(","):
            part = part.strip()
            if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
                out.append(int(part))
        return out
    return None


def select_gpus(gpus: list[dict[str, Any]], indices: "set[int] | None") -> list[dict[str, Any]]:
    if indices is None:
        return gpus
    return [gpu for gpu in gpus if gpu.get("index") in indices]


_AMD_THROTTLE_BITS = {
    0: ("PPT0", "power"),
    1: ("PPT1", "power"),
    2: ("PPT2", "power"),
    3: ("PPT3", "power"),
    4: ("SPL", "power"),
    5: ("FPPT", "power"),
    6: ("SPPT", "power"),
    7: ("SPPT_APU", "power"),
    16: ("TDC_GFX", "current"),
    17: ("TDC_SOC", "current"),
    18: ("TDC_MEM", "current"),
    19: ("TDC_VDD", "current"),
    20: ("TDC_CVIP", "current"),
    21: ("EDC_CPU", "current"),
    22: ("EDC_GFX", "current"),
    23: ("APCC", "current"),
    32: ("TEMP_GPU", "thermal"),
    33: ("TEMP_CORE", "thermal"),
    34: ("TEMP_MEM", "thermal"),
    35: ("TEMP_EDGE", "thermal"),
    36: ("TEMP_HOTSPOT", "thermal"),
    37: ("TEMP_SOC", "thermal"),
    38: ("TEMP_VR_GFX", "thermal"),
    39: ("TEMP_VR_SOC", "thermal"),
    40: ("TEMP_VR_MEM0", "thermal"),
    41: ("TEMP_VR_MEM1", "thermal"),
    42: ("TEMP_LIQUID0", "thermal"),
    43: ("TEMP_LIQUID1", "thermal"),
    44: ("VRHOT0", "thermal"),
    45: ("VRHOT1", "thermal"),
    46: ("PROCHOT_CPU", "thermal"),
    47: ("PROCHOT_GFX", "thermal"),
    56: ("PPM", "other"),
    57: ("FIT", "other"),
}

_NVIDIA_THROTTLE_BITS = {
    1: ("APPLICATIONS_CLOCKS_SETTING", "configuration"),
    2: ("SW_POWER_CAP", "power"),
    3: ("HW_SLOWDOWN", "other"),
    4: ("SYNC_BOOST", "configuration"),
    5: ("SW_THERMAL_SLOWDOWN", "thermal"),
    6: ("HW_THERMAL_SLOWDOWN", "thermal"),
    7: ("HW_POWER_BRAKE_SLOWDOWN", "power"),
    8: ("DISPLAY_CLOCK_SETTING", "configuration"),
}


def _decode_throttle_bits(mask: int, definitions: dict[int, tuple[str, str]]) -> list[dict[str, Any]]:
    reasons = []
    for bit in range(mask.bit_length()):
        if not mask & (1 << bit):
            continue
        name, category = definitions.get(bit, (f"BIT_{bit}", "unknown"))
        reasons.append({"name": name, "category": category, "bit": bit})
    return reasons


def amd_throttle_reasons(status: int | None, indep_status: int | None) -> "list[dict[str, Any]] | None":
    if status == 0xFFFFFFFF:
        status = None
    if indep_status == 0xFFFFFFFFFFFFFFFF:
        indep_status = None
    if indep_status is not None:
        reasons = _decode_throttle_bits(indep_status, _AMD_THROTTLE_BITS)
        if reasons or not status:
            return reasons
    if status is None:
        return None
    if status:
        return [{"name": "UNKNOWN", "category": "unknown", "bit": None}]
    return []


def nvidia_throttle_reasons(mask: int | None) -> "list[dict[str, Any]] | None":
    if mask is None:
        return None
    # Bit 0 means that the GPU is idle, not that a workload is being throttled.
    return _decode_throttle_bits(mask & ~1, _NVIDIA_THROTTLE_BITS)


def _int_value(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("value", "status"):
            if key in value:
                return _int_value(value[key])
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value.upper() in ("THROTTLED", "ACTIVE", "YES", "TRUE"):
            return 1
        if value.upper() in ("UNTHROTTLED", "NOT ACTIVE", "NO", "FALSE"):
            return 0
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _read_str(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: str) -> int | None:
    value = _read_str(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _milli_c(value: int | None) -> float | None:
    return round(value / 1000.0, 1) if value is not None else None


def _hz_mhz(value: int | None) -> float | None:
    return round(value / 1e6, 1) if value is not None else None


def _bytes_mib(value: int | None) -> float | None:
    return round(value / 2**20, 1) if value is not None else None


def _set_io_idle() -> bool:
    nr = {"x86_64": 251, "aarch64": 30}.get(platform.machine())
    if nr is None:
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        # ioprio_set(IOPRIO_WHO_PROCESS, self, IOPRIO_PRIO_VALUE(IOPRIO_CLASS_IDLE, 0))
        return libc.syscall(nr, 1, 0, 3 << 13) == 0
    except OSError:
        return False


def deprioritize(pin_cpu: int | None) -> None:
    """Confine the monitor to one logical CPU at the lowest scheduling and I/O
    priority (taskset/nice/ionice equivalents) so it cannot perturb the benchmark."""
    applied = []
    if pin_cpu is None or pin_cpu >= 0:
        try:
            cpu = pin_cpu if pin_cpu is not None else max(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cpu})
            applied.append(f"cpu{cpu}")
        except (OSError, ValueError) as e:
            log(f"cpu pinning failed: {e}")
    try:
        os.nice(19)
        applied.append("nice 19")
    except OSError as e:
        log(f"nice failed: {e}")
    if _set_io_idle():
        applied.append("ionice idle")
    if applied:
        log("deprioritized: " + ", ".join(applied))


_CPU_HWMON_CHIPS = ("k10temp", "zenpower", "coretemp", "cpu_thermal")


def discover_cpu_temps() -> dict[str, str]:
    """Map temperature label (e.g. k10temp Tctl/Tccd*) to its hwmon input path."""
    inputs: dict[str, str] = {}
    for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        chip = _read_str(os.path.join(hwmon, "name"))
        if chip not in _CPU_HWMON_CHIPS:
            continue
        for temp_input in sorted(glob.glob(os.path.join(hwmon, "temp[0-9]*_input"))):
            label = _read_str(temp_input.replace("_input", "_label"))
            base = label or os.path.basename(temp_input).removesuffix("_input")
            key, n = base, 2
            while key in inputs:
                key = f"{chip}:{base}" if n == 2 else f"{chip}:{base}:{n}"
                n += 1
            inputs[key] = temp_input
    return inputs


class CpuSampler:
    def __init__(self) -> None:
        self.prev: tuple[int, int] | None = None
        self.temp_inputs = discover_cpu_temps()

    def sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "util_pct": None,
            "load1": None,
            "load5": None,
            "load15": None,
            "mem_total_kib": None,
            "mem_available_kib": None,
            "temps_c": None,
        }

        try:
            with open("/proc/stat") as f:
                fields = f.readline().split()
            values = [int(v) for v in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
            total = sum(values)
            if self.prev is not None:
                d_idle = idle - self.prev[0]
                d_total = total - self.prev[1]
                if d_total > 0:
                    result["util_pct"] = round(100.0 * (1.0 - d_idle / d_total), 2)
            self.prev = (idle, total)
        except (OSError, ValueError, IndexError):
            warn_once("cannot read /proc/stat, cpu utilization unavailable")

        try:
            result["load1"], result["load5"], result["load15"] = os.getloadavg()
        except OSError:
            pass

        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        result["mem_total_kib"] = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        result["mem_available_kib"] = int(line.split()[1])
        except (OSError, ValueError, IndexError):
            warn_once("cannot read /proc/meminfo, memory info unavailable")

        temps = {key: _milli_c(_read_int(path)) for key, path in self.temp_inputs.items()}
        result["temps_c"] = {k: v for k, v in temps.items() if v is not None} or None

        return result


def sample_pressure() -> dict[str, Any] | None:
    """Stall percentages and totals from the PSI files under /proc/pressure."""
    result: dict[str, Any] = {}
    for resource in ("cpu", "memory", "io"):
        try:
            with open(f"/proc/pressure/{resource}") as f:
                lines = f.readlines()
        except OSError:
            continue
        data: dict[str, Any] = {}
        for line in lines:
            parts = line.split()
            if not parts or parts[0] not in ("some", "full"):
                continue
            fields = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            try:
                data[f"{parts[0]}_avg10"] = float(fields["avg10"])
                data[f"{parts[0]}_total_us"] = int(fields["total"])
            except (KeyError, ValueError):
                pass
        if data:
            result[resource] = data
    if not result:
        warn_once("cannot read /proc/pressure/*, pressure stall info unavailable")
        return None
    return result


class ProcSampler:
    """pidstat-equivalent statistics for the benchmark process, read from
    /proc/<pid>: CPU usage, faults, context switches and I/O deltas per interval."""

    def __init__(self) -> None:
        self.clk_tck = os.sysconf("SC_CLK_TCK")
        self.page_kib = os.sysconf("SC_PAGE_SIZE") // 1024
        self.pid: int | None = None
        self.prev: dict[str, int] | None = None
        self.prev_t: float | None = None

    def watch(self, pid: int | None) -> None:
        self.pid = pid
        self.prev = None
        self.prev_t = None

    def _read(self) -> tuple[dict[str, int], dict[str, Any]] | None:
        try:
            with open(f"/proc/{self.pid}/stat") as f:
                stat = f.read()
            rest = stat[stat.rindex(")") + 2:].split()  # fields after (comm), starting at state
            counters = {
                "minflt": int(rest[7]),
                "majflt": int(rest[9]),
                "cpu_ticks": int(rest[11]) + int(rest[12]),        # utime + stime
                "child_cpu_ticks": int(rest[13]) + int(rest[14]),  # reaped children
            }
            absolute: dict[str, Any] = {
                "pid": self.pid,
                "threads": int(rest[17]),
                "rss_kib": int(rest[21]) * self.page_kib,
            }
        except (OSError, ValueError, IndexError):
            return None
        try:
            with open(f"/proc/{self.pid}/status") as f:
                for line in f:
                    if line.startswith("voluntary_ctxt_switches:"):
                        counters["vctxsw"] = int(line.split()[1])
                    elif line.startswith("nonvoluntary_ctxt_switches:"):
                        counters["nvctxsw"] = int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open(f"/proc/{self.pid}/io") as f:
                for line in f:
                    if line.startswith(("read_bytes:", "write_bytes:")):
                        counters[line.split(":")[0]] = int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        return counters, absolute

    def sample(self) -> dict[str, Any] | None:
        if self.pid is None:
            return None
        data = self._read()
        now = time.monotonic()
        if data is None:
            self.watch(None)  # process ended, normal at the end of a run
            return None
        counters, result = data
        if self.prev is not None and now > self.prev_t:
            dt = now - self.prev_t
            result["cpu_pct"] = round(100.0 * (counters["cpu_ticks"] - self.prev["cpu_ticks"]) / self.clk_tck / dt, 2)
            result["child_cpu_pct"] = round(100.0 * (counters["child_cpu_ticks"] - self.prev["child_cpu_ticks"]) / self.clk_tck / dt, 2)
            for key in ("minflt", "majflt", "vctxsw", "nvctxsw", "read_bytes", "write_bytes"):
                if key in counters and key in self.prev:
                    result[key] = counters[key] - self.prev[key]
        self.prev = counters
        self.prev_t = now
        return result


class SysfsGpu:
    """One amdgpu device sampled directly from /sys/class/drm/card<N>/device."""

    def __init__(self, card: str) -> None:
        self.index = int(os.path.basename(card).removeprefix("card"))
        self.device = os.path.join(card, "device")
        self.name = self._pci_id()
        self.temp_inputs: dict[str, str] = {}
        self.freq_inputs: dict[str, str] = {}
        self.power_input: str | None = None
        for hwmon in glob.glob(os.path.join(self.device, "hwmon", "hwmon*")):
            for temp_input in sorted(glob.glob(os.path.join(hwmon, "temp[0-9]*_input"))):
                label = _read_str(temp_input.replace("_input", "_label")) or "edge"
                self.temp_inputs.setdefault(label, temp_input)
            for freq_input in sorted(glob.glob(os.path.join(hwmon, "freq[0-9]*_input"))):
                label = _read_str(freq_input.replace("_input", "_label")) or "sclk"
                self.freq_inputs.setdefault(label, freq_input)
            for name in ("power1_average", "power1_input"):
                path = os.path.join(hwmon, name)
                if self.power_input is None and os.path.isfile(path):
                    self.power_input = path

    def _pci_id(self) -> str | None:
        try:
            with open(os.path.join(self.device, "uevent")) as f:
                for line in f:
                    if line.startswith("PCI_ID="):
                        return line.strip().split("=", 1)[1]
        except OSError:
            pass
        return None

    def _throttle(self) -> dict[str, Any]:
        """Throttle status from the binary gpu_metrics blob. The prefix layout is
        stable within each format family: throttle_status sits at offset 68 for
        v1_1+ (dGPU) and 108 for v2_1+ (APU), indep_throttle_status at 112 (v1_3+)
        and 120 (v2_2+)."""
        try:
            with open(os.path.join(self.device, "gpu_metrics"), "rb") as f:
                raw = f.read()
        except OSError:
            return {}
        if len(raw) < 4:
            return {}
        _size, fmt, content = struct.unpack("<HBB", raw[:4])

        def u32(off: int) -> int | None:
            return struct.unpack_from("<I", raw, off)[0] if len(raw) >= off + 4 else None

        def u64(off: int) -> int | None:
            return struct.unpack_from("<Q", raw, off)[0] if len(raw) >= off + 8 else None

        if fmt == 1 and content >= 1:
            return {"throttle_status": u32(68),
                    "indep_throttle_status": u64(112) if content >= 3 else None}
        if fmt == 2 and content >= 1:
            return {"throttle_status": u32(108),
                    "indep_throttle_status": u64(120) if content >= 2 else None}
        warn_once(f"card{self.index}: gpu_metrics v{fmt}.{content} not supported, throttle status unavailable")
        return {}

    def sample(self) -> dict[str, Any]:
        temps = {label: _milli_c(_read_int(path)) for label, path in self.temp_inputs.items()}
        freqs = {label: _hz_mhz(_read_int(path)) for label, path in self.freq_inputs.items()}
        power_uw = _read_int(self.power_input) if self.power_input else None
        gpu = {
            "index": self.index,
            "name": self.name,
            "temp_c": temps.get("edge") if temps.get("edge") is not None else temps.get("junction"),
            "temp_edge_c": temps.get("edge"),
            "temp_junction_c": temps.get("junction"),
            "temp_mem_c": temps.get("mem"),
            "util_pct": _read_int(os.path.join(self.device, "gpu_busy_percent")),
            "mem_util_pct": _read_int(os.path.join(self.device, "mem_busy_percent")),
            "vram_used_mib": _bytes_mib(_read_int(os.path.join(self.device, "mem_info_vram_used"))),
            "vram_total_mib": _bytes_mib(_read_int(os.path.join(self.device, "mem_info_vram_total"))),
            "gtt_used_mib": _bytes_mib(_read_int(os.path.join(self.device, "mem_info_gtt_used"))),
            "gtt_total_mib": _bytes_mib(_read_int(os.path.join(self.device, "mem_info_gtt_total"))),
            "power_w": round(power_uw / 1e6, 2) if power_uw is not None else None,
            "clock_mhz": freqs.get("sclk"),
            "mem_clock_mhz": freqs.get("mclk"),
            "throttle_status": None,
            "indep_throttle_status": None,
            "throttle_reasons": None,
        }
        gpu.update(self._throttle())
        gpu["throttle_reasons"] = amd_throttle_reasons(
            gpu["throttle_status"], gpu["indep_throttle_status"])
        return gpu


def discover_sysfs_gpus() -> list[SysfsGpu]:
    cards = []
    for card in glob.glob("/sys/class/drm/card*"):
        if not re.fullmatch(r"card\d+", os.path.basename(card)):
            continue
        if not os.path.isfile(os.path.join(card, "device", "gpu_busy_percent")):
            continue  # not amdgpu
        cards.append(SysfsGpu(card))
    return sorted(cards, key=lambda c: c.index)


class SysfsGpuSampler:
    source = "sysfs"

    def __init__(self, cards: list[SysfsGpu]) -> None:
        self.cards = cards

    def sample(self) -> list[dict[str, Any]]:
        return [card.sample() for card in self.cards]


class SmiGpuSampler:
    def __init__(self, tool: str, timeout: float) -> None:
        self.source = tool
        self.tool = tool
        self.timeout = timeout

    def sample(self) -> list[dict[str, Any]]:
        return sample_gpus(self.tool, self.timeout)


def make_gpu_sampler(source: str, timeout: float) -> "SysfsGpuSampler | SmiGpuSampler | None":
    if source in ("auto", "sysfs"):
        cards = discover_sysfs_gpus()
        if cards:
            log(f"sampling {len(cards)} GPU(s) via sysfs: " + ", ".join(f"card{c.index} ({c.name})" for c in cards))
            return SysfsGpuSampler(cards)
        if source == "sysfs":
            log("no amdgpu cards under /sys/class/drm, GPU sampling disabled")
            return None
    if source in ("auto", "smi"):
        tool = detect_gpu_tool(timeout)
        if tool is not None:
            log(f"sampling GPUs via {tool}")
            return SmiGpuSampler(tool, timeout)
    if source != "none":
        log("no GPU telemetry source found (amdgpu sysfs or nvidia-smi/amd-smi/rocm-smi), sampling CPU only")
    return None


def detect_gpu_tool(timeout: float) -> str | None:
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True, timeout=timeout)
            if r.stdout.strip():
                return "nvidia-smi"
        except (OSError, subprocess.SubprocessError) as e:
            warn_once(f"nvidia-smi probe failed: {e}")
    if shutil.which("amd-smi"):
        try:
            r = subprocess.run(["amd-smi", "list", "--json"],
                               capture_output=True, text=True, check=True, timeout=timeout)
            json.loads(r.stdout)
            return "amd-smi"
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as e:
            warn_once(f"amd-smi probe failed: {e}")
    if shutil.which("rocm-smi"):
        return "rocm-smi"
    return None


def _gpu_entry() -> dict[str, Any]:
    return {
        "index": None,
        "name": None,
        "temp_c": None,
        "util_pct": None,
        "vram_used_mib": None,
        "vram_total_mib": None,
        "power_w": None,
        "clock_mhz": None,
        "throttle_status": None,
        "indep_throttle_status": None,
        "throttle_reasons": None,
    }


_nvidia_throttle_query = True


def _sample_nvidia(timeout: float) -> list[dict[str, Any]]:
    global _nvidia_throttle_query
    fields = [
        "index", "name", "temperature.gpu", "utilization.gpu",
        "memory.used", "memory.total", "power.draw", "clocks.sm",
    ]
    query_fields = fields + (["clocks_event_reasons.active"] if _nvidia_throttle_query else [])
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(query_fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError:
        if not _nvidia_throttle_query:
            raise
        _nvidia_throttle_query = False
        warn_once("nvidia-smi clock event reasons unavailable, throttle events disabled")
        query_fields = fields
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=timeout)
    gpus = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(query_fields):
            continue
        gpu = _gpu_entry()
        index = _num(parts[0])
        gpu["index"] = int(index) if index is not None else None
        gpu["name"] = parts[1] or None
        gpu["temp_c"] = _num(parts[2])
        gpu["util_pct"] = _num(parts[3])
        gpu["vram_used_mib"] = _num(parts[4])
        gpu["vram_total_mib"] = _num(parts[5])
        gpu["power_w"] = _num(parts[6])
        gpu["clock_mhz"] = _num(parts[7])
        if _nvidia_throttle_query:
            gpu["throttle_status"] = _int_value(parts[8])
            gpu["throttle_reasons"] = nvidia_throttle_reasons(gpu["throttle_status"])
        gpus.append(gpu)
    return gpus


def _amd_gpu_entries(payload: Any) -> list[dict[str, Any]]:
    # amd-smi JSON is either a top-level list of per-GPU dicts or a dict
    # wrapping that list under a version-dependent key
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value and all(isinstance(e, dict) for e in value):
                return value
        return [payload]
    return []


def _pick(section: Any, keys: Sequence[str]) -> float | None:
    if not isinstance(section, dict):
        return _num(section)
    for key in keys:
        if key in section:
            value = _num(section[key])
            if value is not None:
                return value
    return None


_amd_names: dict[int, str] | None = None


def _amd_gpu_names(timeout: float) -> dict[int, str]:
    global _amd_names
    if _amd_names is not None:
        return _amd_names
    _amd_names = {}
    try:
        r = subprocess.run(["amd-smi", "static", "--json"],
                           capture_output=True, text=True, check=True, timeout=timeout)
        for i, entry in enumerate(_amd_gpu_entries(json.loads(r.stdout))):
            index = _num(entry.get("gpu"))
            index = int(index) if index is not None else i
            sections = [entry] + [v for v in entry.values() if isinstance(v, dict)]
            for section in sections:
                name = next((v.strip() for k in ("market_name", "device_name", "asic_name", "name")
                             if isinstance(v := section.get(k), str) and v.strip()), None)
                if name:
                    _amd_names[index] = name
                    break
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError, TypeError) as e:
        warn_once(f"amd-smi static name probe failed: {e}")
    return _amd_names


def _sample_amd_smi(timeout: float) -> list[dict[str, Any]]:
    r = subprocess.run(["amd-smi", "metric", "--json"],
                       capture_output=True, text=True, check=True, timeout=timeout)
    names = _amd_gpu_names(timeout)
    gpus = []
    for i, entry in enumerate(_amd_gpu_entries(json.loads(r.stdout))):
        gpu = _gpu_entry()
        index = _num(entry.get("gpu"))
        gpu["index"] = int(index) if index is not None else i
        gpu["name"] = names.get(gpu["index"])
        gpu["temp_c"] = _pick(entry.get("temperature"), ("edge", "hotspot", "junction"))
        gpu["util_pct"] = _pick(entry.get("usage"), ("gfx_activity", "gfx_usage", "gfx"))
        gpu["power_w"] = _pick(entry.get("power"), ("socket_power", "average_socket_power", "current_socket_power"))
        gpu["vram_used_mib"] = _pick(entry.get("mem_usage"), ("used_vram", "used_visible_vram"))
        gpu["vram_total_mib"] = _pick(entry.get("mem_usage"), ("total_vram", "total_visible_vram"))
        gpu["clock_mhz"] = _pick(entry.get("clock"), ("gfx_0", "gfx", "sclk"))
        power = entry.get("power")
        throttle_status = power.get("throttle_status") if isinstance(power, dict) else None
        if throttle_status is None:
            throttle_status = entry.get("throttle_status")
        indep_status = power.get("indep_throttle_status") if isinstance(power, dict) else None
        if indep_status is None:
            indep_status = entry.get("indep_throttle_status")
        gpu["throttle_status"] = _int_value(throttle_status)
        gpu["indep_throttle_status"] = _int_value(indep_status)
        gpu["throttle_reasons"] = amd_throttle_reasons(
            gpu["throttle_status"], gpu["indep_throttle_status"])
        gpus.append(gpu)
    return gpus


def _sample_rocm_smi(timeout: float) -> list[dict[str, Any]]:
    r = subprocess.run(["rocm-smi", "--showtemp", "--showuse", "--showpower", "--showmeminfo", "vram", "--json"],
                       capture_output=True, text=True, check=True, timeout=timeout)
    payload = json.loads(r.stdout)
    gpus = []
    for card, data in sorted(payload.items()):
        if not isinstance(data, dict):
            continue
        gpu = _gpu_entry()
        m = re.search(r"(\d+)$", card)
        gpu["index"] = int(m.group(1)) if m else None
        temp_edge = temp_junction = None
        for key, value in data.items():
            lk = key.lower()
            if "temperature" in lk and "edge" in lk:
                temp_edge = _num(value)
            elif "temperature" in lk and "junction" in lk:
                temp_junction = _num(value)
            elif gpu["util_pct"] is None and "gpu use" in lk:
                gpu["util_pct"] = _num(value)
            elif gpu["power_w"] is None and "power" in lk:
                gpu["power_w"] = _num(value)
            elif gpu["vram_used_mib"] is None and "vram total used" in lk:
                b = _num(value)
                gpu["vram_used_mib"] = round(b / 2**20, 1) if b is not None else None
            elif gpu["vram_total_mib"] is None and "vram total memory" in lk:
                b = _num(value)
                gpu["vram_total_mib"] = round(b / 2**20, 1) if b is not None else None
        gpu["temp_c"] = temp_edge if temp_edge is not None else temp_junction
        gpus.append(gpu)
    return gpus


def sample_gpus(tool: str | None, timeout: float) -> list[dict[str, Any]]:
    samplers = {
        "nvidia-smi": _sample_nvidia,
        "amd-smi": _sample_amd_smi,
        "rocm-smi": _sample_rocm_smi,
    }
    if tool not in samplers:
        return []
    try:
        return samplers[tool](timeout)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
        warn_once(f"{tool} sampling failed: {e}")
        return []


class JsonlOutput:
    """Writes JSONL records, spooling each segment to a scratch file (tmpfs by
    default) and appending it to the final path only when the segment closes, so
    the benchmark sees no result-directory writes while it runs."""

    def __init__(self, path: str, meta_base: dict[str, Any], spool_dir: str | None,
                 run: "dict[str, Any] | None" = None) -> None:
        self.meta_base = meta_base
        self.spool_dir = spool_dir
        self.file: TextIO = sys.stdout
        self.final_path: str | None = None
        self.spool_path: str | None = None
        self._seq = 0
        self._open(path, run)

    def _spool_candidates(self, path: str) -> list[str]:
        name = f"{os.path.basename(path)}.{os.getpid()}.{self._seq}.part"
        candidates = []
        if self.spool_dir:
            candidates.append(os.path.join(self.spool_dir, name))
        # hidden fallback next to the final path, so result cleanup globs skip it
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(path)), f".{name}"))
        return candidates

    def _open(self, path: str, run: "dict[str, Any] | None") -> None:
        self.final_path = None
        self.spool_path = None
        self.file = sys.stdout
        if path != "-":
            self._seq += 1
            for spool in self._spool_candidates(path):
                try:
                    self.file = open(spool, "w")  # noqa: SIM115
                    self.spool_path = spool
                    self.final_path = path
                    break
                except OSError as e:
                    warn_once(f"cannot spool to {spool}: {e}")
            else:
                try:
                    self.file = open(path, "a")  # noqa: SIM115
                except OSError as e:
                    warn_once(f"cannot open {path}: {e}, writing to stdout")
        self.write({"type": "meta", "version": 2, **timestamps(), **self.meta_base, "run": run})

    def rotate(self, path: str, run: "dict[str, Any] | None") -> None:
        self.close()
        self._open(path, run)

    def write(self, obj: dict[str, Any]) -> None:
        self.file.write(json.dumps(obj) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not sys.stdout:
            self.file.close()
            if self.spool_path is not None:
                try:
                    with open(self.spool_path, "rb") as src, open(self.final_path, "ab") as dst:
                        shutil.copyfileobj(src, dst)
                    os.unlink(self.spool_path)
                except OSError as e:
                    warn_once(f"cannot move spooled samples to {self.final_path}: {e}, data left in {self.spool_path}")
        self.file = sys.stdout
        self.final_path = None
        self.spool_path = None


class EventsOutput:
    def __init__(self, spool_dir: str | None) -> None:
        self.spool_dir = spool_dir
        self._seq = 0
        self.final_path: str | None = None
        self.run: dict[str, Any] | None = None
        self.run_start: dict[str, Any] | None = None
        self.run_end: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.active: dict[tuple[str, Any, str], dict[str, Any]] = {}
        self.seen_gpus: set[tuple[str, Any]] = set()

    def rotate(self, path: str | None, run: "dict[str, Any] | None") -> None:
        self.close()
        if path and path != "-":
            self._seq += 1
            self.final_path = path
            self.run = run

    def mark_run(self, state: str, stamp: dict[str, Any]) -> None:
        if self.final_path is None:
            return
        value = {"ts": stamp["ts"], "t": stamp["t"]}
        if state == "start":
            if self.run_start is None:
                self.run_start = value
        elif state == "end":
            self.run_end = value

    @staticmethod
    def _observation(stamp: dict[str, Any], phase: str) -> dict[str, Any]:
        return {"ts": stamp["ts"], "t": stamp["t"], "phase": phase}

    @staticmethod
    def _raw(gpu: dict[str, Any]) -> dict[str, Any]:
        return {
            key: gpu[key]
            for key in ("throttle_status", "indep_throttle_status")
            if gpu.get(key) is not None
        }

    def observe(self, stamp: dict[str, Any], phase: str, source: str,
                gpus: list[dict[str, Any]]) -> None:
        if self.final_path is None:
            return
        for gpu in gpus:
            reasons = gpu.get("throttle_reasons")
            if not isinstance(reasons, list):
                continue
            gpu_id = gpu.get("index")
            if gpu_id is None:
                gpu_id = gpu.get("name")
            gpu_key = (source, gpu_id)
            initial = gpu_key not in self.seen_gpus
            current: set[tuple[str, Any, str]] = set()
            observation = self._observation(stamp, phase)
            raw = self._raw(gpu)

            for reason in reasons:
                if not isinstance(reason, dict) or not isinstance(reason.get("name"), str):
                    continue
                key = (source, gpu_id, reason["name"])
                current.add(key)
                event = self.active.get(key)
                if event is None:
                    event = {
                        "type": "gpu_throttle",
                        "source": source,
                        "gpu": {"index": gpu.get("index"), "name": gpu.get("name")},
                        "reason": reason["name"],
                        "category": reason.get("category", "unknown"),
                        "bit": reason.get("bit"),
                        "start": observation.copy(),
                        "end": None,
                        "duration_s": None,
                        "last_observed": observation.copy(),
                        "initial": initial,
                        "phases": [phase],
                        "raw_start": raw.copy(),
                        "raw_end": None,
                    }
                    self.events.append(event)
                    self.active[key] = event
                else:
                    event["last_observed"] = observation.copy()
                    if phase not in event["phases"]:
                        event["phases"].append(phase)

            for key, event in list(self.active.items()):
                if key[:2] != gpu_key or key in current:
                    continue
                event["end"] = observation.copy()
                event["duration_s"] = round(stamp["t"] - event["start"]["t"], 6)
                event["raw_end"] = raw.copy()
                del self.active[key]

            self.seen_gpus.add(gpu_key)

    def _spool_candidates(self) -> list[str]:
        name = f"{os.path.basename(self.final_path)}.{os.getpid()}.{self._seq}.part"
        candidates = []
        if self.spool_dir:
            candidates.append(os.path.join(self.spool_dir, name))
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(self.final_path)), f".{name}"))
        return candidates

    def _write(self, payload: dict[str, Any]) -> None:
        spool_path = None
        for candidate in self._spool_candidates():
            try:
                with open(candidate, "w") as f:
                    json.dump(payload, f, indent=2)
                    f.write("\n")
                spool_path = candidate
                break
            except OSError as e:
                warn_once(f"cannot spool events to {candidate}: {e}")
        if spool_path is None:
            warn_once(f"cannot write events for {self.final_path}")
            return
        try:
            with open(spool_path, "rb") as src, open(self.final_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.unlink(spool_path)
        except OSError as e:
            warn_once(f"cannot move spooled events to {self.final_path}: {e}, data left in {spool_path}")

    def close(self) -> None:
        if self.final_path is not None:
            duration = None
            if self.run_start is not None and self.run_end is not None:
                duration = round(self.run_end["t"] - self.run_start["t"], 6)
            self._write({
                "version": 1,
                "run": self.run,
                "run_start": self.run_start,
                "run_end": self.run_end,
                "run_duration_s": duration,
                "events": self.events,
            })
        self.final_path = None
        self.run = None
        self.run_start = None
        self.run_end = None
        self.events = []
        self.active = {}
        self.seen_gpus = set()


class Monitor:
    def __init__(self, out: JsonlOutput, gpu: "SysfsGpuSampler | SmiGpuSampler | None",
                 interval: float, phase: str, control: bool,
                 gpu_indices: "list[int] | None" = None) -> None:
        self.out = out
        self.gpu = gpu
        self.interval = interval
        self.phase = phase
        self.control = control
        self.cpu = CpuSampler()
        self.proc = ProcSampler()
        self.events = EventsOutput(out.spool_dir)
        self._stop = False
        self.set_gpu_indices(gpu_indices)

    def set_gpu_indices(self, indices: "list[int] | None") -> None:
        self.gpu_indices = None if indices is None else set(indices)
        self.out.meta_base["gpu_indices"] = (
            None if indices is None else sorted(set(indices)))

    def stop(self, *_args: Any) -> None:
        self._stop = True

    def _phase_event(self, prev: str | None) -> None:
        self.out.write({"type": "phase", **timestamps(), "phase": self.phase, "prev": prev})

    def _take_sample(self) -> None:
        stamp = timestamps()
        cpu = self.cpu.sample()
        pressure = sample_pressure()
        proc = self.proc.sample()
        gpus = self.gpu.sample() if self.gpu is not None else []
        gpus = select_gpus(gpus, self.gpu_indices)
        if self.gpu is not None:
            self.events.observe(stamp, self.phase, self.gpu.source, gpus)
        self.out.write({
            "type": "sample",
            **stamp,
            "phase": self.phase,
            "cpu": cpu,
            "pressure": pressure,
            "proc": proc,
            "gpus": gpus,
        })

    @staticmethod
    def _command_stamp(cmd: dict[str, Any]) -> dict[str, Any]:
        if isinstance(cmd.get("ts"), str) and isinstance(cmd.get("t"), (int, float)):
            return {"ts": cmd["ts"], "t": float(cmd["t"])}
        return timestamps()

    def _handle_command(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            cmd = json.loads(line)
            name = cmd["cmd"]
        except (json.JSONDecodeError, KeyError, TypeError):
            warn_once(f"ignoring malformed command: {line!r}")
            return
        if name == "rotate":
            if "gpus" in cmd:
                self.set_gpu_indices(parse_gpu_indices(cmd["gpus"]))
            prev = self.phase
            self.phase = cmd.get("phase", self.phase)
            self.events.rotate(cmd.get("events_path"), cmd.get("run"))
            self.out.rotate(cmd["path"], cmd.get("run"))
            self._phase_event(prev)
            self._take_sample()
        elif name == "phase":
            prev = self.phase
            self.phase = cmd.get("phase", self.phase)
            if self.phase != prev:
                self._phase_event(prev)
        elif name == "watch":
            pid = cmd.get("pid")
            try:
                self.proc.watch(int(pid) if pid is not None else None)
            except (TypeError, ValueError):
                warn_once(f"ignoring invalid watch pid: {pid!r}")
        elif name == "run_start":
            self.events.mark_run("start", self._command_stamp(cmd))
        elif name == "run_end":
            self.events.mark_run("end", self._command_stamp(cmd))
        elif name == "gpus":
            self.set_gpu_indices(parse_gpu_indices(cmd.get("gpus")))
        elif name == "stop":
            self._stop = True
        else:
            warn_once(f"ignoring unknown command: {name}")

    def run(self) -> None:
        sel = selectors.DefaultSelector()
        stdin_fd = None
        if self.control:
            stdin_fd = sys.stdin.fileno()
            os.set_blocking(stdin_fd, False)
            sel.register(stdin_fd, selectors.EVENT_READ)
        buf = b""
        next_tick = time.monotonic()
        self._phase_event(None)
        try:
            while not self._stop:
                wait = max(0.0, next_tick - time.monotonic())
                if stdin_fd is not None:
                    for _key, _mask in sel.select(wait):
                        chunk = os.read(stdin_fd, 65536)
                        if not chunk:  # driver closed stdin
                            self._stop = True
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._handle_command(line.decode(errors="replace"))
                elif wait > 0:
                    time.sleep(wait)
                if self._stop:
                    break
                if time.monotonic() >= next_tick:
                    self._take_sample()
                    next_tick += self.interval
                    if next_tick < time.monotonic():  # missed ticks (slow smi call)
                        next_tick = time.monotonic() + self.interval
        finally:
            self.events.close()
            self.out.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample CPU, GPU and benchmark-process telemetry to JSONL with minimal perturbation")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or - for stdout")
    parser.add_argument("--interval", "-i", default=1.0, type=float, help="Sample interval in seconds")
    parser.add_argument("--phase", default="idle", help="Initial phase label attached to samples")
    parser.add_argument(
        "--control-stdin",
        action="store_true",
        help="Read newline-delimited JSON commands "
             "(rotate/phase/watch/run_start/run_end/gpus/stop) from stdin",
    )
    parser.add_argument(
        "--gpu-source",
        default="auto",
        choices=["auto", "sysfs", "smi", "none"],
        help="GPU telemetry source: sysfs reads amdgpu hwmon/sysfs files directly, "
             "smi shells out to nvidia-smi/amd-smi/rocm-smi, auto prefers sysfs and falls back to smi",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU indices to sample (matches ggml_vk_visible_devices). "
             "Default: all GPUs reported by the telemetry source",
    )
    parser.add_argument(
        "--spool-dir",
        default="/dev/shm",
        help="Scratch directory for in-flight samples, appended to --output only when "
             "a segment ends ('none' to write --output directly)",
    )
    parser.add_argument(
        "--pin-cpu",
        type=int,
        default=None,
        help="Logical CPU to pin the monitor to (default: highest allowed CPU, -1 to disable)",
    )
    args = parser.parse_args()

    deprioritize(args.pin_cpu)

    gpu = make_gpu_sampler(args.gpu_source, max(2.0, args.interval))
    spool_dir = args.spool_dir if args.spool_dir not in ("", "none") else None
    gpu_indices = parse_gpu_indices(args.gpus)

    out = JsonlOutput(args.output, {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_source": gpu.source if gpu is not None else None,
        "gpu_indices": None if gpu_indices is None else sorted(set(gpu_indices)),
        "interval": args.interval,
    }, spool_dir)
    monitor = Monitor(out, gpu, args.interval, args.phase, args.control_stdin, gpu_indices)
    signal.signal(signal.SIGTERM, monitor.stop)
    signal.signal(signal.SIGINT, monitor.stop)
    monitor.run()


if __name__ == "__main__":
    main()
