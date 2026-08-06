import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_interface():
    module = load_script("qvac_bench_interface", REPO_ROOT / "scripts" / "qvac_bench_interface.py")
    # Register under the real module name so qvac-bench.py's own import binds
    # to this instance instead of loading a second copy.
    sys.modules["qvac_bench_interface"] = module
    return module


def load_db():
    module = load_script("qvac_bench_db", REPO_ROOT / "scripts" / "qvac_bench_db.py")
    # Same registration dance as load_interface().
    sys.modules["qvac_bench_db"] = module
    return module


def load_qvac_bench():
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = type("DataFrame", (), {})
    old_pandas = sys.modules.get("pandas")
    sys.modules["pandas"] = pandas
    try:
        return load_script("qvac_bench_under_test", REPO_ROOT / "scripts" / "qvac-bench.py")
    finally:
        if old_pandas is None:
            del sys.modules["pandas"]
        else:
            sys.modules["pandas"] = old_pandas


qbm = load_script(
    "qvac_bench_monitor_under_test",
    REPO_ROOT / "scripts" / "qvac-bench-monitor.py",
)
qbi = load_interface()
qbd = load_db()
qb = load_qvac_bench()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def stamp(t: float) -> dict:
    return {"ts": f"2026-07-29T00:00:{int(t):02d}+00:00", "t": t}


def test_parse_and_select_gpu_indices():
    assert qbm.parse_gpu_indices(None) is None
    assert qbm.parse_gpu_indices("") is None
    assert qbm.parse_gpu_indices("  ") is None
    assert qbm.parse_gpu_indices(0) == [0]
    assert qbm.parse_gpu_indices("1") == [1]
    assert qbm.parse_gpu_indices("0,2") == [0, 2]
    assert qbm.parse_gpu_indices([0, "3"]) == [0, 3]

    gpus = [
        {"index": 0, "name": "a"},
        {"index": 1, "name": "b"},
        {"index": 2, "name": "c"},
    ]
    assert qbm.select_gpus(gpus, None) == gpus
    assert qbm.select_gpus(gpus, {1}) == [{"index": 1, "name": "b"}]
    assert qbm.select_gpus(gpus, {0, 2}) == [
        {"index": 0, "name": "a"},
        {"index": 2, "name": "c"},
    ]
    assert qbm.select_gpus(gpus, set()) == []


def test_monitor_filters_gpus_by_indices(tmp_path, monkeypatch):
    output_path = tmp_path / "filtered.jsonl"
    out = qbm.JsonlOutput(
        str(output_path),
        {"host": "test-host", "gpu_source": "fake", "gpu_indices": [1], "interval": 1.0},
        str(tmp_path),
    )

    class FakeGpu:
        source = "fake"

        def sample(self):
            return [
                {"index": 0, "name": "idle", "throttle_reasons": []},
                {"index": 1, "name": "busy", "throttle_reasons": []},
                {"index": 2, "name": "other", "throttle_reasons": []},
            ]

    monitor = qbm.Monitor(out, FakeGpu(), interval=1.0, phase="run", control=False,
                          gpu_indices=[1])
    monkeypatch.setattr(monitor.cpu, "sample", lambda: {"util_pct": 0.0})
    monkeypatch.setattr(qbm, "sample_pressure", lambda: {})
    monkeypatch.setattr(monitor.proc, "sample", lambda: None)

    monitor._take_sample()
    monitor._handle_command(json.dumps({
        "cmd": "rotate",
        "path": str(tmp_path / "run.jsonl"),
        "phase": "cooldown",
        "gpus": "0,2",
        "run": {"benchmark": "llama-bench"},
    }))
    out.close()

    records = read_jsonl(output_path)
    sample = next(record for record in records if record["type"] == "sample")
    assert [gpu["index"] for gpu in sample["gpus"]] == [1]

    run_records = read_jsonl(tmp_path / "run.jsonl")
    assert run_records[0]["gpu_indices"] == [0, 2]
    run_sample = next(record for record in run_records if record["type"] == "sample")
    assert [gpu["index"] for gpu in run_sample["gpus"]] == [0, 2]


def test_watch_command_logs_exec_record_and_events_commands(tmp_path, monkeypatch):
    output_path = tmp_path / "run.jsonl"
    events_path = tmp_path / "run.events.json"
    out = qbm.JsonlOutput(
        str(output_path),
        {"host": "test-host", "gpu_source": None, "interval": 1.0},
        str(tmp_path),
    )
    monitor = qbm.Monitor(out, None, interval=1.0, phase="idle", control=False)
    monkeypatch.setattr(monitor.cpu, "sample", lambda: {"util_pct": 0.0})
    monkeypatch.setattr(qbm, "sample_pressure", lambda: None)
    monkeypatch.setattr(monitor.proc, "sample", lambda: None)

    monitor._handle_command(json.dumps({
        "cmd": "rotate",
        "path": str(output_path),
        "events_path": str(events_path),
        "phase": "run",
        "run": {"benchmark": "llama-bench"},
    }))
    argv = ["llama-bench", "-m", "test.gguf", "-ngl", "99", "-o", "jsonl"]
    monitor._handle_command(json.dumps({"cmd": "watch", "pid": 4242, "argv": argv}))
    assert monitor.proc.pid == 4242
    monitor._handle_command(json.dumps({"cmd": "watch", "pid": None}))
    assert monitor.proc.pid is None
    monitor._handle_command(json.dumps({"cmd": "watch", "pid": "bogus", "argv": argv}))
    monitor.events.close()
    out.close()

    records = read_jsonl(output_path)
    execs = [record for record in records if record["type"] == "exec"]
    assert len(execs) == 1
    assert execs[0]["phase"] == "run"
    assert execs[0]["pid"] == 4242
    assert execs[0]["argv"] == argv

    with events_path.open() as f:
        payload = json.load(f)
    assert len(payload["commands"]) == 1
    command = payload["commands"][0]
    assert command["pid"] == 4242
    assert command["phase"] == "run"
    assert command["argv"] == argv
    assert payload["events"] == []


def test_monitor_client_passes_gpus_cli(tmp_path, monkeypatch):
    seen = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            seen["args"] = list(args)
            self.stdin = type("Stdin", (), {
                "write": staticmethod(lambda _data: None),
                "flush": staticmethod(lambda: None),
                "close": staticmethod(lambda: None),
            })()
            self.returncode = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(qbi.subprocess, "Popen", FakePopen)
    monitor = qbi.MonitorClient(enabled=True, interval=0.5, gpu_source="none")
    meta = {"invocation": {"argv": ["qvac-bench.py", "--config", "cfg.json"],
                           "config": {"benchmark": "llama-bench"}}}
    monitor.start(tmp_path / "sweep.jsonl", phase="setup", gpus=0, meta=meta)
    assert "--gpus" in seen["args"]
    assert seen["args"][seen["args"].index("--gpus") + 1] == "0"
    assert "--meta" in seen["args"]
    assert json.loads(seen["args"][seen["args"].index("--meta") + 1]) == meta

    sent = []
    monitor.proc.stdin = type("Stdin", (), {
        "write": staticmethod(lambda data: sent.append(data)),
        "flush": staticmethod(lambda: None),
        "close": staticmethod(lambda: None),
    })()
    monitor.rotate(tmp_path / "run.jsonl", {"benchmark": "llama-bench"},
                   phase="cooldown", gpus="1")
    payload = json.loads(sent[0])
    assert payload["cmd"] == "rotate"
    assert payload["gpus"] == "1"


def test_monitor_interval_cli_overrides_config():
    args = qb.build_parser().parse_args(["--monitor-interval", "0.25"])
    config = {"options": {"monitor_interval": 2.0}}

    result = qb.apply_overrides(config, args)

    assert result["options"]["monitor_interval"] == 0.25


def test_monitor_interval_config_is_preserved_without_cli_override():
    args = qb.build_parser().parse_args([])
    config = {"options": {"monitor_interval": 0.5}}

    result = qb.apply_overrides(config, args)

    assert result["options"]["monitor_interval"] == 0.5


def test_driver_downloads_models_before_starting_monitor(tmp_path, monkeypatch):
    events = []

    class FakeModel:
        def download(self):
            events.append("download")

    class FakeMonitor:
        active = False

        def __init__(self, _enabled, _interval, _gpu_source):
            pass

        def start(self, _path, phase, **_kwargs):
            events.append(("start", phase))

        def phase(self, label):
            events.append(("phase", label))

        def stop(self):
            events.append("stop")

    for name in ("WORKDIR", "RESULTS_DIR", "MODELS_DIR", "IMAGES_DIR"):
        monkeypatch.setattr(qb, name, tmp_path / name.lower())
    monkeypatch.setattr(qb, "MonitorClient", FakeMonitor)

    benchmark = next(b for b in qb.benchmarks if b.name == "llama-bench")
    monkeypatch.setattr(benchmark, "create_report", lambda *_args: None)

    qb.bench_driver(benchmark, [FakeModel()], [], {"clean_unused": False}, {})

    assert events == [
        "download",
        ("start", "build"),
        ("phase", "idle"),
        "stop",
    ]


def test_jsonl_output_spools_and_rotates_segments(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    sweep_path = tmp_path / "sweep.jsonl"
    run_path = tmp_path / "run.jsonl"
    meta = {
        "host": "test-host",
        "platform": "test-platform",
        "gpu_source": None,
        "interval": 0.1,
    }
    run = {"benchmark": "llama-bench", "backend": "vulkan"}

    out = qbm.JsonlOutput(str(sweep_path), meta, str(spool_dir))
    out.write({"type": "phase", "phase": "setup"})

    assert not sweep_path.exists()
    assert len(list(spool_dir.iterdir())) == 1

    out.rotate(str(run_path), run)

    sweep_records = read_jsonl(sweep_path)
    assert sweep_records[0]["type"] == "meta"
    assert sweep_records[0]["version"] == 2
    assert sweep_records[0]["run"] is None
    assert {key: sweep_records[0][key] for key in meta} == meta
    assert sweep_records[1] == {"type": "phase", "phase": "setup"}
    assert not run_path.exists()

    out.write({"type": "sample", "phase": "run", "gpus": []})
    out.close()

    run_records = read_jsonl(run_path)
    assert run_records[0]["type"] == "meta"
    assert run_records[0]["version"] == 2
    assert run_records[0]["run"] == run
    assert run_records[0]["gpu_source"] is None
    assert run_records[1] == {"type": "sample", "phase": "run", "gpus": []}
    assert list(spool_dir.iterdir()) == []


def test_throttle_reason_decoding():
    reasons = qbm.amd_throttle_reasons(
        status=1,
        indep_status=(1 << 0) | (1 << 22) | (1 << 36) | (1 << 63),
    )

    assert reasons == [
        {"name": "PPT0", "category": "power", "bit": 0},
        {"name": "EDC_GFX", "category": "current", "bit": 22},
        {"name": "TEMP_HOTSPOT", "category": "thermal", "bit": 36},
        {"name": "BIT_63", "category": "unknown", "bit": 63},
    ]
    assert qbm.amd_throttle_reasons(status=1, indep_status=0) == [
        {"name": "UNKNOWN", "category": "unknown", "bit": None},
    ]
    assert qbm.amd_throttle_reasons(status=0, indep_status=0) == []
    assert qbm.amd_throttle_reasons(status=None, indep_status=None) is None
    assert qbm.amd_throttle_reasons(
        status=0xFFFFFFFF, indep_status=0xFFFFFFFFFFFFFFFF) is None

    assert qbm.nvidia_throttle_reasons((1 << 0) | (1 << 2) | (1 << 6)) == [
        {"name": "SW_POWER_CAP", "category": "power", "bit": 2},
        {"name": "HW_THERMAL_SLOWDOWN", "category": "thermal", "bit": 6},
    ]
    assert qbm.nvidia_throttle_reasons(1) == []
    assert qbm.nvidia_throttle_reasons(None) is None


def test_events_output_tracks_throttle_intervals(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    events_path = tmp_path / "run.events.json"
    run = {"benchmark": "llama-bench", "backend": "vulkan"}
    out = qbm.EventsOutput(str(spool_dir))
    out.rotate(str(events_path), run)
    out.mark_run("start", stamp(2.0))

    gpu = {
        "index": 0,
        "name": "test-gpu",
        "throttle_status": 1,
        "indep_throttle_status": (1 << 0) | (1 << 36),
        "throttle_reasons": qbm.amd_throttle_reasons(
            status=1, indep_status=(1 << 0) | (1 << 36)),
    }
    out.observe(stamp(0.0), "cooldown", "sysfs", [gpu])

    gpu["indep_throttle_status"] = 1 << 36
    gpu["throttle_reasons"] = qbm.amd_throttle_reasons(
        status=1, indep_status=1 << 36)
    out.observe(stamp(1.0), "warmup", "sysfs", [gpu])

    # A missing sample does not prove that the active reason ended.
    out.observe(stamp(2.0), "run", "sysfs", [])
    gpu["throttle_status"] = 0
    gpu["indep_throttle_status"] = 0
    gpu["throttle_reasons"] = []
    out.observe(stamp(3.0), "run", "sysfs", [gpu])
    out.mark_run("end", stamp(5.0))

    assert not events_path.exists()
    assert list(spool_dir.iterdir()) == []
    out.close()

    with events_path.open() as f:
        payload = json.load(f)

    assert payload["version"] == 1
    assert payload["run"] == run
    assert payload["run_start"] == stamp(2.0)
    assert payload["run_end"] == stamp(5.0)
    assert payload["run_duration_s"] == 3.0
    assert [event["reason"] for event in payload["events"]] == [
        "PPT0", "TEMP_HOTSPOT",
    ]

    power, thermal = payload["events"]
    assert power["category"] == "power"
    assert power["initial"] is True
    assert power["start"]["phase"] == "cooldown"
    assert power["end"]["phase"] == "warmup"
    assert power["duration_s"] == 1.0
    assert power["phases"] == ["cooldown"]

    assert thermal["category"] == "thermal"
    assert thermal["initial"] is True
    assert thermal["start"]["phase"] == "cooldown"
    assert thermal["end"]["phase"] == "run"
    assert thermal["duration_s"] == 3.0
    assert thermal["phases"] == ["cooldown", "warmup"]
    assert thermal["last_observed"]["phase"] == "warmup"
    assert list(spool_dir.iterdir()) == []


def test_events_output_leaves_active_interval_open(tmp_path):
    events_path = tmp_path / "run.events.json"
    out = qbm.EventsOutput(str(tmp_path))
    out.rotate(str(events_path), {"benchmark": "llama-bench"})
    out.observe(stamp(1.0), "run", "nvidia-smi", [{
        "index": 2,
        "name": "test-nvidia",
        "throttle_status": 1 << 5,
        "indep_throttle_status": None,
        "throttle_reasons": qbm.nvidia_throttle_reasons(1 << 5),
    }])
    out.close()

    with events_path.open() as f:
        payload = json.load(f)
    assert payload["run_start"] is None
    assert payload["run_end"] is None
    assert payload["run_duration_s"] is None
    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["reason"] == "SW_THERMAL_SLOWDOWN"
    assert event["end"] is None
    assert event["duration_s"] is None
    assert event["last_observed"]["t"] == 1.0


def test_optional_gpu_backend_writes_cpu_only_sample(tmp_path, monkeypatch):
    output_path = tmp_path / "cpu-only.jsonl"
    assert qbm.make_gpu_sampler("none", 1.0) is None

    out = qbm.JsonlOutput(
        str(output_path),
        {"host": "test-host", "gpu_source": None, "interval": 1.0},
        str(tmp_path),
    )
    monitor = qbm.Monitor(out, None, interval=1.0, phase="run", control=False)
    cpu = {
        "util_pct": 12.5,
        "load1": 1.0,
        "load5": 2.0,
        "load15": 3.0,
        "mem_total_kib": 1024,
        "mem_available_kib": 512,
        "temps_c": None,
    }
    monkeypatch.setattr(monitor.cpu, "sample", lambda: cpu)
    monkeypatch.setattr(qbm, "sample_pressure", lambda: {"cpu": {"some_avg10": 0.0}})
    monkeypatch.setattr(monitor.proc, "sample", lambda: None)

    monitor._phase_event(None)
    monitor._take_sample()
    out.close()

    records = read_jsonl(output_path)
    assert [record["type"] for record in records] == ["meta", "phase", "sample"]
    assert records[1]["phase"] == "run"
    assert records[1]["prev"] is None
    assert records[2]["phase"] == "run"
    assert records[2]["cpu"] == cpu
    assert records[2]["pressure"] == {"cpu": {"some_avg10": 0.0}}
    assert records[2]["proc"] is None
    assert records[2]["gpus"] == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the monitor samples Linux /proc")
def test_monitor_client_and_sidecar_exchange_commands(tmp_path):
    sweep_path = tmp_path / "sweep.jsonl"
    run_path = tmp_path / "run.jsonl"
    events_path = tmp_path / "run.events.json"
    run = {
        "benchmark": "llama-bench",
        "backend": "vulkan",
        "model": "test.gguf",
    }
    invocation = {"argv": ["scripts/qvac-bench.py", "--config", "bench.json"],
                  "config": {"benchmark": "llama-bench"}}
    monitor = qbi.MonitorClient(enabled=True, interval=0.02, gpu_source="none")

    monitor.start(sweep_path, phase="setup", meta={"invocation": invocation})
    assert monitor.active
    try:
        monitor.rotate(run_path, run, phase="cooldown", events_path=events_path)
        monitor.phase("run")
        monitor.run_start()
        commands = []
        qbi.set_monitor(monitor)
        qbi.set_run_commands(commands)
        try:
            result = qbi.monitored_run(
                [sys.executable, "-c", "import time; time.sleep(0.25)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        finally:
            qbi.set_monitor(None)
            qbi.set_run_commands(None)
            monitor.run_end()
    finally:
        monitor.stop()

    assert result.returncode == 0
    assert commands == [[sys.executable, "-c", "import time; time.sleep(0.25)"]]
    sweep_records = read_jsonl(sweep_path)
    run_records = read_jsonl(run_path)
    with events_path.open() as f:
        event_data = json.load(f)

    assert sweep_records[0]["type"] == "meta"
    assert sweep_records[0]["run"] is None
    assert sweep_records[0]["invocation"] == invocation
    assert any(
        record["type"] == "phase"
        and record["phase"] == "setup"
        and record["prev"] is None
        for record in sweep_records
    )

    assert run_records[0]["type"] == "meta"
    assert run_records[0]["run"] == run
    assert run_records[0]["gpu_source"] is None
    assert run_records[0]["interval"] == 0.02
    assert run_records[0]["invocation"] == invocation
    assert any(
        record["type"] == "phase"
        and record["phase"] == "cooldown"
        and record["prev"] == "setup"
        for record in run_records
    )
    assert any(
        record["type"] == "phase"
        and record["phase"] == "run"
        and record["prev"] == "cooldown"
        for record in run_records
    )

    samples = [record for record in run_records if record["type"] == "sample"]
    assert samples
    assert all(sample["phase"] in {"cooldown", "run"} for sample in samples)
    assert all(sample["gpus"] == [] for sample in samples)
    assert any(sample["proc"] is not None for sample in samples)
    assert {
        "util_pct",
        "load1",
        "load5",
        "load15",
        "mem_total_kib",
        "mem_available_kib",
        "temps_c",
    } <= samples[0]["cpu"].keys()
    execs = [record for record in run_records if record["type"] == "exec"]
    assert len(execs) == 1
    assert execs[0]["phase"] == "run"
    assert execs[0]["argv"][0] == sys.executable

    assert event_data["version"] == 1
    assert event_data["run"] == run
    assert event_data["run_start"]["t"] <= event_data["run_end"]["t"]
    assert event_data["run_duration_s"] > 0
    assert [command["argv"] for command in event_data["commands"]] == [execs[0]["argv"]]
    assert event_data["events"] == []


def test_disabled_monitor_client_is_a_noop(tmp_path):
    output_path = tmp_path / "disabled.jsonl"
    events_path = tmp_path / "disabled.events.json"
    monitor = qbi.MonitorClient(enabled=False, interval=0.1, gpu_source="none")

    monitor.start(output_path, phase="setup")
    monitor.rotate(
        output_path,
        {"benchmark": "llama-bench"},
        phase="run",
        events_path=events_path,
    )
    monitor.phase("idle")
    monitor.run_start()
    monitor.run_end()
    monitor.watch(123)
    monitor.stop()

    assert not monitor.active
    assert not output_path.exists()
    assert not events_path.exists()
