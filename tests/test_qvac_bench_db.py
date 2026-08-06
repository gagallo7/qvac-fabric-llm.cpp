import importlib.util
import json
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


qbd = load_script("qvac_bench_db_under_test", REPO_ROOT / "scripts" / "qvac_bench_db.py")


BENCH_RECORD = {
    "build_commit": "3352bd945",
    "build_number": 9840,
    "gpu_info": "AMD Radeon RX 7900 XTX (RADV NAVI31)",
    "backends": "Vulkan",
    "model_type": "llama 1B Q4_0",
    "model_filename": "Llama-3.2-1B-Instruct-Q4_0.gguf",
    "n_prompt": 512,
    "n_gen": 0,
    "test_time": "2026-08-06T10:00:00Z",
    "avg_ts": 5000.5,
    "stddev_ts": 12.5,
    "samples_ts": [4990, 5000.5, 5010],
}

STATUS = {
    "ref": "3352bd945deadbeefdeadbeefdeadbeefdeadbee",
    "repo": "https://example.invalid/llama.cpp.git",
    "branch": "b9840",
    "backend": "vulkan",
    "model": "Llama-3.2-1B-Instruct-Q4_0.gguf",
    "gpus": ["AMD Radeon RX 7900 XTX - Mesa 25.1", "llvmpipe - Mesa 25.1"],
    "status": "success",
}


def write_run(tmp_path: Path, *, status=STATUS, records=(BENCH_RECORD,),
              samples=(), events=()):
    stem = tmp_path / "llama-bench-3352bd945-vulkan-model.gguf-0123456789"
    stem.with_name(stem.name + ".status").write_text(json.dumps(status))
    stem.with_name(stem.name + ".stdout").write_text(
        "\n".join(json.dumps(r) for r in records)
    )
    if samples:
        lines = [{"type": "meta", "host": "france", "run": {}}]
        lines += [{"type": "sample", **s} for s in samples]
        stem.with_name(stem.name + ".monitor.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines)
        )
    if events:
        stem.with_name(stem.name + ".events.json").write_text(
            json.dumps({"version": 1, "events": list(events)})
        )
    return stem


def test_upsert_sql_keys_and_updates():
    assert "INSERT INTO benchmark_results" in qbd.BENCH_UPSERT_SQL
    assert (
        "ON CONFLICT (time, commit_sha, device, backend, model, metric)"
        in qbd.BENCH_UPSERT_SQL
    )
    # A re-push must never clobber the stored measurement or its link.
    assert "value = EXCLUDED.value" not in qbd.BENCH_UPSERT_SQL
    assert "run_url = EXCLUDED.run_url" not in qbd.BENCH_UPSERT_SQL
    assert "branch = EXCLUDED.branch" in qbd.BENCH_UPSERT_SQL
    assert "value = EXCLUDED.value" in qbd.MONITOR_SAMPLE_UPSERT_SQL
    assert qbd.BENCH_UPSERT_SQL.count("%s") == len(qbd.BENCH_COLUMNS)
    assert qbd.MONITOR_SAMPLE_UPSERT_SQL.count("%s") == len(qbd.MONITOR_SAMPLE_COLUMNS)
    assert qbd.MONITOR_EVENT_UPSERT_SQL.count("%s") == len(qbd.MONITOR_EVENT_COLUMNS)


def test_record_to_row_pp_and_tg_metrics():
    row = qbd.record_to_row(BENCH_RECORD, "upstream/b9840")
    assert dict(zip(qbd.BENCH_COLUMNS, row)) == {
        "time": "2026-08-06T10:00:00Z",
        "commit_sha": "3352bd945",
        "branch": "upstream/b9840",
        "build_number": 9840,
        "device": "AMD Radeon RX 7900 XTX (RADV NAVI31)",
        "backend": "Vulkan",
        "model": "llama 1B Q4_0",
        "metric": "pp512_tokens_per_sec",
        "value": 5000.5,
        "stddev": 12.5,
        "n_samples": 3,
        "samples": [4990.0, 5000.5, 5010.0],
        "run_url": None,
    }
    assert all(isinstance(s, float) for s in row[11])

    tg = dict(BENCH_RECORD, n_prompt=0, n_gen=128)
    assert qbd.record_to_row(tg, "b")[7] == "tg128_tokens_per_sec"


def test_sample_flattening_respects_whitelists():
    ctx = {
        "commit_sha": "3352bd945", "branch": "upstream/b9840", "build_number": 9840,
        "device": "gfx1100", "backend": "Vulkan", "model": "m", "host": "france",
    }
    sample = {
        "ts": "2026-08-06T10:00:01+00:00",
        "phase": "run",
        "cpu": {"util_pct": 50, "temps_c": {"Tctl": 60.5}, "bogus": 1},
        "pressure": {"io": {"some_avg10": 0.5, "junk": 9}},
        "proc": {"rss_kib": 1024, "cmdline": "not-a-number"},
        "gpus": [{"index": 0, "power_w": 250.0, "name": "not-numeric"}],
    }
    rows = qbd.sample_to_metric_rows(sample, ctx)
    metrics = {r[9]: r[10] for r in rows}
    assert metrics == {
        "cpu.util_pct": 50.0,
        "cpu.temp.Tctl": 60.5,
        "pressure.io.some_avg10": 0.5,
        "proc.rss_kib": 1024.0,
        "gpu.0.power_w": 250.0,
    }
    assert all(r[8] == "run" for r in rows)
    assert qbd.sample_to_metric_rows({"phase": "run"}, ctx) == []


def test_event_to_row_defaults_and_open_interval():
    ctx = {
        "commit_sha": "c", "branch": "b", "build_number": 1,
        "device": "d", "backend": "v", "model": "m", "host": "h",
    }
    event = {
        "type": "gpu_throttle",
        "reason": "power_cap",
        "gpu": {"index": 0, "name": "gfx1100"},
        "start": {"ts": "2026-08-06T10:00:00+00:00", "phase": "run"},
        "phases": ["run"],
    }
    row = qbd.event_to_row(event, ctx)
    named = dict(zip(qbd.MONITOR_EVENT_COLUMNS, row))
    assert named["event_type"] == "gpu_throttle"
    assert named["end_time"] is None
    assert named["still_open"] is True
    assert named["gpu_index"] == 0
    assert named["category"] == ""
    assert qbd.event_to_row({"start": {"ts": "x"}}, ctx) is None


def test_context_from_status_and_stdout_enrichment(tmp_path):
    stem = write_run(tmp_path)
    ctx = qbd.context_from_status(STATUS)
    assert ctx == {
        "commit_sha": "3352bd945",
        "branch": "upstream/b9840",
        "backend": "vulkan",
        "model": "Llama-3.2-1B-Instruct-Q4_0.gguf",
        "device": "AMD Radeon RX 7900 XTX",
        "build_number": 9840,
        "host": "",
    }
    # llama-bench's own labels win so keys match importer-inserted rows.
    enriched = qbd.enrich_context(ctx, stem)
    assert enriched["device"] == "AMD Radeon RX 7900 XTX (RADV NAVI31)"
    assert enriched["backend"] == "Vulkan"
    assert enriched["model"] == "llama 1B Q4_0"

    assert qbd.context_from_status({"branch": "b9840"}) is None


def test_collect_run_rows_success(tmp_path):
    stem = write_run(
        tmp_path,
        samples=[{"ts": "2026-08-06T10:00:01+00:00", "phase": "run",
                  "cpu": {"util_pct": 10}}],
        events=[{"type": "gpu_throttle", "reason": "hot",
                 "start": {"ts": "2026-08-06T10:00:02+00:00"}}],
    )
    bench, samples, events = qbd.collect_run_rows(stem)
    assert len(bench) == 1
    assert bench[0][7] == "pp512_tokens_per_sec"
    assert len(samples) == 1
    assert samples[0][7] == "france"  # host from monitor meta
    assert len(events) == 1


def test_collect_run_rows_failure_status_drops_bench_rows(tmp_path):
    stem = write_run(tmp_path, status=dict(STATUS, status="failure"))
    bench, _, _ = qbd.collect_run_rows(stem)
    assert bench == []


def test_collect_run_rows_non_llama_bench_stdout(tmp_path):
    stem = write_run(tmp_path, records=())
    stem.with_name(stem.name + ".stdout").write_text(
        "epoch 1/3 loss=0.5\nnot json at all\n"
    )
    bench, samples, events = qbd.collect_run_rows(stem)
    assert bench == [] and samples == [] and events == []


def test_collect_run_rows_missing_status(tmp_path):
    assert qbd.collect_run_rows(tmp_path / "nope") == ([], [], [])


class FakeCursor:
    def __init__(self, calls):
        self.calls = calls

    def executemany(self, sql, rows):
        self.calls.append((sql, list(rows)))

    def execute(self, sql):
        self.calls.append((sql, None))

    def fetchone(self):
        return (1,)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, calls):
        self.calls = calls
        self.committed = False

    def cursor(self):
        return FakeCursor(self.calls)

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_psycopg(calls, connections, fail=False):
    module = types.ModuleType("psycopg")

    def connect(url, connect_timeout=None):
        if fail:
            raise OSError("connection refused")
        conn = FakeConnection(calls)
        connections.append((url, conn))
        return conn

    module.connect = connect
    return module


def test_pusher_upserts_all_tables_and_commits(tmp_path, monkeypatch):
    stem = write_run(
        tmp_path,
        samples=[{"ts": "2026-08-06T10:00:01+00:00", "cpu": {"util_pct": 10}}],
        events=[{"type": "gpu_throttle", "start": {"ts": "2026-08-06T10:00:02+00:00"}}],
    )
    calls, connections = [], []
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg(calls, connections))
    pusher = qbd.DbPusher("postgresql://example.invalid/db")
    assert pusher.push_run(stem) is True
    assert connections[0][0] == "postgresql://example.invalid/db"
    assert connections[0][1].committed
    assert [sql for sql, _ in calls] == [
        qbd.BENCH_UPSERT_SQL,
        qbd.MONITOR_SAMPLE_UPSERT_SQL,
        qbd.MONITOR_EVENT_UPSERT_SQL,
    ]


def test_pusher_swallows_db_errors(tmp_path, monkeypatch):
    stem = write_run(tmp_path)
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg([], [], fail=True))
    pusher = qbd.DbPusher("postgresql://example.invalid/db")
    assert pusher.push_run(stem) is False


def test_pusher_skips_connect_when_no_rows(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg([], [], fail=True))
    pusher = qbd.DbPusher("postgresql://example.invalid/db")
    # Missing files -> zero rows -> success without touching the DB.
    assert pusher.push_run(tmp_path / "nope") is True


def test_push_sweep_monitor(tmp_path, monkeypatch):
    path = tmp_path / "llama-bench-monitor.jsonl"
    lines = [
        {"type": "meta", "host": "france", "run": None},
        {"type": "sample", "ts": "2026-08-06T10:00:01+00:00", "phase": "idle",
         "cpu": {"util_pct": 5}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))
    calls, connections = [], []
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg(calls, connections))
    ctx = {
        "commit_sha": "3352bd945", "branch": "upstream/b9840", "build_number": 9840,
        "device": "AMD Radeon RX 7900 XTX", "backend": "vulkan", "model": "",
        "host": "",
    }
    pusher = qbd.DbPusher("postgresql://example.invalid/db")
    assert pusher.push_sweep_monitor(path, ctx) is True
    ((sql, rows),) = calls
    assert sql == qbd.MONITOR_SAMPLE_UPSERT_SQL
    assert rows[0][7] == "france"  # host picked up from the meta line
    assert pusher.push_sweep_monitor(path, None) is False


def test_make_pusher(monkeypatch):
    assert qbd.make_pusher(None) is None
    assert qbd.make_pusher("") is None
    monkeypatch.setitem(sys.modules, "psycopg", types.ModuleType("psycopg"))
    assert isinstance(qbd.make_pusher("postgresql://x/y"), qbd.DbPusher)
    monkeypatch.setitem(sys.modules, "psycopg", None)  # forces ImportError
    with pytest.raises(qbd.DbPreflightError):
        qbd.make_pusher("postgresql://x/y")


def test_ping_reachable(monkeypatch):
    calls, connections = [], []
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg(calls, connections))
    qbd.DbPusher("postgresql://example.invalid/db").ping()
    assert calls == [("SELECT 1", None)]


def test_ping_unreachable_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg([], [], fail=True))
    with pytest.raises(qbd.DbPreflightError, match="connection refused"):
        qbd.DbPusher("postgresql://example.invalid/db").ping()


def test_db_url_does_not_affect_options_or_stems(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    sys.modules["qvac_bench_db"] = qbd
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = type("DataFrame", (), {})
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    qb = load_script("qvac_bench_for_db_test", REPO_ROOT / "scripts" / "qvac-bench.py")

    parser = qb.build_parser()
    base = parser.parse_args([])
    with_db = parser.parse_args(["--db-url", "postgresql://example.invalid/db"])
    assert with_db.db_url == "postgresql://example.invalid/db"
    assert base.db_url in (None, "")

    cfg_base = qb.apply_overrides({}, base)
    cfg_db = qb.apply_overrides({}, with_db)
    # --db-url never reaches options, so result stems (hashed from options)
    # are identical with and without it.
    assert cfg_base.get("options") == cfg_db.get("options")
    assert "db_url" not in cfg_db.get("options", {})


def load_qvac_bench(monkeypatch, tmp_path):
    sys.modules["qvac_bench_db"] = qbd
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = type("DataFrame", (), {})
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    qb = load_script("qvac_bench_for_db_test", REPO_ROOT / "scripts" / "qvac-bench.py")
    for name in ("WORKDIR", "RESULTS_DIR", "MODELS_DIR", "IMAGES_DIR"):
        monkeypatch.setattr(qb, name, tmp_path / name.lower())
    return qb


def test_driver_fails_fast_on_unknown_reference(monkeypatch, tmp_path):
    qb = load_qvac_bench(monkeypatch, tmp_path)
    benchmark = next(b for b in qb.benchmarks if b.name == "llama-bench")
    build = qb.Build(repo="https://example.invalid/r.git", branch="x", backend="default")
    with pytest.raises(SystemExit, match="matches no build"):
        qb.bench_driver(benchmark, [], [build], {}, {"reference": "typo"})


def test_driver_fails_fast_on_unreachable_db(monkeypatch, tmp_path):
    qb = load_qvac_bench(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg([], [], fail=True))
    benchmark = next(b for b in qb.benchmarks if b.name == "llama-bench")
    build = qb.Build(repo="https://example.invalid/r.git", branch="x", backend="default")
    with pytest.raises(SystemExit, match="cannot reach database"):
        qb.bench_driver(
            benchmark, [], [build], {}, {},
            db_url="postgresql://example.invalid/db",
        )
