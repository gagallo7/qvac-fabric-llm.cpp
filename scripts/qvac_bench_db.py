"""Direct-to-database push of qvac-bench results.

Upserts llama-bench throughput rows, monitor telemetry samples and monitor
events into the TimescaleDB behind the qvac perf dashboards, so a sweep
populates Grafana live instead of requiring a later run of the dashboards
repo's import_llama_bench_output.py over the results directory.

The column tuples, natural keys and row-mapping semantics below intentionally
mirror that importer: rows pushed from here and rows imported from the same
files must produce identical natural keys, so either path is an idempotent
re-upsert of the other. That includes run_id: both paths read it from the
.status file (minted once per execution by qvac-bench.py), so schema or
semantics changes here must be mirrored in the importer.

psycopg is imported lazily; without it (or without a database URL) the bench
runs exactly as before. Every push is best-effort — a database problem is
logged and never fails the sweep.
"""

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def log(label: str, msg: str) -> None:
    logger.info("[%s] %s", label, msg)


BENCH_COLUMNS = (
    "time", "commit_sha", "branch", "build_number", "device", "backend",
    "model", "metric", "value", "stddev", "n_samples", "samples", "run_url",
    "run_id",
)
BENCH_KEY_COLUMNS = ("time", "commit_sha", "device", "backend", "model", "metric")
# value and run_url keep their stored values when a re-push hits an existing key.
BENCH_UPDATE_COLUMNS = tuple(
    c for c in BENCH_COLUMNS if c not in (*BENCH_KEY_COLUMNS, "value", "run_url")
)

MONITOR_SAMPLE_COLUMNS = (
    "time", "commit_sha", "branch", "build_number", "device", "backend",
    "model", "host", "phase", "run_id", "metric", "value",
)
MONITOR_SAMPLE_KEY_COLUMNS = ("time", "commit_sha", "device", "backend", "model", "metric")
MONITOR_SAMPLE_UPDATE_COLUMNS = tuple(
    c for c in MONITOR_SAMPLE_COLUMNS if c not in MONITOR_SAMPLE_KEY_COLUMNS
)

MONITOR_EVENT_COLUMNS = (
    "time", "end_time", "duration_s", "commit_sha", "branch", "build_number",
    "device", "backend", "model", "host", "event_type", "category", "reason",
    "gpu_index", "gpu_name", "source", "phase", "phases", "initial", "still_open",
    "run_id",
)
MONITOR_EVENT_KEY_COLUMNS = (
    "time", "commit_sha", "device", "backend", "model", "event_type", "reason", "gpu_index",
)
MONITOR_EVENT_UPDATE_COLUMNS = tuple(
    c for c in MONITOR_EVENT_COLUMNS if c not in MONITOR_EVENT_KEY_COLUMNS
)


def _upsert_sql(table: str, columns: tuple, key: tuple, update: tuple) -> str:
    head = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ("
    values = ", ".join(["%s"] * len(columns))
    tail = (
        f") ON CONFLICT ({', '.join(key)}) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in update)
    )
    return head + values + tail


BENCH_UPSERT_SQL = _upsert_sql(
    "benchmark_results", BENCH_COLUMNS, BENCH_KEY_COLUMNS, BENCH_UPDATE_COLUMNS,
)
MONITOR_SAMPLE_UPSERT_SQL = _upsert_sql(
    "monitor_samples", MONITOR_SAMPLE_COLUMNS, MONITOR_SAMPLE_KEY_COLUMNS,
    MONITOR_SAMPLE_UPDATE_COLUMNS,
)
MONITOR_EVENT_UPSERT_SQL = _upsert_sql(
    "monitor_events", MONITOR_EVENT_COLUMNS, MONITOR_EVENT_KEY_COLUMNS,
    MONITOR_EVENT_UPDATE_COLUMNS,
)

BUILD_TAG_RE = re.compile(r"^b(\d+)$")
BUILD_NUMBER_RE = re.compile(r"b(\d+)$")

# One run_id per benchmark execution: "<result_stem>@<utc-iso-timestamp>",
# minted by qvac-bench.py when the job starts and stored in the .status file.
RUN_ID_TABLES = ("benchmark_results", "monitor_samples", "monitor_events")
RUN_ID_MIGRATION = tuple(
    stmt.format(table=t)
    for t in RUN_ID_TABLES
    for stmt in (
        "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS run_id TEXT;",
        "CREATE INDEX IF NOT EXISTS {table}_run_id_idx"
        " ON {table} (run_id text_pattern_ops);",
    )
)

GPU_METRIC_FIELDS = (
    "temp_c", "temp_edge_c", "temp_junction_c", "temp_mem_c",
    "util_pct", "mem_util_pct",
    "vram_used_mib", "vram_total_mib", "gtt_used_mib", "gtt_total_mib",
    "power_w", "clock_mhz", "mem_clock_mhz",
    "throttle_status", "indep_throttle_status",
)
CPU_SCALAR_FIELDS = (
    "util_pct", "load1", "load5", "load15", "mem_total_kib", "mem_available_kib",
)
PROC_SCALAR_FIELDS = (
    "threads", "rss_kib", "cpu_pct", "child_cpu_pct",
    "minflt", "majflt", "vctxsw", "nvctxsw", "read_bytes", "write_bytes",
)
PRESSURE_RESOURCES = ("cpu", "memory", "io")
PRESSURE_FIELDS = ("some_avg10", "some_total_us", "full_avg10", "full_total_us")


def _num(value) -> "float | None":
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_branch_label(branch: "str | None") -> "str | None":
    if not branch:
        return None
    if BUILD_TAG_RE.match(branch):
        return f"upstream/{branch}"
    return branch


def build_number_from_label(label: "str | None") -> "int | None":
    if not label:
        return None
    m = BUILD_NUMBER_RE.search(label)
    return int(m.group(1)) if m else None


def iter_records(file: Path):
    """Yield llama-bench result dicts from a *.stdout file.

    Handles JSON arrays, JSONL and concatenated objects; stray non-JSON lines
    (crash dumps, build noise) and non-dict values are skipped.
    """
    text = file.read_text()
    if not text.lstrip():
        return
    dec = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError:
            nl = text.find("\n", idx)
            if nl < 0:
                break
            idx = nl + 1
            continue
        idx = end
        if isinstance(obj, dict):
            yield obj
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item


def iter_jsonl(file: Path):
    for line in file.open():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def record_to_row(r: dict, ctx: dict) -> tuple:
    metric = (
        f"pp{r['n_prompt']}_tokens_per_sec"
        if r.get("n_prompt")
        else f"tg{r['n_gen']}_tokens_per_sec"
    )
    # JSON may emit whole numbers as ints; psycopg rejects mixed-type lists.
    raw_samples = r.get("samples_ts") or None
    samples = [float(x) for x in raw_samples] if raw_samples else None
    return (
        r.get("test_time"),
        r["build_commit"],
        ctx["branch"],
        r["build_number"],
        r.get("gpu_info") or r.get("cpu_info") or "",
        r.get("backends") or "",
        r.get("model_type") or r.get("model_filename") or "",
        metric,
        float(r["avg_ts"]),
        float(r.get("stddev_ts") or 0),
        len(samples) if samples else 1,
        samples,
        None,
        ctx.get("run_id"),
    )


def sample_to_metric_rows(sample: dict, ctx: dict) -> list:
    """Flatten one monitor sample into narrow (time, ..., metric, value) rows."""
    ts = sample.get("ts")
    if not ts:
        return []
    base = (
        ts,
        ctx["commit_sha"],
        ctx["branch"],
        ctx["build_number"],
        ctx["device"],
        ctx["backend"],
        ctx["model"],
        ctx.get("host") or "",
        sample.get("phase") or "",
        ctx.get("run_id"),
    )
    rows: list = []

    def add(metric: str, value) -> None:
        n = _num(value)
        if n is None:
            return
        rows.append((*base, metric, n))

    cpu = sample.get("cpu") or {}
    for field in CPU_SCALAR_FIELDS:
        add(f"cpu.{field}", cpu.get(field))
    temps = cpu.get("temps_c")
    if isinstance(temps, dict):
        for name, value in temps.items():
            add(f"cpu.temp.{name}", value)

    pressure = sample.get("pressure") or {}
    for resource in PRESSURE_RESOURCES:
        vals = pressure.get(resource)
        if not isinstance(vals, dict):
            continue
        for field in PRESSURE_FIELDS:
            add(f"pressure.{resource}.{field}", vals.get(field))

    proc = sample.get("proc")
    if isinstance(proc, dict):
        for field in PROC_SCALAR_FIELDS:
            add(f"proc.{field}", proc.get(field))

    for gpu in sample.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        idx = gpu.get("index")
        if idx is None:
            continue
        for field in GPU_METRIC_FIELDS:
            add(f"gpu.{idx}.{field}", gpu.get(field))

    return rows


def event_to_row(event: dict, ctx: dict) -> "tuple | None":
    start = event.get("start") or {}
    ts = start.get("ts")
    if not ts or not event.get("type"):
        return None
    end = event.get("end") or {}
    gpu = event.get("gpu") or {}
    gpu_index = gpu.get("index")
    if gpu_index is None:
        gpu_index = -1
    phases = event.get("phases")
    if not isinstance(phases, list):
        phases = None
    return (
        ts,
        end.get("ts"),
        _num(event.get("duration_s")),
        ctx["commit_sha"],
        ctx["branch"],
        ctx["build_number"],
        ctx["device"],
        ctx["backend"],
        ctx["model"],
        ctx.get("host") or "",
        event["type"],
        event.get("category") or "",
        event.get("reason") or "",
        int(gpu_index),
        gpu.get("name") or "",
        event.get("source") or "",
        start.get("phase") or "",
        phases,
        bool(event.get("initial")),
        end.get("ts") is None,
        ctx.get("run_id"),
    )


def context_from_status(status: dict) -> "dict | None":
    """Run dimensions from a qvac-bench .status dict.

    The bench driver knows branch/sha/backend directly, so unlike the
    dashboards importer no git resolution is needed.
    """
    ref = status.get("ref") or ""
    branch = normalize_branch_label(status.get("branch"))
    if not ref or not branch:
        return None
    device = ""
    gpus = status.get("gpus") or []
    if gpus and isinstance(gpus[0], str):
        device = gpus[0].split(" - ")[0].strip()
    return {
        "commit_sha": ref[:9],
        "branch": branch,
        "backend": status.get("backend") or "",
        "model": status.get("model") or "",
        "device": device,
        "build_number": build_number_from_label(branch) or 0,
        "host": "",
        # Absent in status files from before run ids existed; pushed as NULL.
        "run_id": status.get("run_id"),
    }


def enrich_context(ctx: dict, stem: Path) -> dict:
    """Override status-derived dimensions with llama-bench's own labels.

    The importer prefers gpu_info/backends/model_type from the first stdout
    record; matching that here keeps the natural keys of monitor rows aligned
    with bench rows and with importer-inserted rows. Host comes from the
    monitor meta line.
    """
    stdout_path = stem.with_name(stem.name + ".stdout")
    if stdout_path.is_file():
        for rec in iter_records(stdout_path):
            ctx["commit_sha"] = rec.get("build_commit") or ctx["commit_sha"]
            ctx["device"] = rec.get("gpu_info") or rec.get("cpu_info") or ctx["device"]
            ctx["backend"] = rec.get("backends") or ctx["backend"]
            ctx["model"] = rec.get("model_type") or rec.get("model_filename") or ctx["model"]
            if rec.get("build_number") is not None:
                ctx["build_number"] = int(rec["build_number"])
            break
    monitor_path = stem.with_name(stem.name + ".monitor.jsonl")
    if monitor_path.is_file():
        for obj in iter_jsonl(monitor_path):
            if obj.get("type") == "meta":
                ctx["host"] = obj.get("host") or ctx["host"]
                break
    return ctx


def collect_run_rows(stem: Path) -> "tuple[list, list, list]":
    """Rows for one run, identified by its result path stem (no suffix).

    Bench rows require a success status and valid llama-bench records; other
    benchmarks' stdout simply yields none. Monitor/event rows are collected
    for every benchmark kind.
    """
    status_path = stem.with_name(stem.name + ".status")
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return [], [], []
    ctx = context_from_status(status)
    if ctx is None:
        return [], [], []
    ctx = enrich_context(ctx, stem)

    bench_rows: list = []
    stdout_path = stem.with_name(stem.name + ".stdout")
    if status.get("status") == "success" and stdout_path.is_file():
        for rec in iter_records(stdout_path):
            try:
                bench_rows.append(record_to_row(rec, ctx))
            except (KeyError, TypeError, ValueError):
                pass

    sample_rows: list = []
    monitor_path = stem.with_name(stem.name + ".monitor.jsonl")
    if monitor_path.is_file():
        for obj in iter_jsonl(monitor_path):
            if obj.get("type") == "sample":
                sample_rows.extend(sample_to_metric_rows(obj, ctx))

    event_rows: list = []
    events_path = stem.with_name(stem.name + ".events.json")
    if events_path.is_file():
        try:
            payload = json.loads(events_path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            for event in payload.get("events") or []:
                if isinstance(event, dict):
                    row = event_to_row(event, ctx)
                    if row is not None:
                        event_rows.append(row)

    return bench_rows, sample_rows, event_rows


def collect_monitor_file_rows(path: Path, ctx: dict) -> list:
    """Sample rows from a bare monitor jsonl (the sweep-level file)."""
    if not path.is_file():
        return []
    rows: list = []
    for obj in iter_jsonl(path):
        if obj.get("type") == "meta" and obj.get("host"):
            ctx = dict(ctx, host=obj["host"])
        elif obj.get("type") == "sample":
            rows.extend(sample_to_metric_rows(obj, ctx))
    return rows


class DbPreflightError(Exception):
    """A database push was requested but cannot possibly work."""


class DbPusher:
    """Best-effort upserts into the dashboards TimescaleDB.

    One connection per push so a long sweep never holds a stale connection;
    every public method swallows all errors — the database must never fail
    a bench run.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def ping(self) -> None:
        """Preflight connectivity check; raises DbPreflightError if unreachable.

        Unlike the push methods, this does not swallow errors — it exists so
        the driver can fail fast before spending hours on a sweep whose
        results were meant to land in the database.
        """
        import psycopg

        try:
            with psycopg.connect(self.db_url, connect_timeout=10) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.execute(
                        "SELECT table_name FROM information_schema.columns"
                        " WHERE table_name = ANY(%s) AND column_name = 'run_id'",
                        (list(RUN_ID_TABLES),),
                    )
                    have_run_id = {row[0] for row in cursor.fetchall()}
        except Exception as e:  # noqa: BLE001
            raise DbPreflightError(f"cannot reach database: {e}") from e
        missing = [t for t in RUN_ID_TABLES if t not in have_run_id]
        if missing:
            raise DbPreflightError(
                "run_id column missing on: " + ", ".join(missing)
                + " (also mirror this in the dashboards repo schema/importer); apply:\n"
                + "\n".join(RUN_ID_MIGRATION)
            )
        log("db", "database reachable")

    def push_run(self, stem: Path, wait_s: float = 0.0) -> bool:
        try:
            # The monitor sidecar finalizes <stem>.monitor.jsonl/.events.json
            # asynchronously on rotate; give it a moment before reading.
            events_path = stem.with_name(stem.name + ".events.json")
            deadline = time.monotonic() + wait_s
            while not events_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.1)
            bench_rows, sample_rows, event_rows = collect_run_rows(stem)
            return self._push_rows(stem.name, bench_rows, sample_rows, event_rows)
        except Exception as e:  # noqa: BLE001
            log("db", f"push failed for {stem.name}: {e}")
            return False

    def push_sweep_monitor(self, path: Path, ctx: "dict | None") -> bool:
        if ctx is None:
            return False
        try:
            sample_rows = collect_monitor_file_rows(path, ctx)
            return self._push_rows(path.name, [], sample_rows, [])
        except Exception as e:  # noqa: BLE001
            log("db", f"push failed for {path.name}: {e}")
            return False

    def _push_rows(self, label: str, bench_rows: list, sample_rows: list,
                   event_rows: list) -> bool:
        if not (bench_rows or sample_rows or event_rows):
            return True
        try:
            import psycopg

            with psycopg.connect(self.db_url, connect_timeout=10) as conn:
                with conn.cursor() as cursor:
                    if bench_rows:
                        cursor.executemany(BENCH_UPSERT_SQL, bench_rows)
                    if sample_rows:
                        cursor.executemany(MONITOR_SAMPLE_UPSERT_SQL, sample_rows)
                    if event_rows:
                        cursor.executemany(MONITOR_EVENT_UPSERT_SQL, event_rows)
                conn.commit()
        except Exception as e:  # noqa: BLE001
            log("db", f"push failed for {label}: {e}")
            return False
        log(
            "db",
            f"pushed {label}: bench={len(bench_rows)} "
            f"samples={len(sample_rows)} events={len(event_rows)}",
        )
        return True


def make_pusher(db_url: "str | None") -> "DbPusher | None":
    if not db_url:
        return None
    try:
        import psycopg  # noqa: F401
    except ImportError as e:
        raise DbPreflightError(
            "--db-url given but psycopg is not installed "
            "(pip install 'psycopg[binary]')"
        ) from e
    return DbPusher(db_url)
