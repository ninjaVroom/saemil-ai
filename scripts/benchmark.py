"""`python -m scripts.benchmark vision|names|tabular` - Gate 3: 3 fresh-process training runs, BLAS threads = 1."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"  # must precede the first numpy import, here and in each child

import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

RUNS, MAX_SECONDS, MAX_BYTES = 3, 120, 250_000_000
HEADS = ("vision", "names", "tabular")


def peak_working_set():
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux KiB only
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(ctypes.get_last_error())
    return c.PeakWorkingSetSize


def child(head, seed):
    from scripts import train
    quiet = lambda *_: None  # noqa: E731
    with tempfile.TemporaryDirectory() as d:  # time the archive save without touching models/
        out = os.path.join(d, f"{head}.npz")
        if head == "vision":
            r = train.train_vision(seed, out=out, log=quiet)
            r["metric"], r["epochs"] = ("test acc", r["test_accuracy"]), train.EPOCHS
        elif head == "names":
            r = train.train_names(seed, out=out, log=quiet)
        else:
            r = train.train_tabular("uci", seed, out=out, log=quiet)
    print(json.dumps({"peak_bytes": peak_working_set(), "metric": list(r["metric"]), "rows": r["rows"],
                      "epochs": r["epochs"]}))


def cpu_name():
    if sys.platform != "win32":
        return platform.processor() or "unknown"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        return platform.processor() or "unknown"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--child"]:
        return child(argv[1], int(argv[2]))
    if len(argv) != 1 or argv[0] not in HEADS:
        raise SystemExit("usage: python -m scripts.benchmark vision|names|tabular")
    import numpy as np

    from scripts.train import DEFAULT_SEED
    head = argv[0]
    seed, runs = DEFAULT_SEED[head], []
    for i in range(RUNS):
        t0 = time.perf_counter()
        out = subprocess.run([sys.executable, "-m", "scripts.benchmark", "--child", head, str(seed)],
                             capture_output=True, text=True, check=True, env=os.environ)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        r["seconds"] = time.perf_counter() - t0
        runs.append(r)
        print(f"run {i + 1}: {r['seconds']:.2f} s, peak {r['peak_bytes']:,} B, {r['metric'][0]} {r['metric'][1]:.4f}")
    try:
        info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        blas = f"{info.get('name')} {info.get('version')}"
    except Exception:  # noqa: BLE001 - informational only
        blas = "unknown"
    secs, peaks = [r["seconds"] for r in runs], [r["peak_bytes"] for r in runs]
    ok = max(secs) < MAX_SECONDS and max(peaks) < MAX_BYTES
    rows = [("CPU", cpu_name()), ("Python", platform.python_version()), ("NumPy", np.__version__),
            ("BLAS", blas), ("Threads", "OPENBLAS/MKL/OMP_NUM_THREADS=1"), ("Head", head),
            ("Rows (fit)", runs[0]["rows"]), ("Epochs", runs[0]["epochs"]), ("Seed", seed), ("Runs", RUNS),
            ("Wall time per run (s)", ", ".join(f"{s:.2f}" for s in secs)),
            ("Median wall time (s)", f"{statistics.median(secs):.2f}"), ("Slowest wall time (s)", f"{max(secs):.2f}"),
            ("Peak working set (bytes)", ", ".join(f"{p:,}" for p in peaks)), ("Max peak (bytes)", f"{max(peaks):,}"),
            ("Gate (<120 s and <250,000,000 B every run)", "PASS" if ok else "FAIL")]
    print("\n| Metric | Value |\n|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
