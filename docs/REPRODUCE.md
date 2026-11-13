# Reproducing the models and benchmarks

## Benchmark method

Train time and peak memory come from `python -m scripts.benchmark <head>`: three fresh-process runs, each covering start, data load, training, evaluation and archive save, with BLAS threads set to 1, on one machine (AMD Ryzen 5 5500U, Windows 11, Python 3.13.7, NumPy 2.5.4). The gate for every head was under 120 s and under 250,000,000 bytes peak working set on every run. Recorded dataset and benchmark details are in [`data/SOURCES.md`](../data/SOURCES.md).

- Parameter counts are the weight and bias arrays stored in `models/*.npz`.
- Vision memory: the first benchmark recorded a maximum of 56,254,464 bytes; the rerun after a shared-script change (1.95 / 1.88 / 1.87 s) is recorded as 58 MB without per-run bytes.
- Names: of 100 fixed-seed candidates at temperature 0.85, 15% matched a source name and were filtered out; 85 were accepted.
- Benchmarks are from one machine; the power state during the runs was not confirmed.

## From scratch

Downloads happen once, in the `prepare_data` step. Each raw file is checked against a recorded SHA-256 (a mismatch deletes the file and stops). Raw files go to `data/raw/` and are not committed. After that, preparing, training and benchmarking run fully offline (verified with networking blocked in-process, not with an OS firewall rule).

```bash
# 1. Download (once) and build the fixed splits: MNIST seed 17, names seed 23, UCI seed 31
python -m scripts.prepare_data all

# 2. Synthetic tabular fallback: generate the CSV (seed 41, 6,000 rows), then split it
python -m scripts.distill_data
python -m scripts.prepare_data tabular --source synthetic

# 3. Train (seeds shown are the defaults)
python -m scripts.train vision --seed 17
python -m scripts.train names --seed 23
python -m scripts.train tabular --seed 31
python -m scripts.train tabular --source synthetic --seed 31

# 4. Benchmark (3 fresh-process runs each, BLAS threads = 1)
python -m scripts.benchmark vision
python -m scripts.benchmark names
python -m scripts.benchmark tabular

# 5. Compare the models with the recorded hashes
sha256sum models/*.npz          # PowerShell: Get-FileHash models\*.npz -Algorithm SHA256
```

Training writes over `models/*.npz` and `prepare_data` over `data/prepared/*.npz`. `--seed` changes only the training seed (initialisation and batch order); the data splits stay fixed. `distill_data --teacher cli|cloud` is not implemented and exits with a message.

## Byte-identical archives

Retraining with the default seeds gave identical archives in local checks and again with the network blocked (R-008), on the machine above with NumPy 2.5.4. A different NumPy or BLAS build may give different bytes.

| File | SHA-256 |
|---|---|
| `models/vision.npz` | `f47d55fbf026813a1845af7cd3251625416102e4930e80782e9046f204aeefdb` |
| `models/names.npz` | `87ec5f29000748578c59d100ce792f00d898e4c47abb400e9ec1a3d7435ec7fe` |
| `models/tabular.npz` | `10b74cf2043bf98abf17a0fd4f7d869259d492295b0bb21b0cdaef346290588a` |
| `models/tabular_synthetic.npz` | `492e765db0625620fbfd590c9a4c3d9329ed6175c97710335dc96a706b6c2f9f` |

The studio serves `models/tabular.npz` (UCI). `tabular_synthetic.npz` is a training artifact for the synthetic fallback and is not loaded by the studio.
