# Data sources

## MNIST (vision head, decision D-005)

| Field | Value |
|---|---|
| File | `data/raw/mnist.npz` (11,490,434 bytes, not committed; see `.gitignore`) |
| URL | https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz |
| SHA-256 | `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1` |
| Hash source | `file_hash` argument to `get_file` in `keras/src/datasets/mnist.py`, keras-team/keras `master`, read 2026-11-12. The downloaded file matched it. |
| Contents | official 60,000 train / 10,000 test, uint8 28x28 images, uint8 labels |
| Copyright | Yann LeCun and Corinna Cortes. MNIST is a derivative work of the original NIST datasets. |
| License | Creative Commons Attribution-Share Alike 3.0, https://creativecommons.org/licenses/by-sa/3.0/ (as stated in the Keras loader docstring) |

Fetch and prepare with `python -m scripts.prepare_data vision`. The download happens once. A hash mismatch deletes the file and stops.

### Prepared split: `data/prepared/vision.npz`

- Dataset id: `mnist-sha256:731c5ac60275-split-s17-800/100/100`
- Split seed 17 is fixed (`SPLIT_SEED` in `scripts/prepare_data.py`). `python -m scripts.train vision --seed N` only changes the training seed (init and batch order); the split always stays seed 17. `load_vision` refuses a cache whose dataset_id or seed does not match.
- Fit: 800 per digit (8,000) from the official train set.
- Val: the next 100 per digit (1,000) from the official train set, disjoint from fit.
- Test: 100 per digit (1,000) from the official test set.
- Stored: `x_*` uint8 (N,28,28), `y_*` int64, `idx_*` int64 (indices into the official arrays), `source_sha256`, `seed`, `pixel_convention`.
- Pixel convention: uint8 28x28 row-major, 255 = ink on a 0 background. The model input is `uint8 / 255` as float32 (`heads.vision.to_input`).
- Written with `np.savez_compressed` (1,671,173 bytes). File SHA-256 as built 2026-11-12: `5df71d4ee575a9be772055dd60a31187d2a2c078d31eb7c8ece18bf6a34a75b5`. The zip bytes may vary across NumPy versions, so verify the contents rather than this digest. Its arrays are identical to the earlier uncompressed build, and the model retrained from it is bit-identical.
- Centering check: `heads.vision.center_by_mass` reproduces 9,996 of the 10,000 official MNIST test images exactly. The four that differ are indices 4530, 9300 and 9302 (ink box longer side 19 px) and 8688 (20 px). All 1,000 images in the prepared test split are reproduced exactly.

## Split indices: where they live

`data/prepared/*.npz` is the authoritative home of every split: names (`names_fit/val/test`), tabular (`idx_fit/val/test`, row indices into the source CSV) and vision (`idx_*`). Model archives do not copy them. Each names and tabular model manifest records `training_source_id` and `prepared_dataset_id` (the same id as the prepared cache's `dataset_id`), `prepared_cache` (its path) and the full `source_sha256` of the raw source. `load_*` refuses a cache whose dataset_id or split seed differs.

## makemore names.txt (names head, decision D-006)

| Field | Value |
|---|---|
| File | `data/raw/names.txt` (228,145 bytes, 32,033 lines, not committed) |
| URL | https://raw.githubusercontent.com/karpathy/makemore/988aa59e4d8fefa526d06f3b453ad116258398d4/names.txt |
| Pinned commit | `988aa59e4d8fefa526d06f3b453ad116258398d4` (karpathy/makemore `master` head, 2022-11-20, read via the GitHub API 2026-11-12). The GitHub commit history for `names.txt` has one entry, `8f79bd0126eaef5eb82d346c9f85ae20ea1c476a` ("first commit", 2022-06-09). The file at both commits is byte-identical. |
| SHA-256 | `0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d` (computed locally from the download. No upstream checksum is published.) |
| License | Repository: MIT License, Copyright (c) 2022 Andrej Karpathy (`LICENSE` at the pinned commit). |
| Origin note | The makemore README says the file holds the most common 32K names from ssa.gov (US Social Security Administration baby-name data, https://www.ssa.gov/oact/babynames/) for 2018. That is US government public data. These are personal names, so brand suitability is a separate judgement, and novelty is not trademark clearance. |

Fetch and prepare with `python -m scripts.prepare_data names`.

### Prepared split: `data/prepared/names.npz`

- Dataset id: `makemore-988aa59-sha256:0a30b5557f19-clean-s23-4000-80/10/10`
- Clean (`heads.names.clean`): lowercase, strip, keep full matches of `[a-z]{2,16}`, dedupe keeping first occurrence. 32,033 lines become 29,494 unique names. All 2,539 dropped lines were duplicates, and no line failed the pattern.
- Split seed 23 is fixed (`NAMES_SEED` in `scripts/prepare_data.py`): permute, keep 4,000, then split 80/10/10 **by whole name**: fit 3,200 / val 400 / test 400, pairwise disjoint. Windows are built only after the split.
- Windows: context 3, vocab `.abcdefghijklmnopqrstuvwxyz` (0 = boundary). Fit 22,899 / val 2,872 / test 2,793 windows, stored uint8 and cast to int64 at batch time.
- Stored: `names_*` (strings), `X_*` uint8 (N,3), `y_*` uint8, `source_names` (all 29,494 clean names, sorted), `vocab`, `source_sha256`, `seed`, `dataset_id`.
- `models/names.npz` carries `preproc.vocab` (27 single characters) and `preproc.source_names` (all 29,494) as plain string arrays for novelty filtering, no pickle.

## UCI Online Shoppers Purchasing Intention (tabular head, real source)

| Field | Value |
|---|---|
| File | `data/raw/online_shoppers_intention.zip` (1,072,219 bytes, not committed). Its one member is `online_shoppers_intention.csv` (1,072,063 bytes). |
| URL | https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip (UCI ML Repository dataset id 468) |
| SHA-256 | `2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14` (computed locally from the 2026-11-12 download. UCI publishes no checksum, so a UCI re-zip would need a new recorded hash.) |
| Contents | 12,330 sessions and 1,908 purchases (`Revenue` TRUE), prevalence 0.1547. Both counts are checked on every prepare. |
| Citation | Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ (as stated on the UCI dataset page) |

Fetch and prepare with `python -m scripts.prepare_data tabular`. The CSV is read with `zipfile` + `csv.DictReader`. Only the six features in fixed order are used: `Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration`. `Revenue` maps TRUE/FALSE to 1/0, and every value must be finite and >= 0. These features are whole-session totals, so the model describes completed sessions. A live mid-session predictor would need timestamped features, which are out of scope.

### Prepared split: `data/prepared/tabular.npz`

- Dataset id: `uci-468-sha256:2972e6184d3a-strat-s31-70/15/15`
- Stratified by class, seed 31 (`TABULAR_SEED`): fit 8,631 (prevalence 0.1548) / val 1,849 (0.1547) / test 1,850 (0.1546).
- Stored: `X_*` float64 raw units, `y_*` int64, `idx_*` (row indices into the CSV), `features`, `source`, `source_sha256`, `seed`, `dataset_id`.
- Preprocessing is fitted on fit only (`heads.tabular.fit_preproc`): `x <- log1p(max(x, 0))`, then standardize by the fit mean/std, with std < 1e-6 giving scale 1.0. `models/tabular.npz` stores `preproc.features`, `preproc.means`, `preproc.scales`, `preproc.percentile_levels` (0,1,5,25,50,75,95,99,100), `preproc.percentiles` (9x6 raw units) and `preproc.medians` (6, raw units) for the `/api/status` slider limits (D-013).

## Synthetic fallback sessions (tabular head, synthetic source)

| Field | Value |
|---|---|
| File | `data/synthetic/online_shoppers_synthetic.csv` (generated locally, same schema: six features + `Revenue`) |
| Generator | `python -m scripts.distill_data`, `GENERATOR_VERSION = 1`, seed 41, N = 6,000 |
| Params | `u_sigma 0.6, rate_A 2.0, rate_I 0.6, rate_P 12.0, dur_scale_s 25.0, bias -3.0, w_logP 0.45, w_logA 0.15, w_anyI 0.25` |
| Format | LF line ends, rows in generation order, counts as integers, durations with 4 decimals |
| SHA-256 | `b299f270c6eef97a9ce4f38a3215f93455f496f9290d36335c3d38e1133f52a6` (`SYNTHETIC_SHA256`). Reproduced on two runs and by `test_synthetic_csv_reproducible`. Prepare refuses a CSV with another digest. |
| Actual prevalence | 1,014 / 6,000 = 0.1690 |
| Teacher | `--teacher cli|cloud` exits with "not implemented" (milestone M5). |

- Prepare with `python -m scripts.prepare_data tabular --source synthetic`, which writes `data/prepared/tabular_synthetic.npz` with dataset id `synthetic-v1-s41-sha256:b299f270c6ee-strat-s31-70/15/15`. The split is the same stratified 70/15/15 with seed 31: fit 4,200 / val 900 / test 900.
- Train with `python -m scripts.train tabular --source synthetic` to get `models/tabular_synthetic.npz` (`source_kind` "synthetic", with generator version/seed/rows/params in the manifest). The real model stays at `models/tabular.npz`.

## Benchmarks (Gate 3)

`python -m scripts.benchmark vision`, run 2026-11-12 (rerun after the T-201 rework). Each run is a fresh process: start, cache load, train, eval and archive save (to a temp directory). Power source (plugged in) was not verified.

| Metric | Value |
|---|---|
| CPU | AMD Ryzen 5 5500U with Radeon Graphics |
| Python / NumPy / BLAS | 3.13.7 / 2.5.4 / scipy-openblas 0.3.34.106.0 |
| Threads | OPENBLAS/MKL/OMP_NUM_THREADS=1 |
| Rows (fit) / epochs / seed | 8,000 / 6 / 17 |
| Wall time per run (s) | 2.15, 2.24, 2.13 |
| Median / slowest (s) | 2.15 / 2.24 |
| Peak working set (bytes) | 56,078,336 / 55,984,128 / 56,254,464 |
| Test accuracy (each run) | 0.9330 |
| Gate (< 120 s and < 250,000,000 B every run) | PASS |

`python -m scripts.benchmark names` and `python -m scripts.benchmark tabular` (UCI), run 2026-11-12 on the same machine and settings as above. The vision benchmark was rerun after the shared-script change: 1.95 / 1.88 / 1.87 s, test accuracy 0.9330, gate PASS.

| Metric | names | tabular (UCI) |
|---|---|---|
| Rows (fit) / epochs / seed | 22,899 windows / 8 / 23 | 8,631 / 12 / 31 |
| Wall time per run (s) | 1.16, 1.16, 1.17 | 0.78, 0.79, 0.78 |
| Median / slowest (s) | 1.16 / 1.17 | 0.78 / 0.79 |
| Peak working set (bytes) | 47,808,512 / 47,656,960 / 47,476,736 | 42,434,560 / 42,328,064 / 42,434,560 |
| Metric (each run) | test CE 2.3460 | test log loss 0.4025 |
| Gate (< 120 s and < 250,000,000 B every run) | PASS | PASS |
