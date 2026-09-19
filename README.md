# Medical ASL Recognition Model V1.1 — MS-ASL

Training-only isolated word-level ASL recognition for an exact medically
relevant subset of the Microsoft Research MS-ASL dataset. This revision uses
MS-ASL as its sole labeled dataset while retaining the original project's
MediaPipe preprocessing contract, official-split discipline, candidate models,
and reserved-test policy.

This project does not implement webcam recognition, continuous recognition,
sign segmentation, sentence translation, a UI, or test-set evaluation.

## Dataset and license

The metadata package must contain the official files:

```text
data/msasl/MSASL_train.json
data/msasl/MSASL_val.json
data/msasl/MSASL_test.json
data/msasl/MSASL_classes.json
data/msasl/MSASL_synonym.json
data/msasl/C-UDA-0.1_annotated_discussion.pdf
```

MS-ASL is distributed under Microsoft's Computational Use of Data Agreement
(C-UDA). Review and comply with the included agreement. Downloaded source clips
are excluded from Git because their availability and rights remain governed by
their sources and the dataset agreement.

The requested 32-concept list is matched against `clean_text` exactly. The
pipeline neither invents missing labels nor silently substitutes synonyms.

## Requirements

- Python 3.11 or 3.12, 64-bit (this run was verified on Python 3.12.14)
- Internet access for historical MS-ASL source URLs
- Local CPU, memory, and storage

Create an environment and install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`imageio-ffmpeg` supplies a local FFmpeg executable. The downloader uses it to
fetch only each official `start_time`/`end_time` interval instead of retaining
entire source videos.

## Reproduce the pipeline

Run from the project root:

```bash
python src/select_medical_subset.py
python src/download_videos.py
python src/build_dataset.py
python src/train.py
```

### 1. Selection

`select_medical_subset.py` reads all three official MS-ASL metadata files,
preserves their train/validation/test assignments and signer IDs, and writes
`data/medical_subset.json` using exact requested-gloss matches only.

### 2. Acquisition

`download_videos.py` downloads each annotated time interval as its own clip.
Intervals sharing one source URL are grouped into a single extraction request,
and source requests are throttled to avoid YouTube guest-session rate limits.
The script skips valid existing clips, rejects HTML/error bodies and
non-decodable files, logs unavailable historical URLs, and continues after
individual failures. No source failure terminates the complete acquisition
pass.

### 3. Preprocessing

MediaPipe Holistic supplies anatomical left/right hands and six upper-body pose
points. Each frame has exactly 144 ordered features:

```text
21 left-hand landmarks × xyz
21 right-hand landmarks × xyz
6 pose landmarks × xyz
```

Missing points retain all-zero slots. Detected points use the 3D shoulder
midpoint as origin and shoulder distance as scale. A deterministic centroid and
maximum-radius fallback is used when shoulders are unavailable. Every isolated
clip becomes a `30 × 144` sequence: long clips use evenly spaced frames and
short clips use per-feature linear temporal interpolation.

Classes with fewer than five total usable samples after acquisition and
preprocessing are excluded and documented. Official splits are never shuffled.

### 4. Training and selection

Each sequence is flattened in C order to 4,320 values. The script fits only:

- `StandardScaler` + probability-enabled balanced RBF `SVC`
- Balanced 300-tree `RandomForestClassifier`

Both candidates train only on `X_train`/`y_train`. Selection uses MS-ASL
validation macro F1, then validation accuracy and per-sample prediction latency.
The reserved test features are generated but never loaded or evaluated by the
training script.

## Completed MS-ASL run

The checked-in artifacts were generated from the included metadata with these
actual results:

- 32 requested concepts; 18 exact MS-ASL matches; 14 missing
- 747 annotated instances requested
- 451 valid clips and MediaPipe sequences; 296 unavailable source instances
- 326 training, 82 validation, and 43 reserved test samples
- Random Forest selected: validation accuracy 0.3780, macro F1 0.3275
- SVC comparison: validation accuracy 0.3293, macro F1 0.2854

These metrics describe a limited V1 research baseline. The model is not
validated for clinical decisions or patient-safety use. The test split remains
unevaluated for a later formal evaluation phase.

## Outputs

Processed data:

```text
processed/X_train.npy  processed/y_train.npy
processed/X_val.npy    processed/y_val.npy
processed/X_test.npy   processed/y_test.npy
processed/classes.json
```

Model artifacts:

```text
models/medical_asl_model.pkl
models/model_metadata.json
models/label_classes.json
logs/dataset_report.json
```

The serialized classifier supports `predict_proba`. Future inference must
reproduce the exact feature layout, normalization, temporal processing, and
flattening order stored in `model_metadata.json`.

If source attrition leaves insufficient training or validation data, training
stops with a specific error rather than generating a fabricated model.
