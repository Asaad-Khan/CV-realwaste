# RealWaste Image Classification

A computer vision prototype for classifying waste images into nine material
categories, with reproducible data preparation, CPU training, and a
Dockerized FastAPI inference service.

## Current status

- Dataset audit and similarity-aware splits completed.
- Small CNN baseline trained and evaluated on validation data.
- Saved-checkpoint inference reproduces validation metrics.
- All nine HTTP checks passed locally and in a Linux Docker container.
- ## Live demo

**Try the app:** https://asaad-realwaste.streamlit.app/

Upload a JPEG, PNG, or WebP image and click **Classify image**.
The app displays the predicted waste category and top-three model scores.

The deployed app currently uses our trained SmallCNN baseline.
Scores are uncalibrated softmax outputs, and predictions can be incorrect.

Hosted on Streamlit Community Cloud. If the app is asleep, click the
wake-up button and allow it to start.

A separate FastAPI inference service and Dockerfile are also included
for running the REST API locally.
- Transfer-learning model and final test evaluation: pending.

## Dataset

RealWaste contains 4,752 images across nine classes:

Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper,
Plastic, Textile Trash, and Vegetation.

All images were readable and RGB. No exact file duplicates or identical
decoded images were found.

A difference-hash similarity check identified 213 candidate pairs.
Visual inspection found both repeated views and false matches. Connected
candidate pairs were conservatively kept in the same split, without
deleting images or changing labels.

These groups are heuristic similarity groups, not verified object identities.
The method may miss related images and may group unrelated images.
The largest group contains 84 images.

| Split | Images |
|---|---:|
| Training | 3,394 |
| Validation | 679 |
| Test | 679 |

Split assignments are stored in `reports/splits/manifest.csv`.
The test set is reserved for final evaluation.

## Baseline approach

The baseline is a CNN with 98,089 trainable parameters, trained from
random initialization.

- Input size: 160 × 160 RGB.
- ImageNet mean and standard deviation used for normalization.
- Training augmentation: horizontal flips, modest rotations, and color jitter.
- Validation and inference: deterministic resizing and normalization.
- Optimizer: AdamW.
- Loss: unweighted cross-entropy.
- Checkpoint selection: highest validation macro F1.
- Training hardware: Intel i5-13420H CPU; no GPU acceleration.

Macro F1 is the primary selection metric because class sizes differ.

## Validation results

| Experiment | Best epoch | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Initial 12-epoch budget | 12 | 0.6362 | 0.6311 |
| Extended 30-epoch budget | 14 | 0.6966 | 0.6863 |

The extended run stopped at epoch 20 after six epochs without improved
validation macro F1.

The selected checkpoint is bundled as `models/baseline.pt`.
Full experiment configurations and histories are in `reports/training/`.

## Run the API with Docker

From the repository root:

```bash
docker build -t realwaste-api:baseline .
docker run --rm -p 8000:8000 --memory=1g --memory-swap=1g realwaste-api:baseline
```

Open http://localhost:8000/docs to upload an image.

Endpoints:

- `GET /health`: model readiness and version.
- `POST /predict`: predicted class and top-three softmax scores.
- `GET /docs`: interactive API documentation.

The model loads once at startup. The service accepts JPEG, PNG, and
non-animated WEBP images, with a 5 MiB file-size limit and a
12-million-pixel image limit.

The handler processes one prediction at a time and returns HTTP 503
when busy. File-size validation occurs after multipart parsing; it is
not a hosting-level request-body limit.

## Local development on Windows PowerShell

```powershell
python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt --extra-index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt

$env:MODEL_PATH = "models\baseline.pt"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

With the dataset downloaded and extracted, audit and training commands are:

```powershell
.\.venv\Scripts\python.exe -m src.data.audit
.\.venv\Scripts\python.exe -m src.data.review
.\.venv\Scripts\python.exe -m src.data.split
.\.venv\Scripts\python.exe -m src.train --config configs/baseline_long.yaml
```

Expected dataset directory:
`data/raw/realwaste-main/RealWaste/`.

The committed split manifest defines the experiments. Regeneration should
be checked against the original before using it for comparisons.

## HTTP checks

With the service running and dataset available:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_api --base-url http://127.0.0.1:8000
```

Checks cover readiness, valid images, empty uploads, invalid content,
unsupported formats, oversized files, excessive image dimensions,
missing upload fields, and service availability after invalid requests.

## Limitations

- The model predicts a single category per image.
- Repeated backgrounds and collection conditions may affect generalization.
- Some material categories have ambiguous visual boundaries.
- Softmax scores are not calibrated probabilities of correctness.
- The model may confidently misclassify images, including unrelated inputs.
- Validation results do not establish performance at other collection sites.

The initial container reported 495.8 MiB usage after HTTP checks under
a 512 MiB limit. Local development was moved to a 1 GiB limit to provide
headroom. Public-host memory requirements remain to be verified.

## Dataset attribution

RealWaste was created by Sam Single, Saeid Iranmanesh, and Raad Raad.

- UCI dataset: https://archive.ics.uci.edu/dataset/908/realwaste
- Creator repository: https://github.com/sam-single/realwaste
- Paper: https://doi.org/10.3390/info14120633

The creator repository lists CC BY-NC-SA 4.0, while the UCI entry lists
CC BY 4.0. This source discrepancy is recorded here. Dataset sample
figures in the reports derive from RealWaste; the original image
archive is not included in this repository.