# FocusCore
Real-time Deep Learning system for detecting student engagement and distraction during in-person lectures

## Project documents

- [Final application](final_application.md)
- [User scenarios](final_scripts.md)
- [Changes compared to the initial application](changes.md)
- Initial versions are kept in [OLD](OLD/)

---

## Classroom Group Engagement Recognition

Automatic recognition of group engagement levels from classroom video recordings, with post-session engagement timeline reports.

## Quick start

### Desktop application (macOS, Apple Silicon)

```bash
git clone https://github.com/Ayanami-Rei1/FocusCore.git
cd FocusCore
conda create -n focuscore python=3.10 -y && conda activate focuscore
pip install -r requirements-app.txt
python -m scripts.download_model
python -m app.main
```

Register in the «Профиль» section: accounts, synchronization and speech
transcription work through the project's Supabase backend right away
(see `docs/supabase_setup.md` for how the backend is set up). On the first
lecture macOS asks for camera and microphone access for the terminal.

### Analysis core in Docker

The same analysis as in the application, without the interface: a video file
goes in, a JSON report with the engagement series and critical moments and a
PDF report come out.

```bash
docker pull ghcr.io/ayanami-rei1/focuscore:latest
docker run --rm -v "$PWD":/data ghcr.io/ayanami-rei1/focuscore /data/lecture.mp4
docker run --rm ghcr.io/ayanami-rei1/focuscore --help
```

The desktop application transcribes critical moments through the account, so
no key is needed there. The Docker image has no account and analyses
engagement without transcription; to add it, pass your own OpenAI key:
`-e OPENAI_API_KEY=sk-...`.

To build the image locally: `docker build -f docker/analysis/Dockerfile -t focuscore .`

### Tests

```bash
pip install -r requirements-test.txt
pytest -q
```

GitHub Actions runs the tests on every push, then builds the Docker image,
checks it on a generated video and publishes it (`.github/workflows/ci.yml`).

## Problem

Teachers cannot objectively assess collective student engagement during lessons. Manual observation is subjective and inconsistent.

## How it works

The application reads a camera or a video file, takes 8 frames at 3.75 fps
(the same sampling as during training) and runs a SLOW R50 video network
(TorchScript, MPS or CPU) on every clip. The scores are smoothed into an
engagement curve 0..100; stretches below the threshold become critical
moments, which are transcribed from the microphone or the sound track and
listed in a PDF report. Sessions are stored locally in SQLite and, with an
account, synchronized through Supabase.

The model is trained on the OUC-CGE dataset (classroom video, three levels of
group engagement). The official split of the dataset leaks scenes between its
parts, so the shipped model is trained on a scene-disjoint split; see
`docs/scene_disjoint_split.md`.

## Project structure

```
FocusCore/
├── app/                        # desktop application
│   ├── main.py                 #   entry point (PyQt6)
│   ├── cli.py                  #   command-line analysis (Docker image)
│   ├── core/                   #   analysis without Qt or a database: model, clips, smoothing, reports
│   ├── db/                     #   local SQLite storage
│   ├── cloud/                  #   Supabase accounts and synchronization
│   └── ui/                     #   windows, pages, chart, theme
├── src/                        # training code
│   ├── train.py, evaluate.py   #   Lightning entry points (Hydra configs)
│   ├── conf/                   #   structured configs
│   ├── data/                   #   OUC-CGE dataset and transforms
│   ├── models/                 #   LightningModule, model factory, PyTorchVideo backbones
│   └── training/               #   callbacks: MLflow logging, early stopping
├── configs/experiment/         # experiment presets
├── scripts/                    # export, benchmarks, demo data, scene-disjoint split
├── data/                       # DVC pointers to the clips, split files
├── docker/                     # analysis image (CPU) and training image (GPU)
├── docs/                       # backend setup, model statistics, split description
├── supabase/                   # Edge Function for transcription
└── tests/                      # pytest
```

## Training

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
dvc pull data/ouc-cge-320.dvc                                   # clips at 320p
python scripts/scene_split/materialize_splits.py --names grp27a  # scene-disjoint root
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.train +experiment=baseline_ouc_cge data.root=data/grp27a
python -m src.evaluate +checkpoint_path=outputs/.../best.ckpt model=slow
python -m scripts.export_torchscript --ckpt <ckpt> --out app/models/engagement_slow_r50.ts
```

`RUNS="R5" bash scripts/scene_split/train_on_splits.sh` runs the shipped
recipe end to end. MLflow tracks the runs (`mlflow ui --port 5000`), DVC
stores the clips and the checkpoints. A GPU container for training:
`docker compose up -d` (`docker/training/Dockerfile`).

## Requirements files

| file | contents |
|---|---|
| `requirements.txt` | training: torch, lightning, hydra, MLflow, DVC |
| `requirements-app.txt` | desktop application: PyQt6, pyqtgraph, OpenCV, torch for inference |
| `requirements-core.txt` | analysis core without the interface (Docker image, CI) |
| `requirements-test.txt` | unit tests |
| `requirements-docker.txt` | training image on the vast.ai base image |

## Known limitations

- The engagement model is trained on OUC-CGE and expects a camera that sees
  the whole audience; a single person in the frame is scored as low engagement.
- Times of critical moments in a report are counted from the start of the
  lecture without pauses.
- On the scene-disjoint split the model reaches 0.52 test accuracy against
  0.39 for the majority class; the 0.99 reported for the official split
  measures scene memorisation.
