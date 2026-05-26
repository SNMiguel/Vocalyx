# 🎙️ Vocalyx

**Robust voice biometric authentication and anti-spoofing system.**

Vocalyx is a production-grade voice authentication pipeline designed to verify speakers reliably across real-world conditions — different devices, languages, accents, and vocal states — while defending against modern AI-generated voice deepfakes.

---

## 🎯 Why Vocalyx?

Most voice biometric systems break when reality doesn't match training conditions. They lock out legitimate users who switch from a phone mic to AirPods, get sick, or speak in a different language than they enrolled in. At the same time, they're increasingly vulnerable to AI voice cloning tools like ElevenLabs and VALL-E.

Vocalyx is built around a different assumption: **reality is messy, and security tools need to handle it.**

---

## ✨ Key Features

- 🌍 **Cross-language authentication** — enroll in one language, authenticate in another
- 🎧 **Device-agnostic** — works across phone mics, AirPods, Android earbuds, wired headsets
- 🤒 **Vocal condition resilient** — handles illness, fatigue, emotional speech
- 🛡️ **Anti-deepfake defense** — spectral feature detection against TTS synthesis (ElevenLabs, VALL-E, Bark, etc.)
- 🔁 **Replay attack detection** — reverb tail, sub-band energy, and noise floor analysis
- 🔐 **Session management** — retry escalation, step-up auth, and lockout
- 🧩 **Modular architecture** — swap any component as better models emerge
- 🌐 **Accent-diverse** — language-adaptive scoring thresholds per region

---

## 🏗️ Architecture

```
Audio Input
    ↓
Preprocessing (VAD · RMS normalize · resample to 16kHz)
    ↓
┌───────────────────────────┬──────────────────────────────┐
│  Speaker Verification     │  Anti-Spoofing               │
│  ECAPA-TDNN (primary)     │  Deepfake detector           │
│  WavLM · XLS-R · MMS      │    spectral flatness         │
│  (multilingual fusion)    │    HNR · pitch jitter        │
│                           │    MFCC delta variance       │
│  Channel normalization    │  Replay detector             │
│  CMVN · WCCN              │    reverb tail energy        │
│                           │    sub-band ratio            │
│  Channel mismatch check   │    noise floor · spec flux   │
└───────────────────────────┴──────────────────────────────┘
                    ↓
           Decision Fusion Layer
      (adaptive SV threshold · spoof score)
                    ↓
     ACCEPT / REJECT / RETRY / STEP_UP
                    ↓
           Session Manager
    (retry escalation · lockout · history)
                    ↓
           FastAPI REST Service
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch** + **torchaudio**
- **SpeechBrain** — ECAPA-TDNN speaker embeddings
- **Hugging Face Transformers** — WavLM, XLS-R, MMS, Whisper
- **FastAPI** + **uvicorn** — inference REST service
- **soundfile** — audio I/O (no FFmpeg dependency)
- **pytest** — 141-test suite across 8 modules

### Models

| Component | Model |
|---|---|
| Speaker embeddings (primary) | `speechbrain/spkrec-ecapa-voxceleb` |
| Multilingual embeddings | `microsoft/wavlm-base-plus-sv` |
| Cross-lingual embeddings | `facebook/wav2vec2-xls-r-300m` |
| Massively multilingual | `facebook/mms-1b` |
| Language detection | `openai/whisper-base` |
| HF anti-spoof (optional) | `jungjee/HuBERT-base-AS` |

---

## 📊 Target Performance

| Metric | Target |
|---|---|
| EER (normal conditions) | < 3% |
| EER (adverse conditions) | < 6% |
| Cross-language accuracy | > 90% |
| Deepfake detection (known attacks) | > 95% |
| Deepfake detection (unseen attacks) | > 80% |

---

## 🚀 Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or Miniconda
- Python 3.10

### Installation

```bash
git clone https://github.com/SNMiguel/Vocalyx.git
cd Vocalyx

conda create -n voice-biometrics python=3.10 -y
conda activate voice-biometrics
pip install -r requirements.txt
```

### Run the API server

```bash
conda activate voice-biometrics
python run_server.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Enroll a user and authenticate

```bash
# Enroll
curl -X POST http://localhost:8000/enroll \
  -F "user_id=alice" \
  -F "files=@enroll.wav"

# Start a session
curl -X POST http://localhost:8000/sessions \
  -F "user_id=alice"

# Authenticate (use session_id from above)
curl -X POST http://localhost:8000/authenticate \
  -F "session_id=<session_id>" \
  -F "file=@probe.wav"
```

---

## 🧪 Running Tests

```bash
conda activate voice-biometrics

# Fast suite (no model downloads)
pytest tests/ -m "not slow" -q

# Full suite including multilingual model tests
pytest tests/ -q
```

### Run the end-to-end validation

```bash
PYTHONPATH=. python src/evaluation/validate.py
# Exercises all 11 pipeline components
# Saves report to data/validation_report.json
```

---

## 📁 Project Structure

```
vocalyx/
├── configs/
│   ├── api.yaml            # Server and pipeline config
│   └── baseline.yaml       # SV model and threshold defaults
├── src/
│   ├── preprocessing/      # Audio loading, VAD, normalization,
│   │                       # augmentation, CMVN/WCCN, channel norm
│   ├── enrollment/         # ECAPA-TDNN embedder, .pt enrollment DB
│   ├── verification/       # Speaker verifier, multilingual backends,
│   │                       # language-adaptive scoring
│   ├── antispoofing/       # Deepfake detector, replay detector,
│   │                       # channel mismatch, anti-spoof fusion
│   ├── decision/           # Auth decision fusion, session manager
│   ├── evaluation/         # EER/FAR/FRR metrics, benchmark runner,
│   │                       # end-to-end validation
│   └── api/                # FastAPI server and Pydantic models
├── tests/                  # 141 tests across all 8 phases
├── run_server.py           # Uvicorn entrypoint
└── requirements.txt
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/enroll` | Enroll a user with one or more audio files |
| `POST` | `/sessions` | Start an authentication session |
| `POST` | `/authenticate` | Submit a probe audio for a session |
| `GET` | `/sessions/{id}` | Get session status and attempt history |
| `GET` | `/users/{id}` | Check if a user is enrolled |
| `DELETE` | `/users/{id}` | Remove a user's enrollment |
| `GET` | `/health` | Service liveness check |
| `GET` | `/version` | Pipeline version and component list |

Interactive docs available at `/docs` when the server is running.

---

## 🗺️ Roadmap

- [x] **Phase 1:** Baseline speaker verification pipeline
- [x] **Phase 2:** Evaluation framework and test matrix
- [x] **Phase 3:** Cross-language and accent robustness
- [x] **Phase 4:** Device and channel robustness
- [x] **Phase 5:** Anti-spoofing and deepfake detection
- [x] **Phase 6:** Decision fusion and adaptive logic
- [x] **Phase 7:** API and production readiness
- [x] **Phase 8:** Final validation and reporting

### Known Limitations

- Spectral anti-spoof calibrated against ASVspoof 2019 LA; degrades on 2024 challenge attacks and unseen TTS systems
- WCCN requires ≥ 2 samples per speaker at dev time; unavailable at cold start
- Enrollment DB is a flat `.pt` file — not safe for concurrent writes in multi-worker deployments
- Session state is in-process only; lost on server restart
- Multilingual embedders (WavLM, XLS-R, MMS) are lazy-loaded; first request triggers a model download

### Next Steps

- Benchmark against VoxCeleb1-H (hard) and ASVspoof 2024
- Validate `jungjee/HuBERT-base-AS` end-to-end on real spoofed audio
- Collect real device audio and tune channel mismatch thresholds
- Replace flat `.pt` DB with SQLite or Redis for concurrent safety
- Add GPU batching and request queuing for production throughput
- GDPR / biometric compliance review before production deployment

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Miguel** — built as part of a voice biometrics enhancement project.
