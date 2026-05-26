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
- 🛡️ **Anti-deepfake defense** — detects modern TTS synthesis (ElevenLabs, VALL-E, Bark, etc.)
- 🔁 **Replay attack detection**
- 🧩 **Modular architecture** — swap any component as better models emerge
- 🌐 **Accent-diverse** — trained and evaluated on global speech datasets

---

## 🏗️ Architecture

```
Audio Input
    ↓
Preprocessing (VAD, normalization, resampling)
    ↓
┌──────────────────────┬──────────────────────┐
│ Speaker Verification │  Anti-Spoofing       │
│ (multilingual)       │  (deepfake + replay) │
└──────────────────────┴──────────────────────┘
    ↓                            ↓
       Decision Fusion Layer
              ↓
       Accept / Reject / Retry
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch** + **torchaudio**
- **Hugging Face Transformers**
- **SpeechBrain** — speaker recognition toolkit
- **pyannote.audio** — embeddings + VAD
- **FastAPI** — inference API
- **librosa**, **soundfile** — audio processing

### Models Used

| Component | Model |
|---|---|
| Speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` |
| Multilingual embeddings | `microsoft/wavlm-base-plus-sv` |
| Foundation model | `facebook/wav2vec2-xls-r-300m` |
| Anti-spoofing | AASIST / RawNet3 |

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

- Python 3.10 or higher
- pip
- (Optional) CUDA-capable GPU for training

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/vocalyx.git
cd vocalyx
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Quick Start

```bash
# Run baseline verification on sample audio
python -m src.verification.speaker_verifier --enroll sample1.wav --verify sample2.wav
```

---

## 📁 Project Structure

```
vocalyx/
├── data/                  # Audio datasets and test matrix
├── src/
│   ├── preprocessing/     # Audio loading, VAD, normalization
│   ├── enrollment/        # User enrollment + embedding storage
│   ├── verification/      # Speaker verification logic
│   ├── antispoofing/      # Deepfake + replay detection
│   ├── decision/          # Score fusion layer
│   ├── evaluation/        # Metrics and benchmarking
│   └── api/               # FastAPI inference service
├── notebooks/             # Experiments and analysis
├── tests/                 # Unit and integration tests
├── configs/               # Model configurations
└── docs/                  # Documentation
```

See [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) for the full technical brief.

---

## 🗺️ Roadmap

- [ ] **Phase 1:** Baseline speaker verification pipeline
- [ ] **Phase 2:** Evaluation framework and test matrix
- [ ] **Phase 3:** Cross-language and accent robustness
- [ ] **Phase 4:** Device and channel robustness
- [ ] **Phase 5:** Anti-spoofing and deepfake detection
- [ ] **Phase 6:** Decision fusion and adaptive logic
- [ ] **Phase 7:** API and production readiness
- [ ] **Phase 8:** Final validation and reporting

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Miguel** — built as part of a voice biometrics enhancement project.
