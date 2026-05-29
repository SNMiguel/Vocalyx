# 🎙️ Vocalyx

**A complete voice biometric authentication system — neural speaker verification, deepfake detection, challenge-response, and a full React dashboard.**

Vocalyx is a production-ready voice authentication platform that verifies speakers reliably across real-world conditions — different devices, languages, accents, and vocal states — while defending against modern AI voice cloning and replay attacks.

---

## ✨ What It Does

| Capability | How |
|---|---|
| **Speaker verification** | ECAPA-TDNN embeddings (SpeechBrain) — cosine similarity against enrolled voice profile |
| **Neural anti-spoof** | Wav2Vec2 deepfake classifier (HuggingFace) — trained to detect TTS synthesis |
| **Replay attack detection** | Handcrafted spectral features: reverb tail, sub-band energy ratio, noise floor, spectral flux |
| **Challenge-response** | 4 random words per session — Whisper transcription + fuzzy matching ensures liveness |
| **Adaptive session logic** | Retry escalation, threshold tightening on retries, lockout after 4 non-accepts |
| **Rate limiting** | 10 session starts per IP per 60 seconds — HTTP 429 on breach |
| **Probe quality gate** | Rejects recordings with < 2s of speech after VAD trim, before any ML inference |
| **Session persistence** | Completed sessions written to SQLite — survives server restarts |
| **Audit logging** | Every enroll, delete, and role change recorded with actor, target, and timestamp |
| **Role-based access** | `admin` / `ops` / `user` roles enforced on all endpoints and UI routes |

---

## 🖥️ Dashboard

A React 18 + Vite dashboard ships alongside the API.

| Page | Access | What it shows |
|---|---|---|
| **Dashboard** | All | Stat cards (enrolled users, sessions, accept rate, locked accounts), recent sessions, system status, quick actions |
| **Authenticate** | All | Live recording with challenge phrase + countdown timer, per-attempt decision card |
| **Enroll** | All / Admin | Upload voice samples; shows enrollment status badge (new or overwrite) |
| **Sessions** | Admin, Ops | Full session history, auto-refreshes every 30s; expandable rows show challenge phrase and per-attempt scores |
| **Voice Users** | Admin | List and delete enrolled voice profiles |
| **App Users** | Admin | Manage dashboard accounts and roles with inline confirmation |
| **Audit Log** | Admin | Chronological table of all admin actions |

---

## 🏗️ Architecture

```
Audio Upload (WAV · FLAC · OGG · M4A · MP3)
          ↓
  ┌───────────────────────────────────────────┐
  │            Preprocessing                  │
  │  Decode → Mono → Resample (16kHz)         │
  │  RMS Normalize → Energy VAD trim          │
  │  Probe quality gate (≥ 2s of speech)      │
  └───────────────────────────────────────────┘
          ↓
  ┌───────────────────────────────────────────┐
  │         Challenge Verification            │
  │  Whisper base transcribes probe audio     │
  │  Fuzzy-matches all 4 challenge words      │
  │  Fail → 401 before speaker check runs     │
  └───────────────────────────────────────────┘
          ↓
  ┌──────────────────┬────────────────────────┐
  │ Speaker Verif.   │ Anti-Spoofing          │
  │ ECAPA-TDNN       │ Wav2Vec2 deepfake      │
  │ cosine similarity│ classifier (HuggingFace│
  │ vs. enrollment   │ label 0=fake, 1=real)  │
  │                  │                        │
  │ Channel mismatch │ Replay detector        │
  │ detection        │ (spectral heuristics)  │
  └──────────────────┴────────────────────────┘
          ↓
  ┌───────────────────────────────────────────┐
  │           Decision Fusion                 │
  │  SV score (0.60) + spoof score (0.40)     │
  │  → ACCEPT / RETRY / REJECT / STEP_UP      │
  └───────────────────────────────────────────┘
          ↓
  ┌───────────────────────────────────────────┐
  │           Session Manager                 │
  │  Multi-attempt · threshold tightening     │
  │  Lockout after 4 non-accepts              │
  │  Terminal sessions logged to SQLite       │
  └───────────────────────────────────────────┘
          ↓
       FastAPI REST API + React Dashboard
```

---

## 🛠️ Tech Stack

**Backend**
- Python 3.10, PyTorch, torchaudio
- [SpeechBrain](https://speechbrain.github.io/) — ECAPA-TDNN speaker embeddings (`spkrec-ecapa-voxceleb`)
- [Hugging Face Transformers](https://huggingface.co/) — Wav2Vec2 deepfake classifier, Whisper transcription
- FastAPI + Uvicorn — REST API
- SQLite — session history, audit log, dashboard accounts
- imageio-ffmpeg — self-contained audio decoder for M4A/MP3 (no system FFmpeg needed)
- python-jose — JWT authentication

**Frontend**
- React 18 + Vite 5
- React Router v6
- Web Audio API (MediaRecorder → WAV via AudioContext)

### Models

| Component | Model |
|---|---|
| Speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` |
| Deepfake / anti-spoof | `motheecreator/Deepfake-audio-detection` (Wav2Vec2) |
| Challenge transcription | `openai/whisper-base` |

Model weights are downloaded once and cached in `~/.cache/huggingface/` and `~/speechbrain/`.

---

## 🚀 Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or Miniconda
- Python 3.10
- Node.js 18+

### Installation

```bash
git clone https://github.com/SNMiguel/Vocalyx.git
cd Vocalyx

conda create -n voice-biometrics python=3.10 -y
conda activate voice-biometrics
pip install -r requirements.txt

cd ui && npm install && cd ..
```

### Start the backend

```bash
conda activate voice-biometrics
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

> **First start takes ~30–60 seconds** — Wav2Vec2 and Whisper model weights are downloaded and cached on first run.

Verify it's ready:
```bash
curl http://localhost:8000/health
```

### Start the frontend

```bash
cd ui && npm run dev
# Dashboard at http://localhost:5173
```

### Default credentials

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | admin |

Change these in `configs/api.yaml` before deploying.

---

## 🔐 Authentication Flow

1. **Start a session** — `POST /sessions` returns a `session_id` and a 4-word `challenge` phrase with a 10-second countdown.
2. **Record audio** — user speaks the challenge phrase into the microphone.
3. **Submit probe** — `POST /authenticate` with the audio file and session ID.
4. **Pipeline runs** — challenge verified → speaker score computed → anti-spoof score computed → decision fused.
5. **Decision returned** — `ACCEPT`, `RETRY`, `REJECT`, or `STEP_UP`.

Sessions lock after 4 non-accept attempts. Threshold tightens by 0.02 per retry.

---

## 🌐 API Reference

### Auth

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `POST` | `/auth/login` | Public | Exchange credentials for a JWT |
| `POST` | `/auth/register` | Public | Create a new account (role: `user`) |
| `GET` | `/auth/me` | Authenticated | Current user info |
| `GET` | `/auth/users` | Admin | List all dashboard accounts |
| `PATCH` | `/auth/users/{username}` | Admin | Change a user's role |
| `DELETE` | `/auth/users/{username}` | Admin | Delete a dashboard account |

### Voice & Sessions

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `POST` | `/enroll` | Authenticated | Enroll a user with one or more audio files |
| `GET` | `/users` | Authenticated | List all enrolled voice users |
| `GET` | `/users/{user_id}` | Public | Check if a voice user is enrolled |
| `DELETE` | `/users/{user_id}` | Admin | Delete a voice enrollment |
| `POST` | `/sessions` | Authenticated | Start an authentication session |
| `GET` | `/sessions` | Admin, Ops | List all sessions (history + active) |
| `GET` | `/sessions/{id}` | Authenticated | Session status and attempt history |
| `POST` | `/authenticate` | Authenticated | Submit a probe audio for a session |

### Admin

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/audit` | Admin | Retrieve the audit log |
| `GET` | `/health` | Public | Service liveness and enrolled user count |
| `GET` | `/version` | Public | Pipeline version and component list |

Interactive API docs at `http://localhost:8000/docs`.

---

## 👥 Roles

| Role | Capabilities |
|---|---|
| `admin` | Full access — enroll anyone, delete profiles, manage accounts, change roles, view audit log |
| `ops` | Read-only — view sessions and session history |
| `user` | Self-service — enroll and authenticate as themselves only |

---

## 🔒 Security Notes

| Attack vector | Mitigation |
|---|---|
| Digital replay (same file re-submitted) | Upload removed from auth — live microphone recording only |
| Acoustic replay (phone playing pre-recorded audio) | Challenge-response: pre-recorded audio won't contain the session's 4 words |
| Wrong speaker | ECAPA-TDNN embedding distance — genuine users score 0.70+, impostors fall below 0.25 threshold |
| AI voice cloning (ElevenLabs, etc.) | Wav2Vec2 deepfake classifier + replay detector; neural TTS remains the hardest attack vector |
| Brute-force | Session lockout after 4 non-accepts; rate limit of 10 sessions/IP/min |
| Short/silent recordings | Probe quality gate rejects < 2s of speech after VAD |

---

## 📁 Project Structure

```
vocalyx/
├── configs/
│   └── api.yaml                # Server config: JWT secret, admin credentials, thresholds
├── src/
│   ├── preprocessing/
│   │   ├── audio_loader.py     # Load and resample to 16kHz
│   │   ├── normalization.py    # RMS normalization
│   │   ├── vad.py              # Energy-based VAD trim
│   │   └── denoiser.py         # noisereduce wrapper (disabled — conda DLL conflict)
│   ├── enrollment/
│   │   ├── embedder.py         # ECAPA-TDNN via SpeechBrain
│   │   └── enrollment_db.py    # .pt embedding store
│   ├── antispoofing/
│   │   ├── deepfake_detector.py  # Wav2Vec2 neural deepfake classifier
│   │   ├── replay_detector.py    # Spectral replay heuristics
│   │   ├── channel_mismatch.py   # Device-switch detection
│   │   └── fusion.py             # Deepfake + replay score fusion (0.60/0.40)
│   ├── decision/
│   │   ├── fusion_layer.py     # SV + spoof → ACCEPT/RETRY/REJECT/STEP_UP
│   │   └── session.py          # Session state, challenge generation, lockout
│   └── api/
│       ├── server.py           # FastAPI app, all endpoints, lifespan startup
│       ├── auth_router.py      # JWT login, register, user management
│       ├── app_db.py           # SQLite: dashboard accounts, sessions, audit log
│       └── models.py           # Pydantic request/response models
├── ui/
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.jsx   # Stat cards, recent sessions, system status
│       │   ├── Authenticate.jsx# Live recording, challenge countdown, DecisionCard
│       │   ├── Enroll.jsx      # Audio upload with enrollment status badge
│       │   ├── Sessions.jsx    # Session history with auto-refresh
│       │   ├── Users.jsx       # Voice profile management (admin)
│       │   ├── AppUsers.jsx    # Dashboard account management (admin)
│       │   └── AuditLog.jsx    # Admin action history (admin)
│       ├── components/
│       │   ├── Layout.jsx      # Sidebar, nav, role-aware links
│       │   └── ProtectedRoute.jsx
│       ├── contexts/
│       │   └── AuthContext.jsx # JWT auth state
│       ├── hooks/
│       │   └── useAudioRecorder.js  # MediaRecorder → WAV blob
│       └── api.js              # Typed fetch wrappers for all endpoints
├── data/
│   ├── app.db                  # SQLite database (auto-created)
│   └── embeddings/             # Enrolled voice profiles (.pt files)
└── requirements.txt
```

---

## ⚙️ Configuration

Edit `configs/api.yaml`:

```yaml
auth:
  secret_key: "change-me-in-production"
  token_expire_minutes: 60
  admin_username: admin
  admin_password: admin123
```

Decision thresholds (in `src/decision/fusion_layer.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `sv_accept_threshold` | 0.25 | Cosine similarity above this → speaker verified |
| `sv_retry_threshold` | 0.15 | Between 0.15–0.25 → retry |
| `spoof_reject_threshold` | 0.50 | Spoof score above this → reject |
| `spoof_retry_threshold` | 0.35 | Between 0.35–0.50 → retry |

Genuine speaker scores 0.70+ in testing. Do not lower thresholds without new calibration data.

---

## ⚠️ Known Limitations

- **AI voice cloning** — modern neural TTS (ElevenLabs, RVC) could potentially pass both speaker verification and Wav2Vec2. A PIN or TOTP second factor would close this gap.
- **Rate limit not persisted** — in-memory buckets reset on server restart.
- **Noisereduce disabled** — a conda DLL conflict (`gdk_pixbuf`) crashes scipy on this environment. Re-enable after fixing the env and re-enroll all users (processing must match enrollment).
- **Single-process enrollment DB** — the `.pt` embedding store is not safe for concurrent writes across multiple Uvicorn workers. Use `--workers 1` or migrate to a proper DB for multi-worker deployments.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Miguel** — [github.com/SNMiguel](https://github.com/SNMiguel)
