import torch
import json
from pathlib import Path

DB_PATH = Path("data/enrollment_db.pt")


def _load_db() -> dict[str, torch.Tensor]:
    if DB_PATH.exists():
        return torch.load(str(DB_PATH), weights_only=True)
    return {}


def _save_db(db: dict[str, torch.Tensor]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(db, str(DB_PATH))


def enroll_user(user_id: str, embeddings: list[torch.Tensor]) -> None:
    """Store the mean embedding for a user from one or more enrollment samples."""
    db = _load_db()
    stacked = torch.stack(embeddings)
    mean_embedding = stacked.mean(dim=0)
    mean_embedding = torch.nn.functional.normalize(mean_embedding, dim=0)
    db[user_id] = mean_embedding
    _save_db(db)
    print(f"Enrolled user '{user_id}' with {len(embeddings)} sample(s).")


def get_enrollment(user_id: str) -> torch.Tensor:
    """Retrieve stored embedding for user_id. Raises KeyError if not found."""
    db = _load_db()
    if user_id not in db:
        raise KeyError(f"User '{user_id}' not found in enrollment DB.")
    return db[user_id]


def list_users() -> list[str]:
    return list(_load_db().keys())


def delete_user(user_id: str) -> None:
    db = _load_db()
    if user_id in db:
        del db[user_id]
        _save_db(db)
