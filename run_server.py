"""Entry point for the voice biometrics API server."""

import uvicorn
import yaml
from pathlib import Path


def main():
    config_path = Path("configs/api.yaml")
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f).get("server", {})

    uvicorn.run(
        "src.api.server:app",
        host=cfg.get("host", "0.0.0.0"),
        port=cfg.get("port", 8000),
        workers=cfg.get("workers", 1),
        reload=cfg.get("reload", False),
        log_level=cfg.get("log_level", "info"),
    )


if __name__ == "__main__":
    main()
