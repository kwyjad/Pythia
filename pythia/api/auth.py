from fastapi import Header, HTTPException
from pythia.config import load as load_cfg


def require_token(x_pythia_token: str | None = Header(default=None)):
    cfg = load_cfg()
    want = set(cfg["security"]["api_tokens"])
    if not x_pythia_token or x_pythia_token not in want:
        raise HTTPException(status_code=401, detail="Unauthorized")
