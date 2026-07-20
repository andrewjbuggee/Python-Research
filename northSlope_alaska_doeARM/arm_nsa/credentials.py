"""ARM Live web service credential handling.

The ARM Live Data Web Service (https://adc.arm.gov/armlive/) authenticates
every request with `user=<username>:<access_token>`. Registration is free:

    1. Create an ARM account:  https://adc.arm.gov/armuserreg/#/new
    2. Log in at             https://adc.arm.gov/armlive/
       and copy the access token shown for your account.

Credentials are looked up in this order (first hit wins):

    1. Environment variables ARM_LIVE_USERNAME and ARM_LIVE_TOKEN
       (preferred on shared systems / HPC batch jobs).
    2. JSON file ~/.armlive_credentials.json
    3. JSON file <repo>/.armlive_credentials.json (gitignored)

The JSON file format is simply:

    {"username": "your_arm_username", "token": "your_access_token"}

Use `save_credentials()` once to create the file with correct permissions, or
write it by hand. NEVER commit the token to git; .gitignore already excludes
the repo-local file name.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple, Optional

_CRED_FILENAME = ".armlive_credentials.json"
_HOME_CRED_PATH = Path.home() / _CRED_FILENAME
_REPO_CRED_PATH = Path(__file__).resolve().parent.parent / _CRED_FILENAME


class ArmCredentials(NamedTuple):
    """ARM Live username/token pair."""

    username: str
    token: str

    def as_query_value(self) -> str:
        """Format for the `user=` query parameter of ARM Live requests."""
        return f"{self.username}:{self.token}"


def _from_env() -> Optional[ArmCredentials]:
    username = os.environ.get("ARM_LIVE_USERNAME")
    token = os.environ.get("ARM_LIVE_TOKEN")
    if username and token:
        return ArmCredentials(username=username, token=token)
    return None


def _from_file(path: Path) -> Optional[ArmCredentials]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as err:
        raise RuntimeError(f"Could not parse ARM credentials file {path}: {err}")
    username = payload.get("username", "")
    token = payload.get("token", "")
    if username and token:
        return ArmCredentials(username=username, token=token)
    return None


def get_credentials() -> ArmCredentials:
    """Return ARM Live credentials, raising with instructions if absent."""
    for source in (_from_env(), _from_file(_HOME_CRED_PATH), _from_file(_REPO_CRED_PATH)):
        if source is not None:
            return source
    raise RuntimeError(
        "ARM Live credentials not found. Either set the environment variables "
        "ARM_LIVE_USERNAME and ARM_LIVE_TOKEN, or create "
        f"{_HOME_CRED_PATH} containing "
        '{"username": "...", "token": "..."}. Register (free) at '
        "https://adc.arm.gov/armuserreg/#/new and find your token at "
        "https://adc.arm.gov/armlive/"
    )


def save_credentials(username: str, token: str, to_home: bool = True) -> Path:
    """Write a credentials file (default ~/.armlive_credentials.json).

    Parameters
    ----------
    username, token:
        Your ARM account name and ARM Live access token.
    to_home:
        If True write to the home directory (recommended, works for every
        project); otherwise write into the repo root (gitignored).
    """
    path = _HOME_CRED_PATH if to_home else _REPO_CRED_PATH
    path.write_text(json.dumps({"username": username, "token": token}, indent=2))
    path.chmod(0o600)  # user read/write only; the token is a secret
    return path
