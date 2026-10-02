"""ARM Live credentials: your ARM username plus the access token shown after
logging in at https://adc.arm.gov/armlive/ .

Looked up in this order, so nothing secret ever lives in the repository:
  1. environment variables ARM_USERNAME and ARM_TOKEN
  2. the file ~/.arm_credentials (or the path in ARM_CREDENTIALS_FILE):
         [arm]
         username = your_arm_username
         token = your_access_token
  3. an interactive prompt (terminal only), which offers to write that file
"""
from __future__ import annotations

import configparser
import getpass
import os
import stat
import sys
from pathlib import Path
from typing import Tuple

HOW_TO = """ARM credentials not found.
Log in at https://adc.arm.gov/armlive/ to see your access token, then either
  export ARM_USERNAME=<your ARM username>
  export ARM_TOKEN=<your access token>
or create {path} containing
  [arm]
  username = <your ARM username>
  token = <your access token>
and run: chmod 600 {path}"""


def credentials_file() -> Path:
    return Path(os.environ.get("ARM_CREDENTIALS_FILE", "~/.arm_credentials")).expanduser()


def get_credentials(interactive: bool = True) -> Tuple[str, str]:
    user, token = os.environ.get("ARM_USERNAME"), os.environ.get("ARM_TOKEN")
    if user and token:
        return user.strip(), token.strip()

    path = credentials_file()
    if path.is_file():
        _warn_if_shared(path)
        cp = configparser.ConfigParser(interpolation=None)
        cp.read(path)
        try:
            user, token = cp["arm"]["username"].strip(), cp["arm"]["token"].strip()
        except KeyError:
            raise RuntimeError(f"{path} needs an [arm] section with 'username' and 'token'.") from None
        if user and token:
            return user, token
        raise RuntimeError(f"{path} has an empty username or token.")

    if interactive and sys.stdin.isatty():
        return _prompt_and_save(path)
    raise RuntimeError(HOW_TO.format(path=path))


def _prompt_and_save(path: Path) -> Tuple[str, str]:
    print("ARM Live credentials are needed once. Your token is shown after logging in at")
    print("https://adc.arm.gov/armlive/")
    user = input("ARM username: ").strip()
    token = getpass.getpass("ARM access token (input hidden): ").strip()
    if not user or not token:
        raise RuntimeError("Both a username and a token are required.")
    answer = input(f"Save them to {path} (readable only by you)? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"[arm]\nusername = {user}\ntoken = {token}\n")
        os.chmod(path, 0o600)
        print(f"Saved {path}")
    return user, token


def _warn_if_shared(path: Path) -> None:
    try:
        if path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH):
            print(f"Warning: {path} is readable by other users; run: chmod 600 {path}", file=sys.stderr)
    except OSError:
        pass
