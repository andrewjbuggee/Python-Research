import datetime as dt
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_mock import MockArm, make_archive  # noqa: E402

START, END = dt.date(2023, 2, 15), dt.date(2023, 2, 24)
USER, TOKEN = "tester", "tok123"


@pytest.fixture(autouse=True)
def _local_only(monkeypatch):
    """Keep the mock server off any HTTP proxy configured on this machine."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def archive(tmp_path):
    return make_archive(tmp_path / "server", START, END)


@pytest.fixture
def server(archive):
    mock = MockArm(archive, USER, TOKEN).start()
    yield mock
    mock.stop()


@pytest.fixture
def env(monkeypatch, server, tmp_path):
    monkeypatch.setenv("ARM_LIVE_URL", server.url + "/armlive")
    monkeypatch.setenv("ARM_CITATION_URL", server.url + "/citation")
    monkeypatch.setenv("ARM_USERNAME", USER)
    monkeypatch.setenv("ARM_TOKEN", TOKEN)
    monkeypatch.setenv("ARM_CREDENTIALS_FILE", str(tmp_path / "no_credentials_here"))
    monkeypatch.setenv("EPCAPE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("EPCAPE_MACHINE", raising=False)
    monkeypatch.delenv("EPCAPE_CONFIG", raising=False)
    return tmp_path
