"""Client for the ARM Live Data web service (https://adc.arm.gov/armlive/).

Three services are used:
  query     list a datastream's files between two dates
  saveData  download one complete file
  mod       download one file containing only chosen variables; ARM's server
            does the extraction, so e.g. cloud-base heights arrive without the
            much larger backscatter profiles
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import quote

import requests

from .arm_files import NETCDF_MAGIC, in_window

DEFAULT_BASE_URL = "https://adc.arm.gov/armlive"
DEFAULT_CITATION_URL = "https://adc.arm.gov/citationservice/citation/datastream"

# ARM's documentation uses /armlive/<service>; ARM's ACT toolkit uses
# /armlive/data/<service>. Both are tried and whichever answers is remembered.
ENDPOINTS = {
    "query": ("query", "data/query"),
    "saveData": ("saveData", "data/saveData"),
    "mod": ("mod",),
}
# Busy or overloaded server: retry. A 500 usually reflects the request itself
# (e.g. a variable a particular file lacks), so it is not retried.
RETRY_STATUS = {429, 502, 503, 504}


class ArmLiveError(RuntimeError):
    """A request failed in a way that retrying will not fix."""


class FileNotAvailable(ArmLiveError):
    """ARM Live cannot serve this file; it has to be ordered through Data Discovery."""


class ServerBusy(ArmLiveError):
    """Retries were exhausted on a busy server or a dropped connection.

    Kept separate from other failures so callers do not mistake it for a
    problem with the file itself: rerunning later is the remedy, not
    downloading the complete file instead."""


class _Transient(Exception):
    """A failure worth retrying (server busy, dropped connection, short transfer)."""


_TRANSIENT = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    _Transient,
)


class ArmLiveClient:
    def __init__(
        self,
        username: str,
        token: str,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        backoff: float = 2.0,
        timeout=(30, 600),
        log: Callable[[str], None] = print,
    ):
        self._user = username
        self._token = token
        self.base_url = (base_url or os.environ.get("ARM_LIVE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max(1, int(max_retries))
        self.backoff = backoff
        self.timeout = timeout
        self.log = log
        self._resolved: Dict[str, str] = {}
        self._local = threading.local()

    # -- plumbing ---------------------------------------------------------
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = "epcape-arm-tools/1.0 (python-requests)"
            self._local.session = session
        return session

    def redact(self, text) -> str:
        """Remove the access token from text that may contain a request URL."""
        text = str(text)
        for secret in {self._token, quote(self._token, safe="")}:
            if secret:
                text = text.replace(secret, "***")
        return text

    def _url(self, path: str, params: Dict[str, object]) -> str:
        # ARM requires `user` to be the first parameter.
        query = "user=" + quote(self._user, safe="") + ":" + quote(self._token, safe="")
        for key, value in params.items():
            if value is not None:
                query += f"&{key}={quote(str(value), safe=',:-._')}"
        return f"{self.base_url}/{path}?{query}"

    def _request(self, service: str, params: Dict[str, object], **kwargs) -> requests.Response:
        paths = [self._resolved[service]] if service in self._resolved else list(ENDPOINTS[service])
        response = None
        for i, path in enumerate(paths):
            response = self._session().get(self._url(path, params), timeout=self.timeout, **kwargs)
            if response.status_code == 404 and i + 1 < len(paths):
                response.close()
                continue
            if response.status_code in RETRY_STATUS:
                response.close()
                raise _Transient(f"HTTP {response.status_code} from ARM Live {service}")
            if response.status_code != 404:
                self._resolved[service] = path
            break
        return response

    def _retry(self, what: str, fn):
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except _TRANSIENT as exc:
                if attempt == self.max_retries:
                    raise ServerBusy(
                        f"{what}: giving up after {attempt} attempts ({self.redact(exc)})"
                    ) from None
                wait = min(60.0, self.backoff * 2 ** (attempt - 1))
                self.log(f"  {what}: {type(exc).__name__}, retrying in {wait:.0f} s")
                time.sleep(wait)
            except requests.RequestException as exc:  # messages can contain the URL, i.e. the token
                raise ArmLiveError(f"{what}: {self.redact(exc)}") from None

    # -- services ---------------------------------------------------------
    def query(self, datastream: str, start: dt.date, end: dt.date) -> List[str]:
        """Names of the datastream's files dated start..end (both inclusive)."""
        params = {
            "ds": datastream,
            "start": start.isoformat(),
            "end": (end + dt.timedelta(days=1)).isoformat(),  # make the last day inclusive
            "wt": "json",
        }

        def once():
            with self._request("query", params) as r:
                return r.status_code, r.text

        status, text = self._retry(f"query {datastream}", once)
        is_html = text.lstrip().startswith("<")
        if status != 200 or is_html:
            hint = ""
            if is_html or status in (401, 403):
                hint = (
                    " This usually means a wrong username or token; your token is shown "
                    "at https://adc.arm.gov/armlive/ ."
                )
            raise ArmLiveError(
                f"ARM Live query for {datastream} failed (HTTP {status}).{hint} "
                f"Reply begins: {self.redact(text[:150])!r}"
            )
        try:
            body = json.loads(text)
        except ValueError:
            raise ArmLiveError(f"Unexpected reply to query: {self.redact(text[:200])!r}") from None
        if body.get("status") != "success":
            raise ArmLiveError(
                f"ARM Live query for {datastream} returned status {body.get('status')!r}: "
                f"{self.redact(str(body)[:300])}"
            )
        files = sorted(set(body.get("files") or []))
        return [f for f in files if in_window(f, start, end)]

    def download(self, filename: str, dest: Path, variables: Optional[Sequence[str]] = None) -> dict:
        """Save one ARM file to `dest`. With `variables`, ARM's server sends only those
        variables (the mod service) instead of the complete file."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        if variables:
            service = "mod"
            params = {"variables": ",".join(variables), "wt": "cdf"}
            extra = {"data": json.dumps([filename]), "headers": {"Content-Type": "application/json"}}
        else:
            service, params, extra = "saveData", {"file": filename}, {}

        def once():
            with self._request(service, params, stream=True, **extra) as r:
                if r.status_code != 200:
                    text = r.content[:300].decode("utf-8", "replace")
                    if "not available" in text.lower():
                        raise FileNotAvailable(f"{filename}: not available through ARM Live ({text[:120]!r})")
                    raise ArmLiveError(
                        f"{filename}: HTTP {r.status_code} from ARM Live {service}: {self.redact(text)!r}"
                    )
                nbytes = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        nbytes += len(chunk)
                # Content-Length counts bytes on the wire, so compare only when
                # the body was not compressed in transit.
                expected = r.headers.get("Content-Length", "")
                encoded = r.headers.get("Content-Encoding", "identity").lower() not in ("", "identity")
                if expected.isdigit() and not encoded and int(expected) != nbytes:
                    raise _Transient(f"incomplete transfer ({nbytes} of {expected} bytes)")
                return nbytes, _disposition_filename(r.headers)

        try:
            nbytes, server_name = self._retry(f"download {filename}", once)
            with open(tmp, "rb") as f:
                head = f.read(512)
            if not head.startswith(NETCDF_MAGIC):
                text = head.decode("utf-8", "replace")
                if "not available" in text.lower():
                    raise FileNotAvailable(
                        f"{filename}: not available through ARM Live; order it through ARM "
                        f"Data Discovery instead. Server said: {text[:120]!r}"
                    )
                hint = (
                    " (an HTML page usually means a wrong username or token)"
                    if text.lstrip().startswith("<")
                    else ""
                )
                raise ArmLiveError(
                    f"{filename}: ARM Live did not return a netCDF file{hint}. "
                    f"Reply begins: {self.redact(text[:150])!r}"
                )
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()
        return {"bytes": nbytes, "server_filename": server_name}


def _disposition_filename(headers) -> Optional[str]:
    value = headers.get("Content-Disposition") or ""
    if "filename=" not in value:
        return None
    name = value.split("filename=", 1)[1].split(";")[0].strip().strip('"').strip("'")
    return name or None


def citation(datastream: str, start: dt.date, end: dt.date, timeout: float = 30) -> Optional[str]:
    """ARM's recommended citation (with DOI) for a datastream and date range, or None."""
    url = os.environ.get("ARM_CITATION_URL", DEFAULT_CITATION_URL)
    params = {
        "id": datastream,
        "citationType": "apa",
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return (r.json().get("citation") or "").strip() or None
    except (requests.RequestException, ValueError):
        pass
    return None
