"""Offline stand-ins for ARM: synthetic ceilometer files and a mock ARM Live server.

The synthetic files follow the ARM ceil.b1 data object design (variable names,
types, missing_value = -9999, base_time/time_offset/time, qc_ companions,
backscatter profiles) closely enough to exercise the download and combine code.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlsplit

import netCDF4
import numpy as np

EPOCH = dt.datetime(1970, 1, 1)
DETECTION_DESCRIPTIONS = {
    0: "No significant backscatter",
    1: "One cloud base detected",
    2: "Two cloud bases detected",
    3: "Three cloud bases detected",
    4: "Full obscuration determined but no cloud base detected",
    5: "Some obscuration detected but determined to be transparent",
}


def write_ceil_file(path: Path, start: dt.datetime, stop: dt.datetime, n: int = 96, nrange: int = 40,
                    facility: str = "M1", drop: Iterable[str] = ()) -> None:
    drop = set(drop)
    rng = np.random.default_rng(int(start.strftime("%Y%m%d%H%M")))
    midnight = dt.datetime(start.year, start.month, start.day)
    offsets = np.arange(n) * ((stop - start).total_seconds() / n)
    base_time = int((start - EPOCH).total_seconds())

    status = rng.choice([0, 1, 1, 1, 2, 3, 4], size=n).astype("i2")
    first = np.where((status >= 1) & (status <= 3), 350 + 500 * rng.random(n), -9999).astype("f4")
    second = np.where((status == 2) | (status == 3), first + 600, -9999).astype("f4")
    third = np.where(status == 3, first + 1500, -9999).astype("f4")
    visibility = np.where(status == 4, 100 + 100 * rng.random(n), -9999).astype("f4")

    with netCDF4.Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET") as nc:
        nc.setncatts({
            "command_line": f"ceil_ingest -s epc -f {facility}",
            "Conventions": "ARM-1.3",
            "process_version": "ingest-ceil-1.8-0.el7",
            "dod_version": "ceil-b1-3.0",
            "input_source": f"/data/collection/epc/epcceil{facility}.00/{start:%Y%m%d.%H%M%S}.dat",
            "site_id": "epc",
            "platform_id": "ceil",
            "facility_id": facility,
            "data_level": "b1",
            "location_description": "Eastern Pacific Cloud Aerosol Precipitation Experiment (EPCAPE), La Jolla, CA",
            "datastream": f"epcceil{facility}.b1",
            "doi": "10.5439/1181954",
            "history": "created by user dsmgr on machine prod-proc2 (synthetic test file)",
        })
        nc.createDimension("time", None)
        nc.createDimension("range", nrange)
        nc.createDimension("bound", 2)

        def var(name, dtype, dims, data, **attrs):
            if name in drop:
                return
            v = nc.createVariable(name, dtype, dims)
            v.setncatts(attrs)
            v[...] = data

        var("base_time", "i4", (), base_time, string=f"{start:%d-%b-%Y,%H:%M:%S} GMT",
            long_name="Base time in Epoch", units="seconds since 1970-1-1 0:00:00 0:00", ancillary_variables="time_offset")
        var("time_offset", "f8", ("time",), offsets, long_name="Time offset from base_time",
            units=f"seconds since {start:%Y-%m-%d %H:%M:%S} 0:00", ancillary_variables="base_time")
        var("time", "f8", ("time",), (start - midnight).total_seconds() + offsets,
            long_name="Time offset from midnight", units=f"seconds since {start:%Y-%m-%d} 00:00:00 0:00",
            bounds="time_bounds", standard_name="time")
        t = (start - midnight).total_seconds() + offsets
        var("time_bounds", "f8", ("time", "bound"), np.stack([t, t + 16.0], axis=1),
            long_name="Time cell bounds", bound_offsets=np.array([0.0, 16.0]))
        var("range", "f4", ("range",), 10.0 * (np.arange(nrange) + 0.5), long_name="Distance to the center of the corresponding range bin", units="m")
        var("detection_status", "i2", ("time",), status, long_name="Detection status", units="unitless",
            missing_value=np.int16(-9999), flag_values=np.arange(6, dtype="i2"),
            **{f"flag_{k}_description": v for k, v in DETECTION_DESCRIPTIONS.items()})
        var("status_flag", "i2", ("time",), np.zeros(n, "i2"), long_name="Ceilometer status indicator",
            units="unitless", flag_values=np.arange(3, dtype="i2"))
        for name, data, long_name in (
            ("first_cbh", first, "Lowest cloud base height detected"),
            ("second_cbh", second, "Second lowest cloud base height"),
            ("third_cbh", third, "Third cloud base height"),
            ("vertical_visibility", visibility, "Vertical visibility"),
        ):
            var(name, "f4", ("time",), data, long_name=long_name, units="m", valid_min=np.float32(0),
                valid_max=np.float32(7620), missing_value=np.float32(-9999))
            var(f"qc_{name}", "i4", ("time",), (data == -9999).astype("i4"),
                long_name=f"Quality check results on field: {long_name}", units="1", flag_method="bit",
                bit_1_description="Value is equal to missing_value.", bit_1_assessment="Bad")
        var("backscatter", "f4", ("time", "range"), rng.random((n, nrange)).astype("f4"),
            long_name="Backscatter", units="1/(sr*km*10000)", missing_value=np.float32(-9999))
        var("lat", "f4", (), 32.867, long_name="North latitude", units="degree_N")
        var("lon", "f4", (), -117.257, long_name="East longitude", units="degree_E")
        var("alt", "f4", (), 8.0, long_name="Altitude above mean sea level", units="m")


def make_archive(root: Path, first_day: dt.date, last_day: dt.date, facility: str = "M1") -> Path:
    """Daily files for [first_day, last_day] plus one day either side, with realistic quirks:
    day 3 missing, day 5 split in two files (instrument restart), day 7 lacking qc_third_cbh."""
    root.mkdir(parents=True, exist_ok=True)
    day = first_day - dt.timedelta(days=1)
    index = -1
    while day <= last_day + dt.timedelta(days=1):
        midnight = dt.datetime(day.year, day.month, day.day)
        next_midnight = midnight + dt.timedelta(days=1)
        stem = f"epcceil{facility}.b1"
        if index == 2:
            pass  # data gap
        elif index == 4:
            noon = midnight + dt.timedelta(hours=12)
            write_ceil_file(root / f"{stem}.{midnight:%Y%m%d}.000002.nc", midnight + dt.timedelta(seconds=2), noon, n=48, facility=facility)
            write_ceil_file(root / f"{stem}.{midnight:%Y%m%d}.120000.nc", noon, next_midnight, n=48, facility=facility)
        else:
            drop = ["qc_third_cbh"] if index == 6 else []
            write_ceil_file(root / f"{stem}.{midnight:%Y%m%d}.000002.nc", midnight + dt.timedelta(seconds=2),
                            next_midnight, facility=facility, drop=drop)
        day += dt.timedelta(days=1)
        index += 1
    return root


def subset_bytes(src: Path, variables) -> bytes:
    """What ARM's mod service does: a copy of `src` holding only `variables`.
    The mock runs in the same process as the download threads, and netCDF-C is
    not thread-safe, so it must share the package's lock."""
    from epcape.arm_files import NETCDF_LOCK

    with NETCDF_LOCK:
        return _subset_bytes(src, variables)


def _subset_bytes(src: Path, variables) -> bytes:
    with netCDF4.Dataset(str(src)) as s:
        s.set_auto_maskandscale(False)
        unknown = [v for v in variables if v not in s.variables]
        if unknown:
            raise KeyError(", ".join(unknown))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "subset.nc"
            with netCDF4.Dataset(str(out), "w", format=s.data_model) as d:
                d.set_auto_maskandscale(False)
                d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
                for dim in dict.fromkeys(dim for v in variables for dim in s.variables[v].dimensions):
                    sd = s.dimensions[dim]
                    d.createDimension(dim, None if sd.isunlimited() else len(sd))
                for v in variables:
                    sv = s.variables[v]
                    dv = d.createVariable(v, sv.datatype, sv.dimensions)
                    dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs()})
                    dv[...] = sv[...]
            return out.read_bytes()


def _file_time(name: str) -> dt.datetime:
    parts = name.split(".")
    return dt.datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M%S")


class MockArm:
    """Mock ARM Live: /armlive/{query,saveData,mod} and a citation endpoint.

    style="doc" serves /armlive/<service> (ARM's documentation); style="act"
    serves /armlive/data/<service> (ARM's ACT toolkit) and 404s the other.
    """

    def __init__(self, root: Path, user: str, token: str, style: str = "doc"):
        self.files = {p.name: p for p in Path(root).glob("*.nc")}
        self.user, self.token, self.style = user, token, style
        self.fail_next = {}          # service -> number of HTTP 503 replies still to send
        self.truncate_next = {}      # service -> number of truncated transfers still to send
        self.unavailable = set()     # file names ARM Live "cannot serve"
        self.gzip = False            # compress file bodies in transit (Content-Encoding: gzip)
        self.redirect_loop = False   # answer every request with a redirect to itself
        self.requests = []           # (service, raw query string)
        self.citation_text = ("Zhang, D., Ermold, B., & Morris, V. Ceilometer (CEIL). ARM Mobile Facility (EPC). "
                              "https://doi.org/10.5439/1181954")
        self._lock = threading.Lock()

    # -- server lifecycle -------------------------------------------------------
    def start(self) -> "MockArm":
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _take(self, counter: dict, service: str) -> bool:
        with self._lock:
            if counter.get(service, 0) > 0:
                counter[service] -= 1
                return True
        return False

    def _handler(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, code: int, body: bytes, ctype: str = "text/plain", headers: Optional[dict] = None,
                      truncate: bool = False) -> None:
                headers = dict(headers or {})
                if mock.gzip and ctype == "application/octet-stream":
                    body = gzip.compress(body)
                    headers["Content-Encoding"] = "gzip"
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body[: len(body) // 2] if truncate else body)
                if truncate:
                    self.close_connection = True

            def do_GET(self):
                parts = urlsplit(self.path)
                raw, path = parts.query, parts.path
                params = dict(parse_qsl(raw, keep_blank_values=True))
                if path == "/citation":
                    return self._send(200, json.dumps({"citation": mock.citation_text}).encode(), "application/json")
                if path == "/armlive/mod":
                    service = "mod"
                elif mock.style == "doc" and path in ("/armlive/query", "/armlive/saveData"):
                    service = path.rsplit("/", 1)[1]
                elif mock.style == "act" and path in ("/armlive/data/query", "/armlive/data/saveData"):
                    service = path.rsplit("/", 1)[1]
                else:
                    return self._send(404, b"<html>Not found</html>", "text/html")
                with mock._lock:
                    mock.requests.append((service, raw))
                if mock.redirect_loop:
                    self.send_response(302)
                    self.send_header("Location", self.path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if not raw.startswith("user="):
                    return self._send(400, b"user must be the first parameter")
                if params.get("user") != f"{mock.user}:{mock.token}":
                    return self._send(200, b"<!DOCTYPE html><html><body>Invalid user</body></html>", "text/html")
                if mock._take(mock.fail_next, service):
                    return self._send(503, b"Service temporarily unavailable")
                truncate = mock._take(mock.truncate_next, service)

                if service == "query":
                    ds = params.get("ds", "")
                    start = dt.datetime.fromisoformat(params["start"])
                    end = dt.datetime.fromisoformat(params["end"])  # exclusive
                    names = sorted(n for n in mock.files if n.startswith(ds + ".") and start <= _file_time(n) < end)
                    return self._send(200, json.dumps({"status": "success", "files": names}).encode(), "application/json")

                if service == "saveData":
                    name = params.get("file", "")
                    if name in mock.unavailable or name not in mock.files:
                        return self._send(200, b"This data file is not available on /data/archive.")
                    return self._send(200, mock.files[name].read_bytes(), "application/octet-stream", truncate=truncate)

                # mod: JSON list of file names in the body of a GET request
                length = int(self.headers.get("Content-Length", 0) or 0)
                names = json.loads(self.rfile.read(length) or b"[]")
                variables = [v for v in params.get("variables", "").split(",") if v]
                if len(names) != 1:
                    return self._send(400, b"mock supports one file per request")
                name = names[0]
                if name in mock.unavailable or name not in mock.files:
                    return self._send(200, b"This data file is not available on /data/archive.")
                try:
                    body = subset_bytes(mock.files[name], variables)
                except KeyError as exc:
                    return self._send(500, f"Variable not found: {exc}".encode())
                headers = {"Content-Disposition": f'attachment; filename="{name[:-3]}.custom.nc"'}
                return self._send(200, body, "application/octet-stream", headers, truncate=truncate)

        return Handler
