"""End-to-end tests against a mock ARM Live server. Run from the repository folder:

    python -m pytest tests -q
"""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from conftest import END, REPO, START, TOKEN, USER
from EPCAPE.download_data.armlive import ArmLiveClient, ArmLiveError
from EPCAPE.download_data.combine import combine_product
from EPCAPE.download_data.config import Machine, Product, active_machine
from EPCAPE.download_data.credentials import get_credentials
from EPCAPE.download_data.sync import sync_datastream, sync_product

VARS = ["first_cbh", "second_cbh", "third_cbh", "detection_status", "status_flag", "vertical_visibility"]
PRODUCT = Product("cbh_test", "epcceilM1.b1", VARS, "test cloud-base heights")
N_FILES = 10  # 10 days, one missing, one split into two files


def quiet(*_args, **_kwargs):
    pass


def client(token=TOKEN, **kw):
    return ArmLiveClient(USER, token, backoff=0.01, log=quiet, **kw)


def source_series(files, name):
    """Concatenate a variable from raw files, -9999 -> NaN, for comparison."""
    out = []
    for f in files:
        with netCDF4.Dataset(str(f)) as nc:
            nc.set_auto_maskandscale(False)
            if name in nc.variables:
                v = nc.variables[name][:].astype(float)
                v[v == -9999] = np.nan
            else:
                v = np.full(len(nc.dimensions["time"]), np.nan)
            out.append(v)
    return np.concatenate(out)


def in_window(archive):
    return sorted(p for p in archive.glob("*.nc") if START <= dt.datetime.strptime(p.name.split(".")[2], "%Y%m%d").date() <= END)


# -- ARM Live client ----------------------------------------------------------------
def test_query_is_inclusive_and_sends_user_first(env, server):
    names = client().query("epcceilM1.b1", START, END)
    days = sorted({n.split(".")[2] for n in names})
    assert days[0] == "20230215" and days[-1] == "20230224"  # last day included
    assert "20230217" not in days and "20230214" not in days and "20230225" not in days
    assert len(names) == N_FILES
    assert all(raw.startswith("user=") for _, raw in server.requests)


def test_wrong_token_is_explained_and_not_echoed(env, server):
    with pytest.raises(ArmLiveError) as err:
        client(token="wrongtoken").query("epcceilM1.b1", START, END)
    assert "username or token" in str(err.value)
    assert "wrongtoken" not in str(err.value)


def test_connection_failure_does_not_leak_token():
    c = ArmLiveClient(USER, "secret-token-123", base_url="http://127.0.0.1:9/armlive", max_retries=2, backoff=0.01, log=quiet)
    with pytest.raises(ArmLiveError) as err:
        c.query("epcceilM1.b1", START, END)
    assert "giving up after 2 attempts" in str(err.value)
    assert "secret-token-123" not in str(err.value)


def test_redirect_loop_is_reported_without_token(env, server):
    server.redirect_loop = True
    with pytest.raises(ArmLiveError) as err:
        client().query("epcceilM1.b1", START, END)
    assert "redirect" in str(err.value).lower() and TOKEN not in str(err.value)


def test_gzip_transfer_encoding_is_not_mistaken_for_truncation(env, server, archive):
    server.gzip = True
    machine = active_machine()
    result = sync_product(PRODUCT, START, END, machine=machine, client=client(), log=quiet)
    assert result.ok and len(result.downloaded) == N_FILES
    name = result.files[0]
    assert (machine.full_dir("epcceilM1.b1") / name).read_bytes() == (archive / name).read_bytes()


def test_falls_back_to_the_data_endpoint_path(env, archive):
    from arm_mock import MockArm

    mock = MockArm(archive, USER, TOKEN, style="act").start()
    try:
        c = client(base_url=mock.url + "/armlive")
        names = c.query("epcceilM1.b1", START, END)
        info = c.download(names[0], env / "one.nc")
        assert len(names) == N_FILES and info["bytes"] > 0
        assert c._resolved == {"query": "data/query", "saveData": "data/saveData"}
    finally:
        mock.stop()


# -- product (server-side subset) downloads -------------------------------------------
def test_product_download_subsets_records_and_resumes(env, server):
    machine = active_machine()
    result = sync_product(PRODUCT, START, END, machine=machine, client=client(), log=quiet)
    assert result.ok, result.failed
    assert len(result.downloaded) == N_FILES and not result.skipped

    subset_dir = machine.subset_dir(PRODUCT.name)
    for path in subset_dir.glob("*.nc"):
        with netCDF4.Dataset(str(path)) as nc:
            names = set(nc.variables)
        assert "backscatter" not in names and {"time", "first_cbh", "qc_first_cbh", "detection_status"} <= names
    # complete files: the one the variable list was read from, and the odd day (below)
    full = sorted(p.name for p in machine.full_dir("epcceilM1.b1").glob("*.nc"))
    assert full == [result.files[0], "epcceilM1.b1.20230221.000002.nc"]
    subset_sizes = [(subset_dir / n).stat().st_size for n in result.files]
    assert result.full_file_bytes > max(subset_sizes)  # tiny test files; real ones differ far more

    manifest = json.loads((subset_dir / "manifest.json").read_text())
    assert manifest["variables"][:2] == ["time_offset", "time"]  # base_time breaks ARM's mod service
    assert "qc_first_cbh" in manifest["variables"] and "range" not in manifest["variables"]
    assert manifest["citation"].startswith("Zhang") and len(manifest["files"]) == N_FILES

    # the day lacking qc_third_cbh was fetched whole and subset locally
    (odd,) = [n for n in result.fallbacks]
    assert "20230221" in odd and "qc_third_cbh" in result.fallbacks[odd]

    before = len(server.requests)
    again = sync_product(PRODUCT, START, END, machine=machine, client=client(), log=quiet)
    assert len(again.skipped) == N_FILES and not again.downloaded
    assert [s for s, _ in server.requests[before:]] == ["query"]


def test_misspelled_variable_fails_before_any_subset_request(env, server):
    bad = Product("bad", "epcceilM1.b1", ["first_cbh", "frist_cbh"])
    with pytest.raises(ValueError) as err:
        sync_product(bad, START, END, machine=active_machine(), client=client(), log=quiet)
    assert "frist_cbh" in str(err.value) and "second_cbh" in str(err.value)  # lists what exists
    assert not any(s == "mod" for s, _ in server.requests)


def test_broken_subset_service_stops_instead_of_downloading_everything(env, server):
    server.fail_next["mod"] = 10_000
    machine = active_machine()
    with pytest.raises(ArmLiveError) as err:
        sync_product(PRODUCT, START, END, machine=machine, client=client(max_retries=2), log=quiet)
    assert "--full" in str(err.value)
    assert len(list(machine.full_dir("epcceilM1.b1").glob("*.nc"))) == 1  # only the variable-list file


def test_retries_busy_server_and_truncated_transfers(env, server):
    server.fail_next.update(query=2, saveData=2, mod=1)
    server.truncate_next.update(saveData=1, mod=2)
    result = sync_product(PRODUCT, START, END, machine=active_machine(), client=client(), log=quiet)
    assert result.ok and len(result.downloaded) == N_FILES
    for path in active_machine().subset_dir(PRODUCT.name).glob("*.nc"):
        xr.open_dataset(path).close()  # every file is complete and readable


def test_unavailable_file_is_reported_not_saved(env, server):
    target = in_window(Path(server.files[next(iter(server.files))]).parent)[3].name
    server.unavailable.add(target)
    machine = active_machine()
    result = sync_product(PRODUCT, START, END, machine=machine, client=client(), log=quiet)
    assert result.unavailable == [target] and result.ok
    assert not (machine.subset_dir(PRODUCT.name) / target).exists()
    assert not list(machine.subset_dir(PRODUCT.name).glob("*.part"))


def test_changed_variable_list_requires_overwrite(env, server):
    machine = active_machine()
    sync_product(PRODUCT, START, END, machine=machine, client=client(), log=quiet)
    narrower = Product(PRODUCT.name, PRODUCT.datastream, ["first_cbh"])
    with pytest.raises(RuntimeError) as err:
        sync_product(narrower, START, END, machine=machine, client=client(), log=quiet)
    assert "--overwrite" in str(err.value)
    result = sync_product(narrower, START, END, machine=machine, client=client(), overwrite=True, log=quiet)
    assert len(result.downloaded) == N_FILES


def test_full_files_are_byte_identical(env, server, archive):
    machine = active_machine()
    result = sync_datastream("epcceilM1.b1", START, END, machine=machine, client=client(), log=quiet)
    assert result.ok and len(result.downloaded) == N_FILES
    for name in result.files:
        assert (machine.full_dir("epcceilM1.b1") / name).read_bytes() == (archive / name).read_bytes()


# -- combining --------------------------------------------------------------------
def test_combined_file_matches_sources_and_reads_without_xarray(env, server, archive):
    machine = active_machine()
    sync_product(PRODUCT, START, END, machine=machine, client=client(), log=quiet)
    lines = []
    out = combine_product(PRODUCT, START, END, machine=machine, log=lines.append)
    report = "\n".join(lines)
    sources = in_window(archive)

    with xr.open_dataset(out) as ds:
        t = ds["time"].values
        assert np.all(np.diff(t) > np.timedelta64(0, "s"))
        assert t.size == sum(len(netCDF4.Dataset(str(f)).dimensions["time"]) for f in sources)
        assert str(t[0])[:19] == "2023-02-15T00:00:02"
        for name in ("first_cbh", "second_cbh", "vertical_visibility", "detection_status", "qc_third_cbh"):
            np.testing.assert_allclose(ds[name].values.astype(float), source_series(sources, name), equal_nan=True)
        assert {"base_time", "time_offset", "backscatter"}.isdisjoint(ds.variables)
        assert float(ds["lat"]) == pytest.approx(32.867, abs=1e-4)
        assert ds.attrs["source_file_count"] == N_FILES
        assert ds.attrs["citation"].startswith("Zhang")
        assert ds["first_cbh"].attrs["units"] == "m"

    # what MATLAB's ncread sees: raw netCDF, no xarray decoding
    with netCDF4.Dataset(str(out)) as nc:
        nc.set_auto_maskandscale(False)
        time = nc.variables["time"]
        assert time.units == "seconds since 1970-01-01" and time.dtype == np.float64
        first_posix = dt.datetime(2023, 2, 15, 0, 0, 2, tzinfo=dt.timezone.utc).timestamp()
        assert time[0] == pytest.approx(first_posix)
        assert nc.variables["first_cbh"].dtype == np.float32 and np.isnan(nc.variables["first_cbh"]._FillValue)
        status = nc.variables["detection_status"]
        assert status.dtype == np.int16 and status._FillValue == -9999
        qc3 = nc.variables["qc_third_cbh"]
        assert qc3.dtype == np.int32 and qc3._FillValue == -9999 and (qc3[:] == -9999).any()

    assert "days with no data: 1 (2023-02-17)" in report
    assert "qc_third_cbh" in report and "missing from 1 of 10 files" in report
    assert "detection_status:" in report and "One cloud base detected" in report


def test_mounted_archive_needs_no_credentials_or_download(env, archive, monkeypatch):
    monkeypatch.delenv("ARM_USERNAME")
    monkeypatch.delenv("ARM_TOKEN")
    mount = env / "arm_archive"
    shutil.copytree(archive, mount / "epc" / "epcceilM1.b1")
    machine = Machine("arm_test", data_root=env / "data", arm_archive=mount)
    result = sync_product(PRODUCT, START, END, machine=machine, log=quiet)
    assert result.source == "archive" and len(result.files) == N_FILES
    out = combine_product(PRODUCT, START, END, machine=machine, log=quiet)
    with xr.open_dataset(out) as ds:
        assert "first_cbh" in ds and ds.attrs["source_file_count"] == N_FILES


def test_variable_absent_from_first_file_is_kept(env, archive, monkeypatch):
    from arm_mock import write_ceil_file

    mount = env / "arm_archive"
    folder = mount / "epc" / "epcceilM1.b1"
    shutil.copytree(archive, folder)
    first = in_window(folder)[0]
    write_ceil_file(first, dt.datetime(2023, 2, 15, 0, 0, 2), dt.datetime(2023, 2, 16), drop=["qc_second_cbh"])
    machine = Machine("arm_test", data_root=env / "data", arm_archive=mount)
    out = combine_product(PRODUCT, START, END, machine=machine, log=quiet)
    with xr.open_dataset(out) as ds:
        np.testing.assert_allclose(ds["qc_second_cbh"].values.astype(float),
                                   source_series(in_window(folder), "qc_second_cbh"), equal_nan=True)
        assert np.isnan(ds["qc_second_cbh"].values[:96]).all() and not np.isnan(ds["qc_second_cbh"].values[96:]).any()


# -- configuration and credentials ---------------------------------------------------
def test_machine_selection(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "campaign: {start: 2023-02-15, end: 2024-02-14}\n"
        "machines:\n"
        "  local: {data_root: data}\n"
        "  cumulus: {data_root: /gpfs/wolf/proj-shared/PROJECT_ID/epcape, arm_archive: /data/archive}\n"
        "  ucsd: {data_root: ~/epcape_data}\n"
    )
    monkeypatch.setenv("EPCAPE_CONFIG", str(cfg))
    monkeypatch.delenv("EPCAPE_DATA_ROOT", raising=False)
    monkeypatch.delenv("EPCAPE_MACHINE", raising=False)
    assert active_machine().data_root == REPO / "data"
    monkeypatch.setenv("EPCAPE_MACHINE", "ucsd")
    assert active_machine().data_root == Path("~/epcape_data").expanduser()
    monkeypatch.setenv("EPCAPE_MACHINE", "cumulus")
    with pytest.raises(ValueError, match="PROJECT_ID"):
        active_machine()
    monkeypatch.setenv("EPCAPE_DATA_ROOT", str(tmp_path / "elsewhere"))
    assert active_machine().data_root == tmp_path / "elsewhere"
    monkeypatch.setenv("EPCAPE_MACHINE", "laptop2")
    with pytest.raises(KeyError, match="laptop2"):
        active_machine()


def test_credentials_file(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ARM_USERNAME", raising=False)
    monkeypatch.delenv("ARM_TOKEN", raising=False)
    path = tmp_path / "creds"
    path.write_text("[arm]\nusername = andrew\ntoken = ab%cd12\n")
    path.chmod(0o644)
    monkeypatch.setenv("ARM_CREDENTIALS_FILE", str(path))
    assert get_credentials(interactive=False) == ("andrew", "ab%cd12")
    assert "chmod 600" in capsys.readouterr().err
    path.unlink()
    with pytest.raises(RuntimeError, match="adc.arm.gov/armlive"):
        get_credentials(interactive=False)


# -- command line ------------------------------------------------------------------
def test_command_line_end_to_end(env, server):
    run_env = {k: v for k, v in os.environ.items()}

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=REPO, env=run_env, capture_output=True, text=True, timeout=300)

    listing = run("download_data/download_arm.py", "--list")
    assert listing.returncode == 0 and "cbh_ceil_M1" in listing.stdout and "cbh_ceil_S2" in listing.stdout

    dry = run("download_data/download_arm.py", "cbh_ceil_M1", "--start", str(START), "--end", str(END), "--dry-run")
    assert dry.returncode == 0 and f"lists {N_FILES}" in dry.stdout

    down = run("download_data/download_arm.py", "cbh_ceil_M1", "--start", str(START), "--end", str(END), "--workers", "3")
    assert down.returncode == 0, down.stdout + down.stderr
    assert f"Done: {N_FILES} downloaded" in down.stdout and "download_data/combine_product.py cbh_ceil_M1" in down.stdout

    comb = run("download_data/combine_product.py", "cbh_ceil_M1", "--start", str(START), "--end", str(END))
    assert comb.returncode == 0, comb.stdout + comb.stderr
    assert "Wrote" in comb.stdout and (env / "data" / "processed" / "cbh_ceil_M1_20230215_20230224.nc").is_file()

    missing = run("download_data/combine_product.py", "cbh_ceil_S2", "--start", str(START), "--end", str(END))
    assert missing.returncode == 2 and "python download_data/download_arm.py cbh_ceil_S2" in missing.stderr
