"""The HaGRID download, crop, promote and lineage-audit pipeline, end to end, with no network.

A miniature HaGRID archive - the same zip layout, stored images and annotation format - is built
in memory and served through a fake `urlopen` that honours HTTP HEAD and Range requests, exactly
like the Hugging Face server does. The real code then:

1. reads the archive over range requests and builds a small dataset (`crop_hagrid_hands.main`),
2. replays that build byte for byte to recover every crop's source photo and person, audits
   subject leakage, and draws an unseen-people set (`audit_hagrid_lineage.main`).

Every path both scripts write to is redirected into a temporary folder, and the test checks the
real project's dataset folders are untouched afterwards.
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src import audit_hagrid_lineage as audit
from src import crop_hagrid_hands as chh

PER_GESTURE = 8  # photos per gesture in the miniature archive
PER_CLASS = 2  # crops per class the build takes
PROJECT = Path(__file__).resolve().parent.parent


def photo(seed: int) -> bytes:
    """A 384x288 JPEG whose content differs per seed, so every crop has its own hash."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (288, 384, 3), np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return bytes(encoded.tobytes())


def miniature_archive() -> tuple[bytes, dict[str, dict[str, Any]]]:
    """(zip bytes, annotations) laid out like hagrid-sample-500k-384p."""
    annotations: dict[str, dict[str, Any]] = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        seed = 0
        for gesture in chh.GESTURE_FOR.values():
            records: dict[str, Any] = {}
            for index in range(PER_GESTURE):
                seed += 1
                uuid = f"{gesture}-{index:04d}"
                box = [0.30, 0.25, 0.30, 0.40]
                if index == 0:
                    box = [0.5, 0.5, 0.02, 0.02]  # too small to use: skipped when choosing
                records[uuid] = {
                    "bboxes": [box, [0.05, 0.05, 0.1, 0.1]],
                    "labels": [gesture, "no_gesture"],
                    "user_id": f"person-{seed:03d}",
                    "leading_hand": "right",
                }
                if index == 1:
                    continue  # annotated but missing from the archive: skipped as "missing"
                archive.writestr(
                    f"{chh.ROOT}/hagrid_500k/train_val_{gesture}/{uuid}.jpg", photo(seed)
                )
            annotations[gesture] = records
            archive.writestr(
                f"{chh.ROOT}/ann_train_val/{gesture}.json", json.dumps(records).encode()
            )
    return buffer.getvalue(), annotations


class FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self.body = body
        self.headers = headers

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


@pytest.fixture
def hagrid_server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Serves the miniature archive at ARCHIVE_URL through urllib, counting requests."""
    data, annotations = miniature_archive()
    served: dict[str, Any] = {"requests": 0, "annotations": annotations}

    def urlopen(request: urllib.request.Request, timeout: float = 0) -> FakeResponse:
        assert request.full_url == chh.ARCHIVE_URL
        served["requests"] += 1
        if request.get_method() == "HEAD":
            return FakeResponse(b"", {"Content-Length": str(len(data))})
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range") or "")
        assert match, "every GET must be a range request - the archive is never downloaded"
        start, end = int(match.group(1)), int(match.group(2))
        return FakeResponse(data[start : end + 1], {})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return served


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every folder both scripts read or write into `tmp_path`."""
    for name in ("left", "right", "up", "down"):
        (tmp_path / "dataset" / name).mkdir(parents=True)
    monkeypatch.setattr(chh, "DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setattr(chh, "CROPPED_DIR", str(tmp_path / "dataset_cropped"))
    monkeypatch.setattr(chh, "CACHE_DIR", str(tmp_path / ".hagrid_cache"))
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit, "DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(audit, "SPLITS_FILE", tmp_path / "data_splits.json")
    monkeypatch.setattr(audit, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(audit, "EXTERNAL_DIR", tmp_path / "dataset_external")
    monkeypatch.setattr(audit, "DIGEST_CACHE", tmp_path / ".hagrid_cache" / "digests.json")
    return tmp_path


def project_snapshot() -> tuple[bool, int]:
    """Whether dataset_cropped/ exists in the real project, and how many real dataset images."""
    return (PROJECT / "dataset_cropped").exists(), len(list((PROJECT / "dataset").rglob("*.jpg")))


def test_build_then_audit_recovers_every_crop_and_its_person(
    hagrid_server: dict[str, Any],
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = project_snapshot()

    # --- build ---------------------------------------------------------------------------
    status = chh.main(["--per-class", str(PER_CLASS), "--promote", "--keep-cache"])
    output = capsys.readouterr().out
    assert status == 0, output
    assert "RESULT: PASS" in output
    built = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("dataset/*/*.jpg"))
    assert len(built) == 4 * PER_CLASS
    assert not (workspace / "dataset_cropped").exists(), "promotion removes the staging folder"
    assert (workspace / ".hagrid_cache" / "ann_fist.json").exists(), "--keep-cache kept it"

    # --- audit ---------------------------------------------------------------------------
    by_class = {
        name: [p for p in built if f"/{name}/" in p] for name in ("left", "right", "up", "down")
    }
    split = {
        "splits": {
            "train": [{"path": p} for name in by_class for p in by_class[name][:1]],
            "val": [],
            "test": [{"path": p} for name in by_class for p in by_class[name][1:]],
        }
    }
    (workspace / "data_splits.json").write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_hagrid_lineage.py", "--external", "1"])

    assert audit.main() == 0

    lineage = json.loads((workspace / "reports" / "dataset_lineage.json").read_text("utf-8"))
    assert lineage["matched"] == lineage["total"] == 4 * PER_CLASS
    assert all(not files for files in lineage["unmatched"].values())
    annotations = hagrid_server["annotations"]
    for path, origin in lineage["files"].items():
        # Every crop was traced to the exact photo and person it came from.
        record = annotations[origin["gesture"]][origin["uuid"]]
        assert record["user_id"] == origin["user_id"], path

    leakage = json.loads((workspace / "reports" / "subject_leakage.json").read_text("utf-8"))
    assert leakage["shared_between_splits"]["train/test"]["shared_people"] == 0
    assert leakage["lineage_matched"] == f"{4 * PER_CLASS}/{4 * PER_CLASS}"

    manifest = json.loads((workspace / "dataset_external" / "manifest.json").read_text("utf-8"))
    assert manifest["lineage_complete"] is True
    assert len(manifest["files"]) == 4
    dataset_people = {origin["user_id"] for origin in lineage["files"].values()}
    assert not {info["user_id"] for info in manifest["files"].values()} & dataset_people

    # A second audit reuses the cached crop hashes rather than recomputing them.
    assert json.loads((workspace / ".hagrid_cache" / "digests.json").read_text("utf-8"))
    assert project_snapshot() == before, "the real project folders were touched"


def test_too_few_candidates_stops_the_build(
    hagrid_server: dict[str, Any], workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert chh.main(["--per-class", "50", "--classes", "up"]) == 1
    assert "ERROR: only 7 candidates for up, need 50" in capsys.readouterr().out
    assert not list(workspace.rglob("dataset/up/*.jpg")), "nothing was written"


def test_an_incomplete_build_is_not_promoted_and_drops_its_cache(
    hagrid_server: dict[str, Any], workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 7 usable candidates per class, but one is missing from the archive: only 6 can be cropped.
    status = chh.main(["--per-class", "7", "--classes", "left", "--promote"])
    output = capsys.readouterr().out
    assert status == 1
    assert "RESULT: INCOMPLETE" in output
    assert "missing=1" in output  # the member absent from the archive was skipped, not fatal
    assert not list((workspace / "dataset" / "left").glob("*.jpg")), "nothing was promoted"
    assert not (workspace / ".hagrid_cache").exists(), "the annotation cache was dropped"


def test_audit_refuses_annotations_without_people(
    hagrid_server: dict[str, Any],
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache = workspace / ".hagrid_cache"
    cache.mkdir()
    for gesture, records in hagrid_server["annotations"].items():
        stripped = {
            uuid: {k: v for k, v in r.items() if k != "user_id"} for uuid, r in records.items()
        }
        (cache / f"ann_{gesture}.json").write_text(json.dumps(stripped), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_hagrid_lineage.py"])
    assert audit.main() == 1
    assert "no user_id" in capsys.readouterr().out


def test_the_range_reader_reads_seeks_and_retries(
    hagrid_server: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, _archive = chh.open_archive()
    handle.seek(10)
    first = handle.read(5)
    handle.seek(-5, 1)
    assert handle.read(5) == first
    handle.seek(-3, 2)
    assert len(handle.read()) == 3
    assert handle.read(4) == b""  # at the end
    assert handle.tell() == handle.size

    attempts = {"count": 0}
    real = urllib.request.urlopen

    def flaky(request: urllib.request.Request, timeout: float = 0) -> Any:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("slow server")
        return real(request, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(chh.time, "sleep", lambda _seconds: None)
    handle.seek(0)
    assert handle.read(4) == b"PK\x03\x04"  # the zip signature, after two failed attempts
    assert attempts["count"] == 3

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(ConnectionError())
    )
    with pytest.raises(RuntimeError, match="range request failed after retries"):
        handle.read(4)
