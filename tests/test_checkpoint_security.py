"""The model loader's three independent guards: integrity pin, no pickled code, and meaning."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.train_model import (
    CHECKPOINT_PATH,
    FROZEN_CHECKPOINT_SHA256,
    file_sha256,
    load_direction_checkpoint,
)


def test_frozen_checkpoint_matches_its_pinned_digest() -> None:
    assert file_sha256(CHECKPOINT_PATH) == FROZEN_CHECKPOINT_SHA256


def test_frozen_checkpoint_loads_with_the_frozen_mapping() -> None:
    model, payload = load_direction_checkpoint()
    assert payload["class_to_index"] == {"left": 0, "right": 1, "up": 2, "down": 3}
    assert payload["task"] == "pacman_direction_v1"
    assert sum(parameter.numel() for parameter in model.parameters()) > 2_000_000


def test_a_single_flipped_byte_is_refused(tmp_path: Path) -> None:
    data = bytearray(Path(CHECKPOINT_PATH).read_bytes())
    data[len(data) // 2] ^= 0xFF
    tampered = tmp_path / "tampered.pt"
    tampered.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="not the frozen checkpoint"):
        load_direction_checkpoint(str(tampered))


def test_a_different_class_mapping_is_refused_even_with_the_right_shape(tmp_path: Path) -> None:
    """A shape check alone is not enough: a four-output checkpoint trained for different
    gestures would load cleanly and then silently mean the wrong thing at two indices."""
    mismatched = tmp_path / "mismatched.pt"
    torch.save({"class_to_index": {"left": 0, "right": 1, "jump": 2, "neutral": 3}}, mismatched)
    with pytest.raises(RuntimeError, match="was trained for"):
        load_direction_checkpoint(str(mismatched), expected_sha256=None)


class _Payload:
    """Pickles to a call of `print` - harmless, but proof of whether code would have run."""

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (print, ("PICKLED CODE EXECUTED",))


def test_loader_never_executes_pickled_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malicious = tmp_path / "malicious.pt"
    torch.save(
        {"class_to_index": {"left": 0, "right": 1, "up": 2, "down": 3}, "state_dict": _Payload()},
        malicious,
    )
    with pytest.raises(pickle.UnpicklingError):
        load_direction_checkpoint(str(malicious), expected_sha256=None)
    assert "PICKLED CODE EXECUTED" not in capsys.readouterr().out
