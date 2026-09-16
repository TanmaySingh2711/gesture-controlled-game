"""The control seam: buffered requests, replacement, expiry - including property-based checks."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from game.controls import DIRECTIONS, OPPOSITE, REQUEST_GRACE, Controls


def test_newest_request_replaces_the_old_one() -> None:
    controls = Controls()
    controls.request_direction("up")
    controls.request_direction("left")
    assert controls.pending == "left"


def test_unknown_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown direction"):
        Controls().request_direction("diagonal")


def test_request_expires_exactly_at_the_grace_period() -> None:
    controls = Controls()
    controls.request_direction("up")
    controls.tick(REQUEST_GRACE - 1e-6)
    assert controls.pending == "up"
    controls.tick(2e-6)
    assert controls.pending is None


def test_consume_takes_the_request_once() -> None:
    controls = Controls()
    controls.request_direction("down")
    assert controls.consume() == "down"
    assert controls.consume() is None


def test_opposites_are_symmetric() -> None:
    for direction in DIRECTIONS:
        assert OPPOSITE[OPPOSITE[direction]] == direction
        assert OPPOSITE[direction] != direction


operations = st.lists(
    st.one_of(
        st.tuples(st.just("request"), st.sampled_from(DIRECTIONS)),
        st.tuples(st.just("tick"), st.floats(min_value=0.0, max_value=0.5)),
        st.tuples(st.just("consume"), st.none()),
        st.tuples(st.just("clear"), st.none()),
    ),
    max_size=60,
)


@given(operations)
def test_pending_is_always_the_latest_live_request(ops: list[tuple[str, object]]) -> None:
    """Whatever happens, the buffer holds at most the newest request, and never a stale one."""
    controls = Controls()
    latest: str | None = None
    age = 0.0
    for op, value in ops:
        if op == "request":
            assert isinstance(value, str)
            controls.request_direction(value)
            latest, age = value, 0.0
        elif op == "tick":
            assert isinstance(value, float)
            controls.tick(value)
            if latest is not None:
                age += value
                if age >= REQUEST_GRACE:
                    latest, age = None, 0.0
        else:
            getattr(controls, op)()
            latest, age = None, 0.0
        assert controls.pending == latest
        assert controls.age == pytest.approx(age)
        if controls.pending is not None:
            assert controls.age < REQUEST_GRACE
