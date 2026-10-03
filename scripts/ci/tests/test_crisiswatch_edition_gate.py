# Pythia / Copyright (c) 2025 Kevin Wyjad
"""The hs_submit CrisisWatch edition gate retries, then proceeds; never blocks."""

from datetime import date

from scripts.ci import crisiswatch_edition_gate as gate


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_expected_edition_follows_the_publication_day():
    # From the 10th the previous month's edition is expected...
    assert gate.expected_edition(date(2026, 10, 13)) == (2026, 9)
    assert gate.expected_edition(date(2027, 1, 13)) == (2026, 12)
    # ...but a run on the 1st cannot hold an edition ICG has not published.
    assert gate.expected_edition(date(2026, 10, 1)) == (2026, 8)
    assert gate.expected_edition(date(2027, 1, 1)) == (2026, 11)
    assert gate.expected_edition(date(2027, 2, 9)) == (2026, 12)


def test_lands_on_a_retry():
    held = [[(2026, 8)], [(2026, 8)], [(2026, 8), (2026, 9)]]
    calls = []
    clock = _Clock()
    v = gate.run_gate(
        today=date(2026, 10, 13), held_fn=lambda: held[len(calls) - 1],
        refresh_fn=lambda: calls.append(1), deadline_sec=1200, retry_sec=300,
        clock=clock, sleep=clock.sleep,
    )
    assert v["ok"] is True and v["attempts"] == 3


def test_gives_up_within_the_deadline_and_names_the_edition_held():
    clock = _Clock()
    calls = []
    v = gate.run_gate(
        today=date(2026, 10, 13), held_fn=lambda: [(2026, 7), (2026, 8)],
        refresh_fn=lambda: calls.append(1), deadline_sec=1200, retry_sec=300,
        clock=clock, sleep=clock.sleep,
    )
    assert v["ok"] is False
    assert v["expected_edition"] == "2026-09"
    assert v["newest_edition_held"] == "2026-08"
    assert v["attempts"] == 5  # at 0, 300, 600, 900 and 1200; a sixth would pass it
    assert clock.t <= 1200


def test_a_refresh_that_raises_is_retried_not_fatal():
    def boom():
        raise RuntimeError("archive.org refused")

    clock = _Clock()
    v = gate.run_gate(
        today=date(2026, 10, 13), held_fn=lambda: [], refresh_fn=boom,
        deadline_sec=600, retry_sec=300, clock=clock, sleep=clock.sleep,
    )
    assert v["ok"] is False and v["newest_edition_held"] is None
