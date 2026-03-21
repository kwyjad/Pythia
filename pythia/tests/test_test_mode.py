# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Unit tests for pythia.test_mode."""

from pythia.test_mode import is_test_mode


def test_is_test_mode_returns_true_for_1(monkeypatch):
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    assert is_test_mode() is True


def test_is_test_mode_returns_true_for_true(monkeypatch):
    monkeypatch.setenv("PYTHIA_TEST_MODE", "true")
    assert is_test_mode() is True


def test_is_test_mode_returns_true_for_yes(monkeypatch):
    monkeypatch.setenv("PYTHIA_TEST_MODE", "yes")
    assert is_test_mode() is True


def test_is_test_mode_returns_true_with_whitespace(monkeypatch):
    monkeypatch.setenv("PYTHIA_TEST_MODE", "  1  ")
    assert is_test_mode() is True


def test_is_test_mode_returns_false_for_0(monkeypatch):
    monkeypatch.setenv("PYTHIA_TEST_MODE", "0")
    assert is_test_mode() is False


def test_is_test_mode_returns_false_when_unset(monkeypatch):
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    assert is_test_mode() is False


def test_is_test_mode_returns_false_for_empty(monkeypatch):
    monkeypatch.setenv("PYTHIA_TEST_MODE", "")
    assert is_test_mode() is False
