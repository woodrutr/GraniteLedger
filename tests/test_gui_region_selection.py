"""Tests for region selection normalization behavior in the GUI."""

from gui.app import _normalize_region_labels


def test_normalize_removes_individuals_when_all_selected_after_individuals():
    previous = ('Northeast',)
    selection = ['Northeast', 'All']

    assert _normalize_region_labels(selection, previous) == ['All']


def test_normalize_removes_all_when_switching_from_all_to_individual():
    previous = ('All',)
    selection = ['All', 'Northeast']

    assert _normalize_region_labels(selection, previous) == ['Northeast']


def test_normalize_preserves_multiple_individuals_when_all_was_selected():
    previous = ('All',)
    selection = ['All', 'Northeast', 'Southwest']

    assert _normalize_region_labels(selection, previous) == ['Northeast', 'Southwest']


def test_normalize_handles_missing_previous_state():
    selection = ['All', 'Northeast']

    assert _normalize_region_labels(selection, None) == ['All']


def test_normalize_leaves_all_only_selection_unchanged():
    previous = ('All',)
    selection = ['All']

    assert _normalize_region_labels(selection, previous) == ['All']
