"""Tests for the helpers in ccres_disdrometer_processing.utils."""

from ccres_disdrometer_processing.utils import format_ql_file_prefix


def test_format_ql_file_prefix_adds_the_index_pattern() -> None:
    assert format_ql_file_prefix("ql_summary") == "ql_summary_{:02d}.png"


def test_format_ql_file_prefix_drops_the_extension() -> None:
    assert format_ql_file_prefix("ql_summary.png") == "ql_summary_{:02d}.png"
    assert format_ql_file_prefix("out/ql.png") == "out/ql_{:02d}.png"
