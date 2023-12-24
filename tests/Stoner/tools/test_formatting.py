# -*- coding: utf-8 -*-
"""Tests for Stoner.tools.formatting"""
import pytest
from Stoner.tools import formatting


def test_format_error():
    value = 1.2345e-6
    error = 2.456e-8
    assert formatting.format_error(value, error, fmt="text") == "0.00000123+/-0.00000002"
    assert formatting.format_error(value, error, fmt="html") == "0.00000123&plusmin;0.00000002"
    assert formatting.format_error(value, error, fmt="latex") == r"$0.00000123\pm 0.00000002$"

    assert formatting.format_error(value, error, fmt="text", mode="sci").strip() == "1.23+/-0.02E-6"
    assert (
        formatting.format_error(value, error, fmt="html", mode="sci").strip()
        == "1.23&plusmin;0.02&times; 10<sup>-6</sup>"
    )
    assert formatting.format_error(value, error, fmt="latex", mode="sci").strip() == r"$1.23\pm 0.02\times 10^{-6}\,$"

    assert formatting.format_error(value, error, fmt="text", mode="eng").strip() == "1.23+/-0.02 u"
    assert formatting.format_error(value, error, fmt="html", mode="eng").strip() == "1.23&plusmin;0.02 &micro;"
    assert formatting.format_error(value, error, fmt="latex", mode="sci").strip() == r"$1.23\pm 0.02\times 10^{-6}\,$"

    assert (
        formatting.format_error(value, error, fmt="latex", mode="eng", units="<bad\\hr>", escape=True).strip()
        == r"$1.23\pm 0.02\,\mathrm{\mu}\mathrm{\textlessbad\textbackslash{}hr\textgreater}$"
    )


def test_format_value():
    value = 1.2345e-6
    error = 0
    assert formatting.format_error(value, error, fmt="text") == "1.2345e-06"
    assert formatting.format_error(value, error, fmt="html") == "1.2345e-06"
    assert formatting.format_error(value, error, fmt="latex") == r"$1.2345e-06$"

    assert formatting.format_error(value, error, fmt="text", mode="sci").strip() == "1.2345000000000002E-6"
    assert (
        formatting.format_error(value, error, fmt="html", mode="sci").strip()
        == "1.2345000000000002&times; 10<sup>-6</sup>"
    )
    assert (
        formatting.format_error(value, error, fmt="latex", mode="sci").strip()
        == r"$1.2345000000000002\times 10^{-6}$"
    )

    assert formatting.format_error(value, error, fmt="text", mode="eng").strip() == "1.2345000000000002u"
    assert formatting.format_error(value, error, fmt="html", mode="eng").strip() == "1.2345000000000002&micro;"
    assert (
        formatting.format_error(value, error, fmt="latex", mode="sci").strip()
        == r"$1.2345000000000002\times 10^{-6}$"
    )

    assert (
        formatting.format_error(value, error, fmt="latex", mode="eng", units="<bad\\hr>", escape=True).strip()
        == r"$1.2345000000000002\mathrm{\mu}\mathrm{\textlessbad\textbackslash{}hr\textgreater}$"
    )


def test_ordinal():
    try:
        formatting.ordinal("G")
    except ValueError:
        pass
    else:
        assert False, "ordinal didn't raise a ValueError for non integer value"
    assert formatting.ordinal(11).endswith("th"), "Failed special handling for 11th in ordinal"
    assert formatting.ordinal(21).endswith("st"), "Failed to add st to 21st in ordinal"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
