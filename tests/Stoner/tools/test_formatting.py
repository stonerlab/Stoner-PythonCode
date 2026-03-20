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
    assert formatting.ordinal(12).endswith("th"), "Failed special handling for 12th in ordinal"
    assert formatting.ordinal(13).endswith("th"), "Failed special handling for 13th in ordinal"
    assert formatting.ordinal(21).endswith("st"), "Failed to add st to 21st in ordinal"
    assert formatting.ordinal(1).endswith("st"), "Failed to add st to 1st in ordinal"
    assert formatting.ordinal(2).endswith("nd"), "Failed to add nd to 2nd in ordinal"
    assert formatting.ordinal(3).endswith("rd"), "Failed to add rd to 3rd in ordinal"
    assert formatting.ordinal(4).endswith("th"), "Failed to add th to 4th in ordinal"
    assert formatting.ordinal(0).endswith("th"), "Failed to add th to 0th in ordinal"


def test_quantize():
    assert formatting.quantize(1.23, 0.1) == pytest.approx(1.2), "quantize(1.23, 0.1) failed"
    assert formatting.quantize(1.26, 0.1) == pytest.approx(1.3), "quantize(1.26, 0.1) failed"
    assert formatting.quantize(7, 2) == pytest.approx(8), "quantize(7, 2) failed"
    assert formatting.quantize(6, 2) == pytest.approx(6), "quantize(6, 2) failed"


def test_tex_escape():
    assert formatting.tex_escape("&") == r"\&", "tex_escape & failed"
    assert formatting.tex_escape("%") == r"\%", "tex_escape % failed"
    assert formatting.tex_escape("$") == r"\$", "tex_escape $ failed"
    assert formatting.tex_escape("#") == r"\#", "tex_escape # failed"
    assert formatting.tex_escape("_") == r"\_", "tex_escape _ failed"
    assert formatting.tex_escape("{") == r"\{", "tex_escape { failed"
    assert formatting.tex_escape("}") == r"\}", "tex_escape } failed"
    assert formatting.tex_escape("~") == r"\textasciitilde{}", "tex_escape ~ failed"
    assert formatting.tex_escape("^") == r"\^{}", "tex_escape ^ failed"
    assert formatting.tex_escape("\\") == r"\textbackslash{}", "tex_escape \\ failed"
    assert formatting.tex_escape("<") == r"\textless", "tex_escape < failed"
    assert formatting.tex_escape(">") == r"\textgreater", "tex_escape > failed"
    assert formatting.tex_escape("hello") == "hello", "tex_escape plain text should be unchanged"


def test_format_val_modes():
    value = 1.2345e-6
    # eng mode text
    result = formatting.format_val(value, fmt="text", mode="eng")
    assert "u" in result or "1" in result, "format_val eng text mode failed"
    # sci mode html
    result = formatting.format_val(value, fmt="html", mode="sci")
    assert "10" in result, "format_val sci html mode failed"
    # sci mode latex
    result = formatting.format_val(value, fmt="latex", mode="sci")
    assert r"\times" in result, "format_val sci latex mode failed"
    # float mode (default)
    result = formatting.format_val(value, fmt="text", mode="float")
    assert "1.2345" in result, "format_val float text mode failed"
    # bad mode raises RuntimeError
    try:
        formatting.format_val(value, fmt="text", mode="bad")
    except RuntimeError:
        pass
    else:
        assert False, "format_val bad mode didn't raise RuntimeError"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])

