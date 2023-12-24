# -*- coding: utf-8 -*-
"""Functions for formatting data."""

__all__ = ["format_error", "format_val", "quantize", "tex_escape", "ordinal"]

import re
from typing import Optional, Any
from html import escape as html_escape
from numpy import log10, floor, abs, isnan, round  # pylint: disable=redefined-builtin
from ..compat import bytes2str
from ..core.Typing import Numeric, NumericArray

prefs = {
    "text": {
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
        -3: "m",
        -6: "u",
        -9: "n",
        -12: "p",
        -15: "f",
        -18: "a",
        -21: "z",
        -24: "y",
    },
    "latex": {
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
        -3: "m",
        -6: r"\mu",
        -9: "n",
        -12: "p",
        -15: "f",
        -18: "a",
        -21: "z",
        -24: "y",
    },
    "html": {
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
        -3: "m",
        -6: r"&micro;",
        -9: "n",
        -12: "p",
        -15: "f",
        -18: "a",
        -21: "z",
        -24: "y",
    },
}


def format_error(value: Numeric, error: Optional[Numeric] = None, **kargs: Any) -> str:
    r"""Format answer with the uncertaintly to 1sf and the value to no more sf's than the uncertainty.

    Args:
        value (float):
            The value to be formatted
        error (float):
            The uncertainty in the value

    Keyword Arguments:
        fmt (str):
            Specify the output format, options are:
                -  "text" - plain text output
                - "latex" - latex output
                - "html" - html entities
        escape (bool):
            Specifies whether to escape the prefix and units for unprintable characters in non text formats (default
            False)
        mode (string):
            If "float" (default) the number is formatted as is, if "eng" the value and error is converted
            to the next samllest power of 1000 and the appropriate SI index appended. If mode is "sci" then a
            scientific, i.e. mantissa and exponent format is used.
        units (string):
            A suffix providing the units of the value. If si mode is used, then appropriate si prefixes are
            prepended to the units string. In LaTeX mode, the units string is embedded in \mathrm
        prefix (string):
            A prefix string that should be included before the value and error string. in LaTeX mode this is
            inside the math-mode markers, but not embedded in \mathrm.

    Returns:
        String containing the formatted number with the eorr to one s.f. and value to no more d.p. than the error.
    """
    mode = kargs.get("mode", "float")
    units = kargs.get("units", "")
    prefix = kargs.get("prefix", "")
    latex = kargs.get("latex", False)
    fmt = kargs.get("fmt", "latex" if latex else "text")
    escape = kargs.get("escape", False)
    escape_func = {"latex": tex_escape, "html": html_escape}.get(fmt, lambda x: x)

    if error == 0.0 or isnan(error):  # special case for zero uncertainty
        return format_val(value, **kargs)

    if escape:
        prefix = escape_func(prefix)
        units = escape_func(units)

    # Sort out special formatting for different modes
    if mode == "float":  # Standard
        suffix_val = ""
    elif mode == "eng":  # Use SI prefixes
        if -1 <= floor(log10(abs(value))) <= 3:
            v_mag = 0
        else:
            v_mag = floor(log10(abs(value)) / 3.0) * 3.0
        prefixes = prefs.get(fmt, prefs["text"])
        if v_mag in prefixes:
            if fmt == "latex":
                suffix_val = rf"\,\mathrm{{{{{prefixes[v_mag]}}}}}"
            else:
                suffix_val = " " + prefixes[v_mag]
            value /= 10**v_mag
            error /= 10**v_mag
        else:  # Implies 10^-3<x<10^3
            suffix_val = ""
    elif mode == "sci":  # Scientific mode - raise to common power of 10
        v_mag = floor(log10(abs(value)))
        if fmt == "latex":
            suffix_val = rf"\times 10^{{{{{int(v_mag)}}}}}\,"
        elif fmt == "html":
            suffix_val = f"&times; 10<sup>{int(v_mag)}</sup> "
        else:
            suffix_val = f"E{int(v_mag)} "
        value /= 10**v_mag
        error /= 10**v_mag
    else:  # Bad mode
        raise RuntimeError(f"Unrecognised mode: {mode} in format_error")

    # Now do the rounding of the value based on error to 1 s.f.
    e2 = error
    u_mag = floor(log10(abs(error)))  # work out the scale of the error
    error = round(error / 10**u_mag) * 10**u_mag  # round the error, but this could round to 0.x0
    u_mag = floor(log10(error))  # so go round the loop again
    error = round(e2 / 10**u_mag) * 10**u_mag  # and get a new error magnitude
    value = round(value / 10**u_mag) * 10**u_mag
    u_mag = min(0, u_mag)  # Force integer results to have no dp

    # Protect {} in units string
    units = units.replace("{", "{{").replace("}", "}}")
    prefix = prefix.replace("{", "{{").replace("}", "}}")
    if fmt == "latex":  # Switch to latex math mode symbols
        val_fmt_str = rf"${prefix}{{:.{int(abs(u_mag))}f}}\pm "
        if units != "":
            suffix_fmt = rf"\mathrm{{{{{units}}}}}"
        else:
            suffix_fmt = ""
        suffix_fmt += "$"
    elif fmt == "html":  # Switch to latex math mode symbols
        val_fmt_str = rf"{prefix}{{:.{int(abs(u_mag))}f}}&plusmin;"
        suffix_fmt = units
    else:  # Plain text
        val_fmt_str = rf"{prefix}{{:.{int(abs(u_mag))}f}}+/-"
        suffix_fmt = units
    if u_mag < 0:  # the error is less than 1, so con strain decimal places
        err_fmt_str = r"{:." + str(int(abs(u_mag))) + "f}"
    else:  # We'll be converting it to an integer anyway
        err_fmt_str = r"{}"
    fmt_str = val_fmt_str + err_fmt_str + suffix_val + suffix_fmt
    if error >= 1.0:
        error = int(error)
        value = int(value)
    return fmt_str.format(value, error)


def format_val(value: Numeric, **kargs: Any) -> str:
    r"""Format a number as an SI quantity.

    Args:
        value(float):
            Value to format

    Keyword Arguments:
        fmt (str):
            Specify the output format, options are:
                -  "text" - plain text output
                - "latex" - latex output
                - "html" - html entities
        escape (bool):
            Specifies whether to escape the prefix and units for unprintable characters in non text formats (default
            False)
        mode (string):
            If "float" (default) the number is formatted as is, if "eng" the value and error is converted
            to the next samllest power of 1000 and the appropriate SI index appended. If mode is "sci" then a
            scientific, i.e. mantissa and exponent format is used.
        units (string):
            A suffix providing the units of the value. If si mode is used, then appropriate si prefixes are
            prepended to the units string. In LaTeX mode, the units string is embedded in \mathrm
        prefix (string):
            A prefix string that should be included before the value and error string. in LaTeX mode this is
            inside the math-mode markers, but not embedded in \mathrm.

    Returns:
        (str): The formatted value.
    """
    mode = kargs.pop("mode", "float")
    units = kargs.pop("units", "")
    prefix = kargs.pop("prefix", "")
    latex = kargs.pop("latex", False)
    fmt = kargs.pop("fmt", "latex" if latex else "text")
    places = kargs.pop("places", False)
    escape = kargs.pop("escape", False)
    escape_func = {"latex": tex_escape, "html": html_escape}.get(fmt, lambda x: x)

    if escape:
        prefix = escape_func(prefix)
        units = escape_func(units)

    if mode == "float":  # Standard
        suffix_val = ""
        v_mag = floor(log10(abs(value)))
    elif mode == "eng":  # Use SI prefixes
        v_mag = floor(log10(abs(value)) / 3.0) * 3.0
        prefixes = prefs.get(fmt, prefs["text"])
        if v_mag in prefixes:
            if fmt == "latex":
                suffix_val = rf"\mathrm{{{{{prefixes[v_mag]}}}}}"
            else:
                suffix_val = prefixes[v_mag]
            value /= 10**v_mag
        else:  # Implies 10^-3<x<10^3
            suffix_val = ""
    elif mode == "sci":  # Scientific mode - raise to common power of 10
        v_mag = floor(log10(abs(value)))
        if fmt == "latex":
            suffix_val = rf"\times 10^{{{{{int(v_mag)}}}}}"
        elif fmt == "html":
            suffix_val = f"&times; 10<sup>{int(v_mag)}</sup> "
        else:
            suffix_val = f"E{int(v_mag)} "
        value /= 10**v_mag
    else:  # Bad mode
        raise RuntimeError(f"Unrecognised mode: {mode} in format_error")

    # Protect {} in units string
    units = units.replace("{", "{{").replace("}", "}}")
    prefix = prefix.replace("{", "{{").replace("}", "}}")
    if fmt == "latex":  # Switch to latex math mode symbols
        val_fmt_str = rf"${prefix}{{}}"
        if units != "":
            suffix_fmt = rf"\mathrm{{{{{units}}}}}"
        else:
            suffix_fmt = ""
        suffix_fmt += "$"
    elif fmt == "html":  # Switch to latex math mode symbols
        val_fmt_str = rf"{prefix}{{}}"
        suffix_fmt = units
    else:  # Plain text
        val_fmt_str = rf"{prefix}{{}}"
        suffix_fmt = units
    fmt_str = val_fmt_str + suffix_val + suffix_fmt
    if places:
        value = round(value, places)
    ret = fmt_str.format(value)
    return ret


def quantize(number: NumericArray, quantum: Numeric) -> NumericArray:
    """Round a number to the nearest multiple of a quantum.

    Args:
        number (float,array):
            Number(s) to be rounded to the nearest qyuantum
        quantum (float):
            Quantum to round to
    Returns:
        number rounded to qunatum
    """
    return round(number / quantum) * quantum


def tex_escape(text: str) -> str:
    """Escapes spacecial text characters in a string.

    Parameters:
        text (str):
            a plain text message

    Returns:
        the message escaped to appear correctly in LaTeX

    From `Stackoverflow <http://stackoverflow.com/questions/16259923/
    how-can-i-escape-latex-special-characters-inside-django-templates>`
    """
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless",
        ">": r"\textgreater",
    }
    regex = re.compile("|".join(re.escape(bytes2str(key)) for key in sorted(conv.keys(), key=lambda item: -len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def ordinal(value: int) -> str:
    """Format an integer into an ordinal string.

    Args:
        value (int):
            Number to be written as an ordinal string

    Return:
        (str):
            Ordinal String such as '1st','2nd' etc.
    """
    if not isinstance(value, int):
        raise ValueError

    last_digit = value % 10
    if value % 100 in [11, 12, 13]:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"][last_digit]

    return "{}{}".format(value, suffix)
