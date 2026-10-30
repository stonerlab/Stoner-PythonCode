"""Provide string-valued choices for Stoner's public APIs."""

from enum import Enum, StrEnum

__all__ = ["FitOutput"]


class FitOutput(StrEnum):
    """Select an existing fitting output without changing its result contract.

    Notes:
        Members retain the corresponding legacy string values. Reports and full
        results remain algorithm-specific. Polynomial fitting has no output
        selector. Repeated-initial-guess lmfit calls retain their existing array
        return, except DATA which returns a results table.
    """

    #: Fitted parameters and covariance.
    FIT = "fit"
    #: The algorithm's report object.
    REPORT = "report"
    #: Interleaved parameters/errors and goodness of fit.
    ROW = "row"
    #: The algorithm's full-result tuple.
    FULL = "full"
    #: The measurement object, or a repeated-fit results table.
    DATA = "data"
    #: A dictionary of parameters, errors and fit statistics.
    DICT = "dict"


def _normalise_fit_output(output):
    """Unwrap a fitting choice while retaining legacy string dispatch."""
    if isinstance(output, FitOutput):
        return output.value
    if isinstance(output, Enum):
        raise TypeError("output must be a FitOutput member or an existing fitting output string")
    return output
