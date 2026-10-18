"""VSM corrections using the current Data selection and numerical APIs."""

import numpy as np

from Stoner import Data
from Stoner.analysis.fitting.models.generic import linear


class VSMAnalysis(Data):
    """Augment Data with in-place magnetometry corrections and loop measurements."""

    def _replace_signal(self, column, values):
        """Commit corrected values without clearing the column's existing exclusions."""
        mask = np.ma.getmaskarray(self.column(column)) | np.ma.getmaskarray(values)
        self[:, column] = np.ma.array(values, mask=mask)

    def true_m(self):
        """Recalculate moment and its volume/mass normalisations from lock-in magnitude."""
        s_vol = np.ma.masked_invalid(self.column("Mvol") / self.column("m (emu)")).mean()
        s_mass = np.ma.masked_invalid(self.column("Mmass") / self.column("m (emu)")).mean()
        l_ratio = np.ma.masked_invalid(self.column("m (emu)") / self.column("X")).mean()
        moment = np.ma.sqrt(self.column("X") ** 2 + self.column("Y") ** 2) * l_ratio
        self._replace_signal("m (emu)", moment)
        self._replace_signal("Mvol", moment * s_vol)
        self._replace_signal("Mmass", moment * s_mass)
        return self

    def correct_drift(self, threshold=0.95):
        """Subtract the fitted time dependence of each lock-in-derived signal.

        Args:
            threshold (float):
                Fraction of the maximum field above which points are fitted.

        Returns:
            VSMAnalysis:
                The corrected object, retaining existing exclusions and metadata.
        """
        field = np.ma.masked_invalid(self.column("H_vsm"))
        high_field = np.ma.filled(field > threshold * field.max(), False)
        for column in self.find_col(["m (emu)", "Mvol", "Mmass", "X", "Y"]):
            coeff, _ = self.curve_fit(linear, "Time", column, where=high_field)
            self._replace_signal(column, self.column(column) - linear(self.column("Time"), *coeff))
        return self

    def remove_diamagnetism_and_offset(self, threshold=0.85):
        """Subtract the average linear background fitted to both saturated arms.

        Args:
            threshold (float):
                Fraction of the positive/negative extreme field defining saturation.

        Returns:
            VSMAnalysis:
                The corrected object, retaining existing exclusions and metadata.
        """
        field = np.ma.masked_invalid(self.column("H_vsm"))
        upper = np.ma.filled(field > field.max() * threshold, False)
        lower = np.ma.filled(field < field.min() * threshold, False)
        for column in self.find_col(["m (emu)", "Mvol", "Mmass", "X", "Y"]):
            positive, _ = self.curve_fit(linear, "H_vsm", column, where=upper)
            negative, _ = self.curve_fit(linear, "H_vsm", column, where=lower)
            coeff = (positive + negative) / 2
            self._replace_signal(column, self.column(column) - linear(self.column("H_vsm"), *coeff))
        return self

    def _crossings(self, column, coordinate):
        """Find falling/rising zero crossings after the initial maximum field."""
        original_mask = self.mask.copy()
        try:
            turning_point = int(np.ma.argmax(np.ma.masked_invalid(self.column("H_vsm"))))
            self.mask |= ~np.isfinite(self.to_numpy(masked=False))
            self.mask[:turning_point, :] = True
            falling = self.threshold(0.0, col=column, rising=False, falling=True, xcol=coordinate)
            rising = self.threshold(0.0, col=column, rising=True, falling=False, xcol=coordinate)
            return falling, rising
        finally:
            self.mask = original_mask

    def find_Hc(self):
        """Return the falling/rising moment-zero fields without changing exclusions."""
        return self._crossings("m (emu)", "H_vsm")

    def find_Br(self):
        """Return the falling/rising field-zero moments without changing exclusions."""
        return self._crossings("H_vsm", "m (emu)")
