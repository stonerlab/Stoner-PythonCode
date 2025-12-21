# -*- coding: utf-8 -*-
"""VSM Analysis v3.

Created on Sat Aug 24 20:18:05 2013

@author: Gavin Burnell
"""
# pylint: disable=invalid-name
import numpy as np

from Stoner import Data
from Stoner.analysis.fitting.models.generic import Linear


class VSMAnalysis(Data):
    """Augment Data with some extra methods."""

    def true_m(self):
        """Calculate correct m from lockin X and Y components."""
        # Get some constants that scale betweent he columns
        s_vol = np.mean(self.column("Mvol") / self.column("m (emu)"))
        s_mass = np.mean(self.column("Mmass") / self.column("m (emu)"))
        l_ratio = np.mean(self.column("m (emu)") / self.column("X"))

        # Now calculate new column values and put them into self.data
        self.data[:, self.find_col("m (emu)")] = (
            np.sqrt(self.column("X") ** 2 + self.column("Y") ** 2) * l_ratio
        )
        self.data[:, self.column("Mvol")] = self.column("m (emu)") * s_vol
        self.data[:, self.column("Mmass")] = self.column("m (emu)") * s_mass
        return self

    def correct_drift(self, threshold=0.95):
        """Correct for drift in the signal.

        Masks out data that isn't above threshold*maximum field and fits a straight line to
        Moment(Time) and then subtracts this from the data.

        Args:
            threshold (float): fraction of the maximum field to look at to work out drift in time.

        Returns:
            the current object with a new corrected moment.
        """
        H_max = max(self.column("H_vsm"))
        for m in self.find_col(
            ["m (emu)", "Mvol", "Mmass", "X", "Y"]
        ):  # Correct for all lockin derived signals
            self.filter(
                lambda r: r[0] > threshold * H_max, ["H_vsm"]
            )  # Mask out only the max filed data
            coeff, _ = self.curve_fit(
                Linear, "Time", m
            )  # Do linear fit wrt to time
            self.mask = False  # Now we work with all the data
            correct_m = self.column(m) - Linear(
                self.column("Time"), *coeff
            )  # calculate corrected data
            self.data[:, m] = correct_m  # and push it back

    def remove_diamagnetism_and_offset(self, threshold=0.85):
        """Remove a diamagnetic component from the data.

        Fits straight lines to the upper and lower parts of the curve (within
        threshold of the extreme fields) and uses this to remove diamangeteic omponents
        and recentre the loop

        Args:
            threshold (float): Fraction of maximum/minimum  field to assume is saturated

        ReturnsL
            a copy of self with the corrections applied
        """
        H_max = max(self.column("H_vsm"))
        H_min = min(self.columns("H_vsm"))
        for m in self.find_col(["m (emu)", "Mvol", "Mmass", "X", "Y"]):
            self.filter(
                lambda r: r[0] > H_max * threshold, ["H_vsm"]
            )  # mask out everything expcet max field data
            pcoeff, _ = self.curve_fit(Linear, "H_vsm", m)  # Get a linear fit
            self.filter(
                lambda r: r[0] < H_min * threshold, ["H_vsm"]
            )  # mask out everything except min field data
            ncoeff, _ = self.curve_fit(Linear, "H_vsm", m)  # Get a linear fit
            coeff = (
                pcoeff + ncoeff
            ) / 2.0  # Average the co-=fficients of the two fits
            correct_m = self.column(m) - Linear(
                self.column("H_vsm"), *coeff
            )  # Calculate corrected fits
            self.data[:, m] = correct_m  # Apply corrected fits
        return self

    def find_Hc(self):
        """Use thresholding and interpolation to find fields for zero crossing moments."""
        h_m = int(self.peaks(ycol="H_vsm", wiodth=15)[0])
        mask = self.mask
        self.mask = np.zeros(self.shape)
        self.mask[1:h_m, :] = True
        hc_p = self.threshold(0.0, col="m (emu)", xcol="H_vsm")
        hc_m = self.threshold(
            0.0, col="m (emu)", rising=False, falling=True, xcol="H_vsm"
        )
        self.mask = mask
        return (hc_m, hc_p)

    def find_Br(self):
        """Use thresholding and interpolation to find moments for zero crossing fields."""
        h_m = int(self.peaks(ycol="H_vsm", width=15)[0])
        mask = self.mask
        self.mask = np.zeros(self.shape)
        self.mask[1:h_m, :] = True
        br_p = self.threshold(0.0, col="H_vsm", xcol="m (emu)")
        br_m = self.threshold(
            0.0, col="H_vsm", rising=False, false=True, xcol="m (emu)"
        )
        self.mask = mask
        return (br_m, br_p)
