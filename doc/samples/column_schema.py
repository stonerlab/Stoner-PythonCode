"""Select duplicate measurement channels and preserve their roles and masks."""

import re

import numpy as np

from Stoner import Data


# Two voltage measurements, each with its own field and voltage uncertainty.
field = np.linspace(-1.0, 1.0, 21)
values = np.column_stack((field, np.full(21, 0.01), 2 * field + 0.1, np.full(21, 0.04),
                          field, np.full(21, 0.02), 3 * field - 0.2, np.full(21, 0.06)))
measurement = Data(values, setas="xdyexdye",
                   column_headers=["Field (T)", "dField (T)", "Voltage (V)", "dVoltage (V)"] * 2)
measurement.mask[10, 6] = True

# An exact name wins, even when it contains punctuation. All duplicates match.
assert measurement.find_col("Voltage (V)") == [2, 6]
assert measurement.find_col("Volt*") == [2, 6]  # Whole-header glob.
assert measurement.find_col("Voltage") == [2, 3, 6, 7]  # Literal substring.
assert measurement.find_col(re.compile(r"^voltage", re.I)) == [2, 6]
assert measurement.find_col("Field (T)", first=True, force_list=True) == [0]

# Scalar statistics require one explicit channel, rather than silently choosing.
try:
    measurement.max("Voltage (V)")
except ValueError as error:
    assert "exactly one" in str(error)
else:
    raise AssertionError("A scalar statistic must reject duplicate voltage names")
first_voltage = measurement.find_col("Voltage (V)", first=True)
maximum_voltage, maximum_row = measurement.max(first_voltage)
assert np.isclose(maximum_voltage, 2.1) and maximum_row == 20

# Saved views follow the owner. Cloned descriptions are independent.
roles = measurement.setas
headers = measurement.column_headers
detached = roles.clone
measurement.rename("Voltage (V)", "Signal (V)")
assert headers[2] == headers[6] == "Signal (V)"
assert detached.column_headers[2] == "Voltage (V)"
roles["Signal (V)"] = "y"  # Broadcast one role to both exact matches.

# Reorder a clone and repeat its masked signal column. Data, masks and roles move
# together; the first occurrence keeps its identity and the second gets a new one.
repeated = measurement.clone
source_ids = repeated.column_ids
repeated.reorder_columns([4, 5, 6, 7, 6])
assert repeated.column_ids[:4] == tuple(source_ids[i] for i in [4, 5, 6, 7])
assert repeated.column_ids[4] not in source_ids
assert repeated.mask[10, 2] and repeated.mask[10, 4]
assert list(repeated.setas) == list("xdyey")
np.testing.assert_array_equal(repeated.to_numpy().data[:, 2], repeated.to_numpy().data[:, 4])

# Lossless interchange retains IDs, exclusions and roles. Native pandas access
# uses ordinary exact labels and raw values, without applying the Stoner mask.
restored = Data.from_storage(repeated.export_storage())
assert restored.column_ids == repeated.column_ids
np.testing.assert_array_equal(restored.to_numpy().mask, repeated.to_numpy().mask)
assert restored.data.loc[:, "Signal (V)"].shape[1] == 2
assert restored.data.iloc[10, 2] == repeated.to_numpy().data[10, 2]

# Use the masked public plotting path. Keep the example runnable under pytest.
ax = measurement.plot(startx=0, fmt="o", label="Sensor 1")
measurement.plot(startx=4, ax=ax, fmt="s", label="Sensor 2")
# The second group's assigned errors come from absolute columns 5 and 7.
second_errors = ax.containers[-1]
horizontal, vertical = second_errors.lines[2]
np.testing.assert_allclose(np.diff(horizontal.get_segments()[0][:, 0]), [0.04])
np.testing.assert_allclose(np.diff(vertical.get_segments()[0][:, 1]), [0.12])
measurement.plot.title("Duplicate channel names with independent uncertainties")
