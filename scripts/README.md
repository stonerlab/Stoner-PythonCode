# Analysis scripts

These repository scripts use the Stoner 0.12 plotting accessor and pandas/xarray
storage APIs. Run them in the supported project environment; see
[tests/README.md](../tests/README.md). They are examples for particular scientific
workflows, so check column names, units and analysis settings against your data.
Paths to bundled samples are resolved relative to the script, so the current
working directory does not have to be `scripts`.

| Script | Input and entry point |
| --- | --- |
| `B-G Fit.py` | Run to select a resistance/temperature file, or call `fit(filename)`. Guesses the two columns and prompts if ambiguous; fits with exponent `n=5`. |
| `Tc_Fit.py` | Run against the bundled temperature-ramp sample, or call `fit(filename, t_col=..., r_cols=..., iterator=...)`. Fits each resistance channel independently and returns a Tc/error table. |
| `Kiessig.py` | Run to select a low-angle reflection scan, or call `analyse(filename, sensitivity=..., critical_edge=..., fringe_offset=...)`. Requires `Angle`, `Counts` and wavelength metadata `Lambda` in angstroms. Returns the scan, peak table, thickness and slope-derived uncertainty. |
| `Mask_selection.py` | Run for interactive masking of the bundled image. `demonstrate(filename, select=False)` runs the processing with the supplied image mask. Returns the normalised image and a quantised clone. |
| `PCAR-New.py` | Configuration-driven PCAR fitting; use the command below. `working().load_config(config, filename).Fit()` also runs without a dialog. |
| `BNL_FileSplitter.py` | Run to enter a directory and filename, or call `split_file(filename, output_directory=...)`. Writes numbered scans and `title.txt` into `BNLSplitFiles` by default. Matching scan outputs are overwritten; one-point scans are retained and empty scans omitted. |
| `VSManalysis_v2.py` | Interactive VSM editor. Saves corrected copies under `EditedFiles`. The numerical corrections use a bounded `edit_numpy()` transaction. |
| `VSManalysis_v3.py` | Import `VSMAnalysis` for chainable moment, drift and background corrections and zero-crossing measurements. Exclusions and metadata are retained. |
| `XMCD_Reduction.py` | `python scripts/XMCD_Reduction.py DATA_DIRECTORY --output-directory OUTPUT_DIRECTORY`. Without the output option it returns/analyses results without saving them. `reduce(...)` also accepts a DataFolder. |
| `GenX/GenX PNR Asymmetry Plotting.py` | Run to select a directory of GenX exports, or call `plot_exports(folder)`. Expects three datasets ordered by `dataset`: SLD, down-spin and up-spin reflectivity. |
| `GenX/stoner.py` | External GenX data-loader plugin; install in the GenX data-loader plugin directory. Uses `genx.plugins` and the current dataset-object loader callback. Requires GenX and wxPython; it is not a standalone script. |

For example, from the repository root:

```powershell
python scripts/Tc_Fit.py
python scripts/PCAR-New.py "sample-data/PCAR Co Data.csv" --config scripts/PCAR-New.ini
python scripts/PCAR-New.py "sample-data/PCAR Co Data.csv" --config "scripts/PCAR-chi^2.ini"
```

The PCAR configuration controls showing plots, saving data, reporting, rescaling,
discarding high-bias data, decomposition and normalisation. `PCAR-chi^2.ini`
fixes/steps the polarisation parameter to map the fit quality. Edit these settings
for the measurement; default `PCAR-New.ini` requests a save dialog after fitting.
The INI files also document the CSV loader controls: zero-based
`header_line` and `data_line`, plus independent `header_delim` and `data_delim`.
The loader can skip preambles/intervening lines and use different header/data separators.
`header_line: None` selects headerless input with numeric column selectors;
`None` delimiters request automatic detection. Defaults match the bundled CSV.
The historical names `start_line` and `separator` are described alongside their
current replacements, and the INI files remain a reference for supported options.
The INI-to-loader forwarding defect is deferred to
[issue #41](https://github.com/stonerlab/Stoner-PythonCode/issues/41):
`cfg_data_from_ini()` currently ignores those layout entries. Custom layouts work
when passed directly to `Data.load(..., filetype="CSVFile", ...)`; the bundled
PCAR examples use the loader defaults. Three strict expected-failure regressions
record the deferred INI behaviour and will flag an unexpected pass when it is fixed.

All standalone scripts can be imported with `runpy.run_path()` without opening
dialogs, creating figures or writing results. Functions accepting a file or Data
object copy that input unless they explicitly describe an in-place correction.
Drawing uses `data.plot(...)`/named methods, formatting uses the returned Axes,
and the entry points explicitly show their figures. Panels and insets use the
stonerplots contexts exposed by `data.plot`.

XMCD reduction retains the original intensity-difference definition `I+ - I-`
for the column named `Asymmetry`. Grouped scans and opposite helicities must have
matching energy grids; the reduction does not regrid them. The default run range,
geometry keys and background/normalisation windows are specific to the original
I10 example and can be supplied to `reduce(...)`.

Run the script regressions headlessly with:

```powershell
$env:MPLBACKEND = 'Agg'
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest tests/Stoner/test_scripts.py
```

Tests exercise real resistivity, temperature-ramp, PCAR, VSM, BNL, diffraction and
image samples, plus controlled fringe/XMCD/GenX data. The GenX loader tests supply
the external host interface and exercise real Stoner loading/selection. They do
not certify the plugin UI in a running GenX application; interactive mask and
VSM user input are likewise separate from headless numerical validation.
The GenX callback follows the upstream
[data-loader framework](https://github.com/aglavic/genx/blob/master/genx/genx/plugins/data_loader_framework.py).
