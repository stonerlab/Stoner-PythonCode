"""Exercise the repository analysis scripts against the public Stoner API."""

import configparser
from pathlib import Path
import runpy
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest

from Stoner import Data, DataFolder


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SAMPLES = ROOT / "sample-data"
pytestmark = [pytest.mark.analysis, pytest.mark.plotting]


@pytest.fixture(autouse=True)
def isolated_figures():
    """Render and close test-owned figures without leaking style changes."""
    with plt.rc_context():
        yield
        for number in plt.get_fignums():
            plt.figure(number).canvas.draw()
    plt.close("all")


def load_script(name):
    """Load a script without executing its interactive entry point."""
    return runpy.run_path(str(SCRIPTS / name))


@pytest.mark.parametrize("script", sorted(path.relative_to(SCRIPTS) for path in SCRIPTS.rglob("*.py")
                                         if path.name != "stoner.py"))
def test_import_is_noninteractive(script, monkeypatch, tmp_path):
    """Imports must not prompt, load data, create figures or depend on the working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Unexpected input prompt"))
    load_script(script)
    assert not plt.get_fignums()
    assert not list(tmp_path.iterdir())


def test_bloch_gruneisen_real_resistivity():
    data = load_script("B-G Fit.py")["fit"](SAMPLES / "Cu_resistivity_vs_T.txt")
    assert np.isfinite(data["BlochGrueneisen:thetaD"])
    assert data["BlochGrueneisen:n"] == 5
    assert data.plot.ax.get_xlabel() == "Temperature (K)"
    assert len(data.plot.ax.lines) == 2


def test_critical_temperature_real_ramps():
    data = load_script("Tc_Fit.py")["fit"]()
    assert data.shape[1] == 5 and len(data) >= 2
    assert list(data.setas) == list("xyeye")
    assert np.all((data.to_numpy()[:, [1, 3]] > 5) & (data.to_numpy()[:, [1, 3]] < 10))
    assert np.all(data.to_numpy()[:, [2, 4]] >= 0)


def test_kiessig_fringe_thickness_and_inset():
    angle = np.linspace(0.1, 4, 2001)
    wavelength, thickness = 1.54, 180
    counts = 1e4 * np.exp(-angle) * (1 + 0.25 * np.cos(4 * np.pi * thickness
                                                       * np.sin(np.radians(angle / 2)) / wavelength))
    source = Data(np.column_stack((angle, counts)), column_headers=["Angle", "Counts"])
    source["Lambda"] = wavelength
    scan, peaks, measured, uncertainty = load_script("Kiessig.py")["analyse"](
        source, critical_edge=0.1, sensitivity=50)
    assert measured == pytest.approx(thickness, rel=0.05)
    assert np.isfinite(uncertainty) and uncertainty >= 0
    assert len(peaks) >= 3 and len(scan.plot.figure.axes) == 2
    assert source.shape[1] == 2


def test_bnl_splitter_real_scan_and_single_point(tmp_path):
    split = load_script("BNL_FileSplitter.py")["split_file"]
    original = (SAMPLES / "BNL-data.txt").read_text()
    single = original.replace("#S 3 ", "#S 8 ")
    single = "\n".join(line for line in single.splitlines() if line.startswith("#") or line.strip().startswith("700.00"))
    source = tmp_path / "combined.txt"
    source.write_text("#C ignored\n" + original + "\n" + single)
    original_directory = Path.cwd()
    paths = split(source)
    assert [path.name for path in paths] == ["3.bnl", "8.bnl"]
    assert len(Data(paths[1], filetype="BNLFile")) == 1
    assert "#C" not in paths[0].read_text()
    assert Path.cwd() == original_directory


def test_image_mask_example_without_interaction():
    image, binary = load_script("Mask_selection.py")["demonstrate"](select=False)
    assert image.shape == binary.shape
    assert set(np.unique(binary.to_numpy())) <= {-1, 1}
    assert np.ma.getmaskarray(image.to_numpy()).shape == image.shape


def pcar_config(tmp_path, source="PCAR-New.ini"):
    """Copy a real fitting configuration while disabling user-facing output."""
    config = configparser.ConfigParser()
    config.read(SCRIPTS / source)
    for option in ("show_plot", "save_fit", "print_report", "decompose"):
        config.set("Options", option, "False")
    filename = tmp_path / source
    with filename.open("w") as stream:
        config.write(stream)
    return config, filename


def test_pcar_real_data_fit_and_plot(tmp_path):
    _, filename = pcar_config(tmp_path)
    data = load_script("PCAR-New.py")["working"]().load_config(filename, SAMPLES / "PCAR Co Data.csv")
    fit = data.Fit()
    assert fit.success and np.isfinite(fit.chisqr)
    data.plot_results()
    assert len(data.plot.ax.lines) == 2 and data.plot.ax.texts


@pytest.mark.parametrize("source", ["PCAR-New.ini", "PCAR-chi^2.ini"])
@pytest.mark.xfail(strict=True, raises=(KeyError, ValueError, IndexError),
                   reason="INI CSV layout forwarding is deferred: Stoner-PythonCode#41")
def test_pcar_ini_controls_nondefault_csv_layout(tmp_path, source):
    config, filename = pcar_config(tmp_path, source)
    csvfile = tmp_path / "measurement.csv"
    csvfile.write_text("Preamble\nSecond preamble\nV|G|dG\nIntervening record\n-1;2;0.1\n0;3;0.2\n1;4;0.3\n")
    for key, value in {"header_line": "2", "data_line": "4", "header_delim": "|", "data_delim": ";",
                       "e": "dG"}.items():
        config.set("Data", key, value)
    with filename.open("w") as stream:
        config.write(stream)
    data = load_script("PCAR-New.py")["working"]().load_config(filename, csvfile)
    assert data.column_headers == ["V", "G", "dG"]
    np.testing.assert_allclose(data.to_numpy(), [[-1, 2, 0.1], [0, 3, 0.2], [1, 4, 0.3]])
    assert list(data.setas) == list("xye")


@pytest.mark.xfail(strict=True, raises=(KeyError, ValueError, IndexError),
                   reason="INI CSV layout forwarding is deferred: Stoner-PythonCode#41")
def test_ini_csv_headerless_import_and_delimiter_detection(tmp_path):
    from Stoner.analysis.fitting.models import cfg_data_from_ini

    csvfile = tmp_path / "headerless.csv"
    csvfile.write_text("1;2\n3;4\n5;6\n")
    config = tmp_path / "layout.ini"
    config.write_text(f"[Data]\nfilename: {csvfile}\ntype: CSVFile\nx: 0\ny: 1\n"
                      "header_line: None\ndata_line: 0\nheader_delim: None\ndata_delim: None\n")
    data = cfg_data_from_ini(str(config))
    np.testing.assert_array_equal(data.to_numpy(), [[1, 2], [3, 4], [5, 6]])
    assert list(data.setas) == list("xy")


def test_pcar_rescale_discard_and_selection_preserve_mask(tmp_path):
    config, filename = pcar_config(tmp_path)
    data = load_script("PCAR-New.py")["working"]().load_config(filename, SAMPLES / "PCAR Co Data.csv")
    data.config = config
    data.mask[int(np.argmin(abs(data.column(data.vcol)))), data.gcol] = True
    initial = data.to_numpy()
    config.set("Options", "rescale_v", "True")
    config.set("Data", "v_scale", "2")
    config.set("Data", "discard", "True")
    config.set("Data", "v_limit", "10")
    data.RescaleV().Discard()
    keep = abs(initial.data[:, data.vcol] * 2) <= 10
    np.testing.assert_allclose(data.column(data.vcol), initial[keep, data.vcol] * 2)
    np.testing.assert_array_equal(data.mask, initial.mask[keep])


def test_pcar_chi_squared_panels(tmp_path):
    config, filename = pcar_config(tmp_path, "PCAR-chi^2.ini")
    config.set("P", "min", "0.4")
    config.set("P", "max", "0.45")
    config.set("P", "step", "0.025")
    config.set("Options", "show_plot", "True")
    with filename.open("w") as stream:
        config.write(stream)
    data = load_script("PCAR-New.py")["working"]().load_config(filename, SAMPLES / "PCAR Co Data.csv")
    fit = data.Fit()
    assert len(fit) == 3
    assert len(fit.plot.figure.axes) == len(fit.setas.cols.ycol)
    assert fit.plot.figure.axes[-1].get_yscale() == "log"


def test_pcar_default_decomposition_and_fancy_normalisation(tmp_path):
    config, filename = pcar_config(tmp_path)
    cls = load_script("PCAR-New.py")["working"]
    data = cls().load_config(filename, SAMPLES / "PCAR Co Data.csv")
    config.set("Options", "decompose", "True")
    data.config = config
    data.Decompose()
    assert len(data) and np.all(np.isfinite(data.to_numpy(masked=False)))
    voltage = np.linspace(-10, 10, 101)
    background = 1 + 0.01 * voltage**2
    data = cls(np.column_stack((voltage, background)), setas="xy")
    data.config, data.vcol, data.gcol = config, 0, 1
    config.set("Options", "normalise", "True")
    config.set("Options", "fancy_normaliser", "True")
    data.mask[50, 1] = True
    data.Normalise()
    np.testing.assert_allclose(data.y.compressed(), 1, atol=1e-10)
    assert data.mask[50, 1]


def test_pcar_down_sweep_retains_selected_exclusions():
    down_only = load_script("PCAR-New.py")["down_only"]
    voltage = np.r_[np.linspace(-1, 1, 30), np.linspace(1, -1, 30)[1:], np.linspace(-1, 1, 30)[1:]]
    data = Data(np.column_stack((voltage, np.arange(len(voltage)))), column_headers=["Set V", "Signal"])
    data.mask[40, 1] = True
    down_only(data)
    assert len(data) >= 25 and np.all(np.diff(data.column(0)) <= 0)
    assert np.ma.getmaskarray(data.column(1))[list(data.to_numpy(masked=False)[:, 1]).index(40)]


def test_vsm_v2_edits_preserve_exclusions_and_headers(monkeypatch):
    script = load_script("VSManalysis_v2.py")
    data = Data(SAMPLES / "VSM-Data.fld")
    headers = list(data.column_headers)
    data.mask[10, 2] = True
    data.add_column(2 * abs(data.column("X")), header="R (V)")
    assert script["make_true_m"](data) is data and data.mask[10, 2]
    headers.append("R (V)")
    original_mask = data.mask.copy()
    monkeypatch.setattr("builtins.input", lambda *args: "5")
    assert script["edit_data"](data, [2, 3, 4, 5]) is data
    np.testing.assert_array_equal(data.mask, original_mask)
    assert data.column_headers == headers
    assert np.all(np.isfinite(data.column(2)[2:].compressed()))


def test_vsm_v3_real_corrections_preserve_exclusions():
    cls = load_script("VSManalysis_v3.py")["VSMAnalysis"]
    data = cls(SAMPLES / "VSM-Data.fld")
    data.mask[10, 2] = True
    original_mask = data.mask.copy()
    assert data.true_m().correct_drift().remove_diamagnetism_and_offset() is data
    assert np.all(data.mask[original_mask])
    # The real fixture starts with two NaN rows; invalid calculated moments are excluded.
    assert np.all(data.mask[:2, 2:5])
    corrected_mask = data.mask.copy()
    for measure in (data.find_Hc, data.find_Br):
        assert len(measure()) == 2
        np.testing.assert_array_equal(data.mask, corrected_mask)


def test_vsm_v3_background_correction_has_expected_numerical_result():
    cls = load_script("VSManalysis_v3.py")["VSMAnalysis"]
    field = np.r_[np.linspace(0, 1, 51), np.linspace(1, -1, 101)[1:], np.linspace(-1, 1, 101)[1:]]
    moment = np.sign(field) + 0.12 * field + 0.3
    values = np.column_stack((np.arange(len(field)), field, moment, moment * 2, moment * 3, moment * 4, moment * 5))
    data = cls(values, column_headers=["Time", "H_vsm", "m (emu)", "Mvol", "Mmass", "X", "Y"])
    data.mask[100, 2] = True
    data.remove_diamagnetism_and_offset()
    np.testing.assert_allclose(data.column(2).compressed(), np.delete(np.sign(field), 100), atol=1e-10)
    assert data.mask[100, 2]


def test_xmcd_group_reduction_and_metadata(tmp_path):
    script = load_script("XMCD_Reduction.py")
    energy = np.linspace(610, 680, 141)
    folder = DataFolder(readlist=False)
    for index, helicity in enumerate(("pos", "neg")):
        signal = 0.2 * energy + 5 + (2 + index) * (energy >= 640)
        data = Data(np.column_stack((energy, signal)), column_headers=[f"Energy {helicity}", "fluo"])
        data["run"], data["sensor_temp"], data["magj1yins"] = 49173 + index, 10, -8.5
        folder.append(data)
    result = script["reduce"](folder, output_directory=tmp_path)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].column(1), 0, atol=1e-12)
    assert result[0]["Position"] == "Perp"
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert len(folder) == 2 and not folder.groups


def test_genx_pnr_plot_uses_measured_down_spin_signal():
    plot_exports = load_script("GenX/GenX PNR Asymmetry Plotting.py")["plot_exports"]
    folder = DataFolder(readlist=False)
    q = np.linspace(0.01, 0.2, 20)
    for index, values in enumerate((np.column_stack((q, q + 1, q - 1)),
                                    np.column_stack((q, q + 4, q + 3, q * 0 + 0.2)),
                                    np.column_stack((q, q + 6, q + 5, q * 0 + 0.1)))):
        data = Data(values)
        data["dataset"] = index
        if index == 0:
            data.column_headers = ["z", "real sld +", "real sld -"]
        folder.append(data)
    data, sld = plot_exports(folder)
    reflectivity = plt.figure(plt.get_fignums()[0]).axes[0]
    np.testing.assert_allclose(reflectivity.lines[3].get_ydata(), q + 3)
    np.testing.assert_allclose(data.column("SpinAsym_data"), 2 / (2 * q + 8))
    assert reflectivity.get_yscale() == "log" and sld.shape[1] == 5
    assert folder[2].shape[1] == 4


@pytest.fixture
def genx_plugin(monkeypatch):
    """Supply only the external GenX host boundary; use real Stoner objects below."""
    class Template:
        def __init__(self, parent):
            self.parent = parent

    modules = {
        "wx": types.SimpleNamespace(Dialog=object),
        "genx": types.ModuleType("genx"),
        "genx.plugins": types.ModuleType("genx.plugins"),
        "genx.plugins.data_loader_framework": types.SimpleNamespace(Template=Template),
        "genx.plugins.utils": types.SimpleNamespace(ShowWarningDialog=lambda *args: pytest.fail(str(args))),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return load_script("GenX/stoner.py")


def test_genx_loader_real_xray_and_repeated_column_settings(genx_plugin):
    plugin = genx_plugin["Plugin"](None)
    commands = []
    dataset = types.SimpleNamespace(run_command=lambda: commands.append(True))
    for _ in range(2):
        plugin.LoadData(dataset, str(SAMPLES / "New-XRay-Data.dql"), data_id=0)
    assert plugin.x_col == "Angle" and plugin.y_col == "Counts"
    assert len(dataset.x_raw) > 1 and len(commands) == 2


def test_genx_loader_keeps_exclusions_aligned(genx_plugin, monkeypatch):
    data = Data(np.array([[1., 2., 0.1], [2., 4., 0.2], [3., np.nan, 0.3], [4., 8., 0.4]]), setas="xye")
    data.mask[1, 2] = True
    monkeypatch.setitem(genx_plugin["Plugin"].LoadData.__globals__, "Data", lambda *args, **kwargs: data)
    plugin = genx_plugin["Plugin"](None)
    dataset = types.SimpleNamespace(run_command=lambda: None)
    plugin.LoadData(dataset, "test")
    np.testing.assert_array_equal(dataset.x_raw, [1, 4])
    np.testing.assert_array_equal(dataset.y_raw, [2, 8])
    np.testing.assert_array_equal(dataset.error_raw, [0.1, 0.4])


def test_genx_loader_unassigned_columns_do_not_conflict_with_error_role(genx_plugin, monkeypatch):
    data = Data(np.array([[1., 2.], [2., 4.]]), column_headers=["Angle", "Counts"])
    monkeypatch.setitem(genx_plugin["Plugin"].LoadData.__globals__, "Data", lambda *args, **kwargs: data)
    plugin = genx_plugin["Plugin"](None)
    dataset = types.SimpleNamespace(run_command=lambda: None)
    plugin.LoadData(dataset, "test")
    np.testing.assert_array_equal(dataset.x_raw, [1, 2])
    np.testing.assert_array_equal(dataset.y_raw, [2, 4])
    np.testing.assert_array_equal(dataset.error_raw, [2, 4])
