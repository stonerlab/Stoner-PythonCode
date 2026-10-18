"""Experimental Stonerlab DataFile.

Hacked up version of the default data loader plugin that uses the Stoner package classes to
read in data files. Supports loading from the Brucker D8
"""

# pylint: disable=invalid-name
import numpy as np
import wx
from genx.plugins.data_loader_framework import Template
from genx.plugins.utils import ShowWarningDialog

from Stoner import Data
from Stoner.core.exceptions import StonerUnrecognisedFormat


class Plugin(Template):
    """Plugin class from GenX."""

    def __init__(self, parent):
        """Initialise plugin and column assignments."""
        Template.__init__(self, parent)
        self.x_col = "Angle"
        self.y_col = "Counts"
        self.e_col = "Counts"

    def LoadData(self, dataset, filename, data_id=0):
        """Load one measurement into the dataset supplied by the GenX loader framework.

        Args:
            dataset (genx.data.DataSet):
                Destination dataset. Raw X, Y and error arrays are replaced together.
            filename (str or pathlib.Path):
                Measurement to load with Stoner's registered loaders.
            data_id (int):
                Dataset number for multi-dataset sources. This loader reads one
                table per file, so the default zero is the supported selection.

        Notes:
            Assigned Stoner roles take precedence over the configured column
            names. Excluded and non-finite triples are omitted together. GenX's
            framework handles list updates and GUI events after this call.
        """
        if data_id != 0:
            raise ValueError("The Stoner loader reads one dataset per file")
        try:
            datafile = Data(filename)
        except (StonerUnrecognisedFormat, ValueError, OSError) as e:
            ShowWarningDialog(
                self.parent,
                f"""Could not load the file: {filename}\nPlease check the format.

                Stoner.Data gave the following error:\n{e}""",
            )
        else:
            # For the freak case of only one data point
            try:
                columns = datafile.setas.cols
                if "x" not in datafile.setas or "y" not in datafile.setas:
                    x = datafile.column(self.x_col)
                    y = datafile.column(self.y_col)
                    error = datafile.column(self.e_col)
                else:
                    x = datafile.column(columns["xcol"])
                    y = datafile.column(columns["ycol"][0])
                    error = datafile.column(columns["yerr"][0]) if columns["yerr"] else np.ones(len(datafile))
            except Exception as e:
                ShowWarningDialog(
                    self.parent,
                    f"The data file does not contain all the columns specified in the opions\n{e}",
                )
                # Okay now we have showed a dialog lets bail out ...
                return
            # The data is set by the default Template.__init__ function, neat hu
            # Know the loaded data goes into *_raw so that they are not
            # changed by the transforms
            # GenX accepts ordinary arrays: remove excluded or non-finite triples together.
            values = np.ma.column_stack((x, y, error))
            valid = ~np.any(np.ma.getmaskarray(values) | ~np.isfinite(values.data), axis=1)
            dataset.x_raw = values.data[valid, 0].copy()
            dataset.y_raw = np.where(values.data[valid, 1] == 0, 1e-8, values.data[valid, 1])
            dataset.error_raw = values.data[valid, 2].copy()
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

    def SettingsDialog(self):
        """Implement a dialog box that allows the user set import settings for example."""
        col_values = {
            "x": self.x_col,
            "y": self.y_col,
            "y error": self.e_col,
        }
        dlg = SettingsDialog(self.parent, col_values)
        if dlg.ShowModal() == wx.ID_OK:
            col_values = dlg.GetColumnValues()
            self.y_col = col_values["y"]
            self.x_col = col_values["x"]
            self.e_col = col_values["y error"]
        dlg.Destroy()


class SettingsDialog(wx.Dialog):
    """Plugin Settings dialog class."""

    def __init__(self, parent, col_values):
        """Housekeeping."""
        wx.Dialog.__init__(self, parent, -1, "Data loader settings")

        box_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Make the box for putting in the columns
        col_box = wx.StaticBox(self, -1, "Columns")
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL)

        col_grid = wx.GridBagSizer(len(col_values), 2)
        self.col_controls = col_values.copy()
        keys = sorted(col_values)
        for i, name in enumerate(keys):
            text = wx.StaticText(self, -1, name + ": ")
            control = wx.TextCtrl(
                self, value=str(col_values[name]), style=wx.EXPAND
            )
            col_grid.Add(
                text,
                (i, 0),
                flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                border=5,
            )
            col_grid.Add(
                control,
                (i, 1),
                flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                border=5,
            )
            self.col_controls[name] = control

        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALIGN_CENTRE | wx.ALL | wx.EXPAND, 5)

        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(box_sizer, 1, wx.GROW | wx.ALIGN_CENTER_HORIZONTAL, 20)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.ALIGN_CENTER_HORIZONTAL, 30)

        sizer.Add(button_sizer, 0, flag=wx.ALIGN_RIGHT, border=20)
        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()

    def GetColumnValues(self):
        """Read the data column headings."""
        values = {}
        for key in self.col_controls:
            values[key] = self.col_controls[key].GetValue()
        return values
