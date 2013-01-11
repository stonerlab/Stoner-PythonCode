''' <h1>Experimental Stonerlab DataFile</h1>
Hacked up version of the default data loader plugin that uses the Stoner package classes to
read in data files. Supports loading from the Brucker D8
'''

import numpy as np
import wx
import Stoner as SC
import Stoner.FileFormats as SF

from wx.lib.masked import NumCtrl

from plugins.data_loader_framework import Template
from plugins.utils import ShowErrorDialog, ShowWarningDialog, ShowInfoDialog

class Plugin(Template):
    def __init__(self, parent):
        Template.__init__(self, parent)
        self.x_col = 'Angle'
        self.y_col = 'Counts'
        self.e_col = 'Counts'
    
    def LoadData(self, data_item_number, filename):
        '''LoadData(self, data_item_number, filename) --> none
        
        Loads the data from filename into the data_item_number.
        '''
        try:
            datafile=SC.DataFile(str(filename)) # does all the hard work here
        except Exception, e:
            ShowWarningDialog(self.parent, 'Could not load the file: ' +\
                    filename + ' \nPlease check the format.\n\n Stoner.DataFile'\
                    + ' gave the following error:\n'  +  str(e))
        else:
            # For the freak case of only one data point
            try:
                self.x_col=datafile.find_col(self.x_col)
                self.y_col=datafile.find_col(self.y_col)
                self.e_col=datafile.find_col(self.e_col)
            except KeyError:
                ShowWarningDialog(self.parent, 'The data file does not contain'\
                        + 'all the columns specified in the opions\n')
                # Okay now we have showed a dialog lets bail out ...
                return
            # The data is set by the default Template.__init__ function, neat hu
            # Know the loaded data goes into *_raw so that they are not
            # changed by the transforms
            datafile=SC.AnalyseFile(datafile)
            datafile=datafile.apply(lambda x:x1E-8 if [self.y_col]==0.0 else x[self.y_col], self.y_col, header=datafile.column_headers[self.y_col]) # clear out zeroes
            self.data[data_item_number].x_raw = datafile.column(self.x_col)
            self.data[data_item_number].y_raw =  datafile.column(self.y_col)
            self.data[data_item_number].error_raw =  datafile.column(self.e_col)
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            self.data[data_item_number].run_command()
            
            # Send an update that new data has been loaded
            self.SendUpdateDataEvent()
        
    def SettingsDialog(self):
        '''SettingsDialog(self) --> None
        
        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        '''
        col_values = {'x': self.x_col,'y': self.y_col,'y error': self.e_col}
        dlg = SettingsDialog(self.parent, col_values)
        if dlg.ShowModal() == wx.ID_OK:
            col_values = dlg.GetColumnValues()
            self.y_col = col_values['y']
            self.x_col = col_values['x']
            self.e_col = col_values['y error']
        dlg.Destroy()
        
        
class SettingsDialog(wx.Dialog):
    
    def __init__(self, parent, col_values):
        wx.Dialog.__init__(self, parent, -1, 'Data loader settings')
        
        box_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Make the box for putting in the columns
        col_box = wx.StaticBox(self, -1, "Columns" )
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL )
        
        
        col_grid = wx.GridBagSizer(len(col_values), 2)
        self.col_controls = col_values.copy()
        keys = col_values.keys()
        keys.sort()
        for i, name in enumerate(keys):
            text = wx.StaticText(self, -1, name+': ')

            control = wx.TextCtrl(self, value = str(col_values[name]),\
                style = wx.EXPAND)
            col_grid.Add(text, (i,0),\
                flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                border = 5)
            col_grid.Add(control, (i,1),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
            self.col_controls[name] = control
        
        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALIGN_CENTRE|wx.ALL|wx.EXPAND, 5)
        
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(box_sizer, 1, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL, 20)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL, 30)
        
        sizer.Add(button_sizer,0,\
                flag = wx.ALIGN_RIGHT, border = 20)
        self.SetSizer(sizer)
        
        sizer.Fit(self)
        self.Layout()
        
    def GetColumnValues(self):
        values = {}
        for key in self.col_controls:
            values[key] = self.col_controls[key].GetValue()
        return values
    
    def GetMiscValues(self):
        values = {}
        for key in self.misc_controls:
            val = self.misc_controls[key].GetValue()
            if (type(val) == type(u'') or type(val) == type('')):
                if val.lower() == 'none':
                    val = None
            values[key] = val
        return values
