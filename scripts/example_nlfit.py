import Stoner, os



inifile='C:\\PythonCode\\scripts\\nlfit_ex.ini'

#data, ax = Stoner.nlfit.nlfit(inifile, 'quadratic')

b=Stoner.AnalyseFile()

b.load('C:\\PythonCode\\scripts\\example_nlfit_data.txt') #remember not putting a file in here will bring up the open file dialogue.

data, ax = b.nlfit(inifile, 'quadratic')
