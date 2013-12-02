# Test Suite for Stoner package

# $Log: test.py,v $

# Revision 1.2  2010/12/25 22:28:21  cvs

# Update to match new syntax and functionaility

#



import Stoner

from scipy import *



# Basic Tests of DataFile functionaluity

file=Stoner.DataFile()

file.load('../sample-data/TDI_Format_RT.txt')



print file.column_headers

print file.data

print file.metadata

print file.typehint

print file



print len(file)

for col in file.columns():

    print "Mean:"+str(mean(col))+" Max:"+str(max(col))+" Min:"+str(min(col))



# Tests of PlotFile functionality

plot=Stoner.PlotFile(file)

plot.plot_xy('Temp', 'Res')



# Tests of AnalysisFile Functionality





