#-----------------------------------------------------------------------------

#   FILE:       VSM ANALYSIS PROG (version 0.1)

#   AUTHOR:     CHRIS ALLEN

#   DATE:       10/12/2010

#-----------------------------------------------------------------------------

#



import Stoner

import os

import easygui

import numpy as np

import matplotlib as mp

from matplotlib import pylab as plt



aFile=easygui.fileopenbox()





theData=Stoner.DataFile()

theData.loadVSM(aFile)



plt.ion()

aFig=plt.figure(figsize=(10,6),facecolor='w')

aAx=plt.axes([0.125,0.1,0.6,0.8])

aAx.plot(theData.data[:,1],theData.data[:,2],'k-')

plt.xlabel(str(theData.column_headers[1])+' (T)')

plt.ylabel(str(theData.column_headers[2])+' (emu)')

title=(os.path.basename(theData.filename))

plt.title(title)



def pressSub(event):

     tempData=np.nan_to_num(theData.data)

     maxHarg=np.argmax(tempData[:,1])

     minHarg=np.argmin(tempData[:,1])



     '''linear fit to saturated data (+- 25 points from max/min field)

     '''

     highFit=np.polyfit(theData.data[maxHarg-25:maxHarg+25,1],theData.data[maxHarg-25:maxHarg+25,2],1)

     lowFit=np.polyfit(theData.data[minHarg-25:minHarg+25,1],theData.data[minHarg-25:minHarg+25,2],1)



     '''  Average grad

     '''

     fitGrad=(highFit[0]+lowFit[0])/2





     '''  Delete linear grad from all data - put in next column

      '''

     theData.add_column(theData.data[:,2]-(fitGrad*theData.data[:,1]),'m_corr')

     aAx.plot(theData.data[:,1],theData.data[:,7],'r-')



def pressSat(event):

     tempData=np.nan_to_num(theData.data)

     maxHarg=np.argmax(tempData[:,1])

     minHarg=np.argmin(tempData[:,1])



     '''linear fit to saturated data (+- 25 points from max/min field)

     '''

     highFit=np.polyfit(theData.data[maxHarg-25:maxHarg+25,1],theData.data[maxHarg-25:maxHarg+25,2],1)

     lowFit=np.polyfit(theData.data[minHarg-25:minHarg+25,1],theData.data[minHarg-25:minHarg+25,2],1)



     '''  Average intercept

     '''

     fitSat=(highFit[1]-lowFit[1])/2





     ''' display ms

      '''

     txtSat='ms = '+str(fitSat)

     plt.figtext(0.75, 0.4,txtSat)

     theData.fitSat=fitSat



axSub=plt.axes([0.75, 0.8, 0.18, 0.075])

bSub=plt.Button(axSub,'Subtract linear',color='w',hovercolor='r')

bSub.on_clicked(pressSub)



axSat=plt.axes([0.75, 0.6, 0.18, 0.075])

bSat=plt.Button(axSat,'Extract ms',color='w',hovercolor='r')

bSat.on_clicked(pressSat)





