#----------------------------------------------------------------------------- 
#	FILE:		STONER DATA CLASS (version 0.1)
#	AUTHOR:		MATTHEW NEWMAN
#	DATE:		24/11/2010
#-----------------------------------------------------------------------------
#

import csv
#import sys
import re
#import string
#from scipy import *
import scipy
#import pdb # for debugging
import os
import sys
import numpy
import pylab


class DataFolder:
	
	#	CONSTANTS
	
	#	INITIALISATION
	
	def __init__(self, foldername):
		self.data = [];
		self.metadata = dict();
		self.foldername = foldername;
		self.__parseFolder();
		
	#	PRIVATE FUNCTIONS
	
	def __parseFolder(self):
		path="C:/Documents and Settings/pymn/workspace/Stonerlab/src/folder/run1"  # insert the path to the directory of interest
		dirList=os.listdir(path)
		for fname in dirList:
			print(fname)

#	PUBLIC METHODS

		
class DataFile:

#	CONSTANTS

	regexGetType = re.compile(r'{(.*?)\}') # Match word between { }
	typeInteger = "{I32}{U16}"
	typeFloat = "{Double Float}"
	typeBoolean = "{Boolean}"
	defaultDumpLocation='C:\\dump.csv'
	
#	INITIALISATION

	def __init__(self):
		self.data = []
		self.metadata = dict()
		self.filename = ''

#	PRIVATE FUNCTIONS

	def __parse_data(self):
		bFirstRow = True;
		reader = csv.reader(open(self.filename, "rb"), delimiter='\t', quoting=csv.QUOTE_NONE)
		
		for row in reader:
			if bFirstRow == True:
				self.column_headers = row[1:len(row)]
				bFirstRow = False
			else:
				if self.__contains(row[0], '=') == True:
					self.metadata[row[0].split('=')[0]] = row[0].split('=')[1]
				if (len(row[1:len(row)]) > 1) or len(row[1]) > 0:
					self.data.append(row[1:len(row)])

		i = 0;
		while i < len(self.data): # Converts data to floating point numbers
			j = 0
			while j < len(self.data[i]):
				self.data[i][j] = float(self.data[i][j])
				j += 1
			i += 1
			
		self.data = scipy.array(self.data)
			
	def __contains(self, theString, theQueryValue):
		return theString.find(theQueryValue) > -1

#	PUBLIC METHODS

	def get_data(self,filename):
		self.filename = filename;
		self.__parse_data();
		
	def metadata_value(self, text):
		value = self.metadata[text]
		text = self.regexGetType.findall(text)[0]
		if self.__contains(self.typeInteger, text) == True:
			value = int(value);
		elif self.__contains(self.typeFloat, text) == True:
			value = float(value);
		elif self.__contains(self.typeBoolean, text) == True:
			value = bool(value);
		else:
			value = str(value);
		return value

	def data(self):
		return self.data

	def metadata(self):
		return self.metadata

	def column_headers(self):
		return self.column_headers
	
	def do_polyfit(self,column_x,column_y,polynomial_order):
		x_data = self.data[:,column_x]
		y_data = self.data[:,column_y]
		return numpy.polyfit(x_data,y_data,polynomial_order)
	
	def add_column(self,column_data,column_header=''):
		self.column_headers.append(column_header)
		
		# The following 2 lines make the array we are adding a
		# [1, x] array, i.e. a column by first making it 2d and
		# then transposing it.
		
		column_data=numpy.atleast_2d(column_data)
		column_data=column_data.T # Transposes Array
		self.data=numpy.append(self.data,column_data,1)
		print('Column header "'+column_header+'" added to array')
		return True
	
	def plot_simple_xy(self,column_x, column_y,title='',save_filename='',
					show_plot=True):
			x=self.data[:,column_x]
			y=self.data[:,column_y]
			if show_plot == True:
				pylab.ion()
			pylab.plot(x,y)
			pylab.draw()
			pylab.xlabel(str(self.column_headers[column_x]))
			pylab.ylabel(str(self.column_headers[column_y]))
			pylab.title(title)
			pylab.grid(True)
			if save_filename != '':
				pylab.savefig(str(save_filename))
			
	def csvArray(self,dump_location=defaultDumpLocation):
		spamWriter = csv.writer(open(dump_location, 'wb'), delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		i=0
		spamWriter.writerow(self.column_headers)
		while i< self.data.shape[0]:
			spamWriter.writerow(self.data[i,:])
			i+=1
