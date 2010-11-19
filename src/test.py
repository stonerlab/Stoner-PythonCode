import Stoner

# Test Imports
import numpy
import pylab
import tkFileDialog
import time

file=Stoner.DataFile()
file.get_data('data.txt')

print(file.do_polyfit(1,2,1))

'''
i=0
while i < 6:
    time.sleep(1)
    file.plot_simple_xy(1, 2,'','')
    i+=1
    print(i)
   
pylab.ioff()
'''
# examples:
def open_directory():
    filename = tkFileDialog.askdirectory()
    print filename  # test
'''  
def save_it():
    filename = tkFileDialog.askopenfilename()
    print filename  # test
    
def save_as():
    filename = tkFileDialog.asksaveasfilename()
    print filename  # test
''' 
#open_directory()

