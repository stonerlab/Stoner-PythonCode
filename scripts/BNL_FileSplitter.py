"""Created 08/2011 Rowan Temple
Splits Brookhaven files up into seperate scan files and dumps them in a folder 'BNLSplitFiles'
You must be working in the directory of the file to run the program.
"""


import os
import re

#Get file open ###############
while(True):
    try:
        directory=raw_input("Enter the directory path where your data is stored\n")
        os.chdir(directory)
        filename=raw_input("Enter the filename.extension for your file\r\n")
        mainFP=open(filename, 'r')
        break
    except(IOError):
        print ("Oops I couldn't find that file, are you in the right directory?")
os.mkdir('BNLSplitFiles')
os.chdir('BNLSplitFiles')

#Main algorithm ###########
    
writeName=re.split(r'[.]',filename)
writeFP=open(writeName[0]+'0.txt','w') #title sequence goes in this file
counter=1   #this will label the files
for line in mainFP:
    if line[0:2]=='#S':
        if int(line.split()[1])!=counter:
            raise ValueError                #check for inconsistencies with filenames and scan numbers
        writeFP.close()
        writeFP=open(writeName[0]+str(counter)+'.txt','w')
        counter+=1
    if line[0:2]!='#C':
      writeFP.write(line)
    """ignore #C statements which are usually abort and rarely useful, they come
    after data and before the next #S"""
writeFP.close()
mainFP.close()
print 'Done.'     
