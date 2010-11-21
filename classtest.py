import Stoner

#Create object
theData=Stoner.DataFile()
theData=Stoner.DataFile('data.txt')
theData=Stoner.DataFile(array([[2, 3][1, 2]]))
theData=Stoner.DataFile(dict('on':1, 'two', 2))

#import data
theData.get_data('datafile.txt')

# Print the Data
print theData.data

# Print Meta data
print theData.metadata

# Print Column Headers
print theData.column_headers

# Return a metadata value
print theData.meta("User") # Returns an exact match
print theData.meta("multi\[1\]") # Returns all matches to the regular expression
print theData .metadata_value('multi[1]:Control: Gate samples') # Included for backwards compatibility but don't include the type information

# Return a single column of data
print theData.column(0) # Match by numberidal index
print theData.column("Temperature") #match by exact name
print theData.clumn('Voltage.*') # match by regular expression (first column that matches only)

#Return rows of data
print theData.search('Temp',lambda x: x>5 and x<10,['Temp','Resis'])    # First argument can be an integer,, string or regular expression
                                                                                                                            # Second argument can be a value or function that returns true/false
                                                                                                                            # Third arguement is optional is a list of integers, strings or regular expressions
print theData.search(0, lambda x: True,  [1, 2, 0]) # Just re-orders the columns
