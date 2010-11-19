import Stoner

#Create object
theData=Stoner.DataFile()

#import data
theData.get_data('datafile.txt')

# Print the Data
print(theData.data)
# Print Meta data
print(theData.metadata)
# Print Column Headers
print(theData.column_headers)
# Return a metadata value
print(theData   .metadata_value('multi[1]:Control: Gate samples{I32}'))