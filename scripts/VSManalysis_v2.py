"""VSM Analysis Code example v2.

Author: Rowan Temple.     Date:11/2011

***YOU CANNOT RUN THIS FILE WITHIN IDLE! You must run it in a command window. (matplotlib interactive
thing gets mixed up from within IDLE.)***

Editing VSM files (taking off drift, setting middle to zero etc.)
When entering path names you need the folder structure up to the point where this program is.
I think there will be difficulties if this program is run in ipython to do with graph not displaying,
easily fixed

v2 corrections: less automation saving time
                graph is used in interactive mode
                shift algorithm is different to make for better zeroing

"""

# pylint: disable=invalid-name, redefined-outer-name
import os

import numpy as np
import matplotlib.pyplot as plot

import Stoner

plot.ion()  # put the plotting in interactive mode
raw_input = input
print("Analysis program for VSM files\n\n\n")
print("Please wait while program loads...\n")
# VSMcalibration=0.86224   #m to X ratio, determined by reference Py sample


def deleteCorruptLines(data):
    """Take data array and returns data with lines with 6:0 or --- in them deleted."""
    delCount = 0
    for line in data:
        if line.find("6:0") != -1:
            data.pop(data.index(line))
            delCount += 1
        if line.find("---") != -1:
            data.pop(data.index(line))
            delCount += 1
    if delCount > 6:
        input(
            "I've detected a lot of bad data in your file you should check it"
        )
    return data


def driftEliminator(data, N):
    """Take off a linear drift of m with time from the data."""
    maxHarg = int(np.argmax(data[: len(data[:, 0]) / 2, 1]))
    finalHarg = int(len(data[:, 1]) - 1)
    # linear fit to saturated data (+- N points from max field and final field)
    finalFit = np.polyfit(
        data[finalHarg - N : finalHarg, 1],
        data[finalHarg - N : finalHarg, 2],
        1,
    )
    firstFit = np.polyfit(
        data[maxHarg - N : maxHarg + N, 1],
        data[maxHarg - N : maxHarg + N, 2],
        1,
    )
    pt = np.zeros((2, 2))
    pt[0] = [
        data[maxHarg, 0],
        firstFit[0] * data[maxHarg, 1] + firstFit[1],
    ]  # (time,m) at max field
    pt[1] = [data[finalHarg, 0], finalFit[0] * data[finalHarg, 1] + finalFit[1]]
    # Delete drift
    for i in range(len(data[:, 0])):
        data[i, 2] = data[i, 2] - (pt[1, 1] - pt[0, 1]) / (
            pt[1, 0] - pt[0, 0]
        ) * (data[i, 0] - pt[0, 0])
    return data


def shift(data, N):
    """Translate curve in y so that halfway between the saturated fieldsis zero.

    (uses the average y value of all the saturated points given to determine upper and lower bounds of curve
    """
    maxHarg = int(
        np.argmax(data[: len(data[:, 0]) / 2, 1])
    )  # gives the data index of maximum field (in the first half of the data
    minHarg = int(np.argmin(data[:, 1]))
    # average saturated data (+- N points from max/min field)
    highAve = np.average(data[(maxHarg - N) : (maxHarg + N), 2])
    lowAve = np.average(data[minHarg - N : minHarg + N, 2])
    data[:, 2] = data[:, 2] - (highAve + lowAve) / 2.0
    return data


def diamagBackgroundRem(data, N):
    """Remove a diamagnetic background using a linear fit to the N data points surrounding the max and min field."""
    maxHarg = int(np.argmax(data[: len(data[:, 0]) / 2, 1]))
    minHarg = int(np.argmin(data[:, 1]))

    # linear fit to saturated data (+- N points from max/min field)
    highFit = np.polyfit(
        data[maxHarg - N : maxHarg + N, 1],
        data[maxHarg - N : maxHarg + N, 2],
        1,
    )
    lowFit = np.polyfit(
        data[minHarg - N : minHarg + N, 1],
        data[minHarg - N : minHarg + N, 2],
        1,
    )
    # Average grad
    fitGrad = (highFit[0] + lowFit[0]) / 2
    # Delete linear grad from all data
    data[:, 2] = data[:, 2] - (fitGrad * data[:, 1])
    return data


def invert(Data):
    """Flip data in y axis."""
    for i in range(len(Data.data[:, "0"])):
        Data.data[i, Data.find_col("m (emu)")] = -Data.data[
            i, Data.find_col("m (emu)")
        ]
    return Data


def makeTruem(Data):
    """VSM takes m from lock in X, this takes m from lock in R, useful if theta!=0."""
    VSMcalibration = (
        Data.data[5, Data.find_col("m (emu)")]
        / Data.data[5, Data.find_col("X (V)")]
    )  # taken from row 5 at random
    for i in range(len(Data.data[:, "0"])):
        Data.data[i, Data.find_col("m (emu)")] = (
            VSMcalibration * Data.data[i, Data.find_col("R (V)")]
        )
    return Data


def plotmH(data):
    """Take a stoner type data source."""
    plot.clf()
    plot.xlabel("H(T)")
    plot.ylabel("m(1e-5 emu)")
    plot.plot(data.column("H"), data.column("m (emu)"), "b-")
    plot.draw()


def splitFileName(myFileName):
    """Split a file name into its name and its extension part.

    (returns two part list ['name','ext'] or ['name',''] if no extension"""
    for i in range(len(myFileName) - 1, -1, -1):
        if myFileName[i] == ".":
            fileName = myFileName[:i]
            fileExt = myFileName[i:]
            return [fileName, fileExt]
    return [myFileName, ""]


def editData(Data, operations):
    """Take stoner type Data file and an operations list and performs the operations listed."""
    if 0 in operations:
        return Data
    N = int(input("Input the number of saturated data points on each arm:   "))
    if 1 in operations:
        Data = makeTruem(Data)
    if 2 in operations:
        Data.data = driftEliminator(Data.data, N)
    if 3 in operations:
        Data.data = diamagBackgroundRem(Data.data, N)
    if 4 in operations:
        Data.data = shift(Data.data, N)
    if 5 in operations:
        Data = invert(Data)
    return Data


# Set up a directory and determine files to be processed
directoryName = input("Enter path to directory in which files are stored:   ")
os.chdir(directoryName)
filenames = os.listdir(directoryName)
i = 0
for item in filenames:
    print(i, ". ", item)
    i += 1
fCounter = int(
    input(
        "Enter the number of the file you wish to start at (program will cycle through files from that point.):\n"
    )
)
if "EditedFiles" not in filenames:
    os.mkdir("EditedFiles")

timeout = 0
# Main program loop, cycle through selected files
while True:
    path = filenames[fCounter]
    fCounter += 1
    fr = open(path, "r", encoding="utf-8")
    data = fr.readlines()  # Get the file into an array
    fr.close()
    pathsplit = splitFileName(path)
    fw = open(  # pylint: disable=unspecified-encoding
        "EditedFiles/" + pathsplit[0] + "_edit.txt", "w", enconcoding="utf-8"
    )
    data = deleteCorruptLines(data)
    fw.writelines(
        data
    )  # put the uncorrupted lines into the file so that Stoner can open it
    fw.close()
    while True:  # open the file
        try:
            Data = Stoner.Data("EditedFiles/" + pathsplit[0] + "_edit.txt")
            break
        except ValueError:
            try:
                Data = Stoner.Data("EditedFiles/" + pathsplit[0] + "_edit.txt")
                break
            except ValueError:
                timeout += 1  # if get 5 files unreadable in a row then finish the program
                print("Could not read file ", path)
                if timeout <= 5:
                    break
    fw.close()
    while True:
        plotmH(Data)
        if not ("Original m (emu)" in Data.column_headers):
            Data.add_column(Data.column("m (emu)"), "Original m (emu)")
        print(
            "\nOK you have 5 options here, have a look at the plot and please tell me which ones ",
            'you would like by entering a string eg "125" for options 1, 2 and 5\n',
        )
        print(
            "0. Do nothing \n",
            "1.  Give true m from R (if theta is not zeroed in your data this will give you true m) \n",
            "2.  Remove a drift in m with time \n",
            "3.  Remove a diamagnetic background \n",
            "4.  Remove an overall shift in m \n",
            "5.  Reflect graph in y axis \n",
        )

        strOp = input("")
        operations = []  # array of options selected
        for i in range(len(strOp.strip())):
            operations.append(int(strOp[i]))
        t = Data.clone  # edit a copied array.
        t = editData(t, operations)
        plotmH(t)
        whatNext = input(
            "Press enter to save changes, r to restart or q to quit the program:  "
        )
        if whatNext == "r":
            continue
        if whatNext == "q":
            break
        Data = t
        Data.save(
            "EditedFiles/" + pathsplit[0] + "_edit.txt"
        )  # overwrite the file created earlier
        break
    if (
        whatNext == "q"
        or input(
            "Press enter to do file {} or q to quit:".format(
                filenames[fCounter]
            )
        )
        == "q"
    ):
        break
plot.close()
plot.ioff()
