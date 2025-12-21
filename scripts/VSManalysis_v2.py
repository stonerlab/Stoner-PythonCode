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

import matplotlib.pyplot as plot
import numpy as np

import Stoner

plot.ion()  # put the plotting in interactive mode
raw_input = input
print("Analysis program for VSM files\n\n\n")
print("Please wait while program loads...\n")
# vsm_calibration=0.86224   #m to X ratio, determined by reference Py sample


def delete_corrupt_lines(data):
    """Take data array and returns data with lines with 6:0 or --- in them deleted."""
    del_count = 0
    for line in data:
        if line.find("6:0") != -1:
            data.pop(data.index(line))
            del_count += 1
        if line.find("---") != -1:
            data.pop(data.index(line))
            del_count += 1
    if del_count > 6:
        input(
            "I've detected a lot of bad data in your file you should check it"
        )
    return data


def drift_eliminator(data, N):
    """Take off a linear drift of m with time from the data."""
    max_harg = int(np.argmax(data[: len(data[:, 0]) / 2, 1]))
    final_harg = int(len(data[:, 1]) - 1)
    # linear fit to saturated data (+- N points from max field and final field)
    final_fit = np.polyfit(
        data[final_harg - N : final_harg, 1],
        data[final_harg - N : final_harg, 2],
        1,
    )
    first_fit = np.polyfit(
        data[max_harg - N : max_harg + N, 1],
        data[max_harg - N : max_harg + N, 2],
        1,
    )
    pt = np.zeros((2, 2))
    pt[0] = [
        data[max_harg, 0],
        first_fit[0] * data[max_harg, 1] + first_fit[1],
    ]  # (time,m) at max field
    pt[1] = [
        data[final_harg, 0],
        final_fit[0] * data[final_harg, 1] + final_fit[1],
    ]
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
    max_harg = int(
        np.argmax(data[: len(data[:, 0]) / 2, 1])
    )  # gives the data index of maximum field (in the first half of the data
    min_harg = int(np.argmin(data[:, 1]))
    # average saturated data (+- N points from max/min field)
    high_ave = np.average(data[(max_harg - N) : (max_harg + N), 2])
    low_ave = np.average(data[min_harg - N : min_harg + N, 2])
    data[:, 2] = data[:, 2] - (high_ave + low_ave) / 2.0
    return data


def diamag_background_rem(data, N):
    """Remove a diamagnetic background using a linear fit to the N data points surrounding the max and min field."""
    max_harg = int(np.argmax(data[: len(data[:, 0]) / 2, 1]))
    min_harg = int(np.argmin(data[:, 1]))

    # linear fit to saturated data (+- N points from max/min field)
    highFit = np.polyfit(
        data[max_harg - N : max_harg + N, 1],
        data[max_harg - N : max_harg + N, 2],
        1,
    )
    low_fit = np.polyfit(
        data[min_harg - N : min_harg + N, 1],
        data[min_harg - N : min_harg + N, 2],
        1,
    )
    # Average grad
    fit_grad = (highFit[0] + low_fit[0]) / 2
    # Delete linear grad from all data
    data[:, 2] = data[:, 2] - (fit_grad * data[:, 1])
    return data


def invert(data):
    """Flip data in y axis."""
    for i in range(len(data.data[:, "0"])):
        data.data[i, data.find_col("m (emu)")] = -data.data[
            i, data.find_col("m (emu)")
        ]
    return data


def makeTruem(data):
    """VSM takes m from lock in X, this takes m from lock in R, useful if theta!=0."""
    vsm_calibration = (
        data.data[5, data.find_col("m (emu)")]
        / data.data[5, data.find_col("X (V)")]
    )  # taken from row 5 at random
    for i in range(len(data.data[:, "0"])):
        data.data[i, data.find_col("m (emu)")] = (
            vsm_calibration * data.data[i, data.find_col("R (V)")]
        )
    return data


def plotmH(data):
    """Take a stoner type data source."""
    plot.clf()
    plot.xlabel("H(T)")
    plot.ylabel("m(1e-5 emu)")
    plot.plot(data.column("H"), data.column("m (emu)"), "b-")
    plot.draw()


def split_filename(my_filename):
    """Split a file name into its name and its extension part.

    (returns two part list ['name','ext'] or ['name',''] if no extension.
    """
    for i in range(len(my_filename) - 1, -1, -1):
        if my_filename[i] == ".":
            fileName = my_filename[:i]
            fileExt = my_filename[i:]
            return [fileName, fileExt]
    return [my_filename, ""]


def edit_data(data, operations):
    """Take stoner type data file and an operations list and performs the operations listed."""
    if 0 in operations:
        return data
    N = int(input("Input the number of saturated data points on each arm:   "))
    if 1 in operations:
        data = makeTruem(data)
    if 2 in operations:
        data.data = drift_eliminator(data.data, N)
    if 3 in operations:
        data.data = diamag_background_rem(data.data, N)
    if 4 in operations:
        data.data = shift(data.data, N)
    if 5 in operations:
        data = invert(data)
    return data


if __name__ == "__main__":
    # Set up a directory and determine files to be processed
    directory_name = input(
        "Enter path to directory in which files are stored:   "
    )
    os.chdir(directory_name)
    filenames = os.listdir(directory_name)
    i = 0
    for item in filenames:
        print(i, ". ", item)
        i += 1
    f_counter = int(
        input(
            "Enter the number of the file you wish to start at (program will cycle through files from that point.):\n"
        )
    )
    if "EditedFiles" not in filenames:
        os.mkdir("EditedFiles")

    timeout = 0
    # Main program loop, cycle through selected files
    while True:
        path = filenames[f_counter]
        f_counter += 1
        with open(path, "r", encoding="utf-8") as fr:
            data = fr.readlines()  # Get the file into an array
        pathsplit = split_filename(path)
        with open(  # pylint: disable=unspecified-encoding
            "EditedFiles/" + pathsplit[0] + "_edit.txt",
            "w",
            enconcoding="utf-8",
        ) as fw:
            data = delete_corrupt_lines(data)
            fw.writelines(
                data
            )  # put the uncorrupted lines into the file so that Stoner can open it
        while True:  # open the file
            try:
                data = Stoner.Data("EditedFiles/" + pathsplit[0] + "_edit.txt")
                break
            except ValueError:
                try:
                    data = Stoner.Data(
                        "EditedFiles/" + pathsplit[0] + "_edit.txt"
                    )
                    break
                except ValueError:
                    timeout += 1  # if get 5 files unreadable in a row then finish the program
                    print("Could not read file ", path)
                    if timeout <= 5:
                        break
        fw.close()
        while True:
            plotmH(data)
            if "Original m (emu)" not in data.column_headers:
                data.add_column(data.column("m (emu)"), "Original m (emu)")
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
            t = data.clone  # edit a copied array.
            t = edit_data(t, operations)
            plotmH(t)
            what_next = input(
                "Press enter to save changes, r to restart or q to quit the program:  "
            )
            if what_next == "r":
                continue
            if what_next == "q":
                break
            data = t
            data.save(
                "EditedFiles/" + pathsplit[0] + "_edit.txt"
            )  # overwrite the file created earlier
            break
        if (
            what_next == "q"
            or input(
                f"Press enter to do file {filenames[f_counter]} or q to quit:"
            )
            == "q"
        ):
            break
    plot.close()
    plot.ioff()
