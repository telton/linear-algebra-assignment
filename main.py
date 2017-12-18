# Tyler Elton
# MTH220 - Fall 2017
# Homework 6/7
# Best fit line for large dataset.

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# Style for the plot.
style.use('fivethirtyeight')

if __name__ == "__main__":
    # Read the data from the CSV file.
    # Open the file in universal line ending mode.
    with open('film-death-counts.csv', 'rU') as infile:
        # Read the file as a dictionary for each row ({header : value}).
        reader = csv.DictReader(infile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]

    # Set the variables we need.
    length = data['Length_Minutes']
    rating = data['IMDB_Rating']
    bodyCount = data['Body_Count']

    # Create the x and y data.
    x = np.array(bodyCount, dtype = np.float64)
    y = np.array(rating, dtype = np.float64)

    # Create A from x values, and all 1's for the b values (y = mx + b).
    A = np.vstack([x, np.ones(len(x))]).T

    # Compute least square to find m, b.
    m, b = np.linalg.lstsq(A, y)[0]

    # Set up the equation for the best fit line.
    bestFitLine = (m * x) + b

    # Compute the mean square error.
    # SS = (1/n)||b - bHat||
    xHat = np.vstack([m, b])
    ssNorm = np.matmul(A, xHat) - y
    meanSquareError = ((1/len(x)) * np.linalg.norm(ssNorm))
    print('Mean square error: ', meanSquareError)

    fig = plt.figure()
    fig.suptitle('IMDB Ratings of Movies Based on Body Count', fontsize = 16)

    # Plot the points.
    plt.scatter(x, y, color = 'r', label = 'IMDB rating by body count', s = 8)
    plt.plot(x, bestFitLine, color = 'b', label = 'Best fit line', linewidth = 2)

    # Add labels to the x/y axes, create the legend, then show the graph.
    plt.ylabel('IMDB Rating', fontsize = 12)
    plt.xlabel('Body count', fontsize = 12)
    plt.legend(title = 'Legend')
    plt.show()
