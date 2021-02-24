import numpy as np
import pandas as pd

'''
READER CLASS :
Handles the reading operations to give an usable database.
'''


class Reader(object):
    """
    @attr   _folder     folder where all the files related to the database are
                        located
    """
    __slots__ = ["_folder"]

    def __init__(self, folder):
        """
        Initializes a dataset reader.

        @param folder : folder path where all the files are located
        """
        self._folder = folder

    def read_data(self, filename):
        """
        Read the file where the data is.
        Stores the interesting data from the file in a numpy matrix.
        - columns = tweetText, sentimentLabel
        - rows = data samples

        @param filename : file containing data (.csv for example)
        """
        file = pd.read_csv(self._folder + filename, sep=';', usecols=[1, 3])
        return np.array([data for data in file.values])
