import numpy as np

"""
DATA CLASS :
Create a dataset as a numpy array :
     ________________________
    |     0     |     1      | 
    |___________|____________|
    |   TWEET   |   POS/NEG  | 
    |___________|____________|
    |    ...    |    ...     |    
    |___________|____________|
Allows to create training and testing set for our model.
"""


class Data:
    """
    @attr   _data               matrix containing the data
    @attr   _pos_data	  	    matrix of positive samples
    @attr   _neg_data	  	    matrix of negative samples
    @attr   _pos_spl_nb	  	    number of positive samples in the training set
    @attr   _neg_spl_nb	  	    number of negative samples in the training set
    """
    __slots__ = ["_data", "_pos_data", "_neg_data", "_pos_spl_nb", "_neg_spl_nb"]

    def __init__(self, data):
        """
        Initializes a new data set with given data.
        Calculates the number of positive and negative samples.
        @param  data
        """
        self._data = data[1:, :]
        np.random.shuffle(self._data)
        self._pos_data = self._data[self._data[:, 1] == 1]
        self._neg_data = self._data[self._data[:, 1] == 0]
        self._pos_spl_nb = 0
        self._neg_spl_nb = 0

    def create_sets_holdout(self, percent=0.8):
        """
        Given a dataset, constructs the training and test sets with Holdout.
        @param  percent     proportion of the dataset that will become training set
        """
        assert percent > 0.0, print("Holdout percent should be greater than 0%")
        split_pos_nb = int(len(self._pos_data) * percent)
        split_neg_nb = int(len(self._neg_data) * percent)
        self._pos_spl_nb = split_pos_nb
        self._neg_spl_nb = split_neg_nb

        # training set
        train_pos = self._pos_data[:split_pos_nb]
        train_neg = self._neg_data[:split_neg_nb]

        # testing set
        test_pos = np.array(self._pos_data[split_pos_nb:])
        test_neg = np.array(self._neg_data[split_neg_nb:])
        test_set = np.concatenate((test_pos, test_neg), axis=0)
        np.random.shuffle(test_set)

        return train_pos, train_neg, test_set

    def create_sets_cv(self, set_number, k=5):
        """
        Given a dataset, constructs a training and a testing set from the cross validation.
        @param  set_number      index of the k-fold that we use as testing set
        @param  k               number of parts into what we divide the dataset
        """
        n_pos_data = int(len(self._pos_data) / k)
        n_neg_data = int(len(self._neg_data) / k)
        self._pos_spl_nb = len(self._pos_data) - n_pos_data
        self._neg_spl_nb = len(self._neg_data) - n_neg_data

        # training set
        train_pos = np.concatenate((np.array(self._pos_data[:(set_number - 1) * n_pos_data]),
                                    np.array(self._pos_data[set_number * n_pos_data:])), axis=0)
        train_neg = np.concatenate((np.array(self._neg_data[:(set_number - 1) * n_neg_data]),
                                    np.array(self._neg_data[set_number * n_neg_data:])), axis=0)

        # testing set
        test_pos = np.array(self._pos_data[(set_number - 1) * n_pos_data:set_number * n_pos_data])
        test_neg = np.array(self._neg_data[(set_number - 1) * n_neg_data:set_number * n_neg_data])
        test_set = np.concatenate((test_pos, test_neg), axis=0)
        np.random.shuffle(test_set)

        return train_pos, train_neg, test_set

    def get_pos_spl_nb(self):
        """
        Once the training and testing sets are created, returns how many positive samples are in the training set
        """
        return self._pos_spl_nb

    def get_neg_spl_nb(self):
        """
        Once the training and testing sets are created, returns how many negative samples are in the training set
        """
        return self._neg_spl_nb

    def to_string(self):
        """
        Represents the database as a string
        """
        txt = "Class: database.py\n"
        txt += "  [X] Positive samples in the database:	 %d samples\n" % len(self._pos_data)
        txt += "  [X] Negative samples in the database:	 %d samples\n" % len(self._neg_data)
        txt += "  [X] Positive samples in training set:	 %d samples\n" % self._pos_spl_nb
        txt += "  [X] Negative samples in training set:	 %d samples\n" % self._neg_spl_nb
        return txt
