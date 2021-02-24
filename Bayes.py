import numpy as np
import re
import matplotlib.pyplot as plt

from Constants import *

"""
BAYES CLASS :
Giving 2 dictionaries (one positive, the other negative) and a testing set, the
Naïve Bayes algorithm will predict sentiments for each tweet of this testing set.
For this purpose, probabilities must be calculated :
- with Laplace Smoothing:

                      CARD(WORD) + alpha
    p = ________________________________________________
        sum(CARD(WORD i)) + alpha*[length(dictionary)+1]
    * alpha = 1

- or without Laplace Smoothing:

            CARD(WORD)
    p = _________________
        sum(CARD(WORD i))
    * alpha = 1

"""


class Bayes:
    """
    @attr   _pos_dictionary     dictionary containing words of positive samples
    @attr   _neg_dictionary     dictionary containing words of negative samples
    @attr   _test_set           testing set containing tweets for which we want to predict corresponding sentiment
    @attr   _predictions        array of predicted sentiments for each tweet of the testing set
    @attr   _metrics            list containing the evaluation metrics
    @attr   _conf_matrix        array representing the confusion matrix :
                                 _________________
                                |        |        |
                                |   TN   |   FP   |
                                |________|________|
                                |        |        |
                                |   FN   |   TP   |
                                |________|________|

    """
    __slots__ = ["_pos_dictionary", "_neg_dictionary", "_test_set", "_predictions", "_metrics", "_conf_matrix"]

    def __init__(self, pos_dictionary, neg_dictionary, test_set):
        """
        Initializes a new Bayes class.
        @param  pos_dictionary      dictionary with positive samples of the training set
        @param  neg_dictionary      dictionary with negative samples of the training set
        @param  test_set            testing set
        """
        self._pos_dictionary = pos_dictionary
        self._neg_dictionary = neg_dictionary
        self._test_set = test_set
        self._predictions = []
        self._metrics = None
        self._conf_matrix = [0, 0, 0, 0]

    def predict_sentiments(self, laplace_smoothing, pos_spl_nb, neg_spl_nb):
        """
        Determine if a tweet is rather positive, negative or undetermined.   
        @param  laplace_smoothing   boolean that gives the information if we want to use L. Smoothing for probabilities
        @param  pos_spl_nb          number of positive tweets in the training set
        @param  neg_spl_nb          number of negative tweets in the training set
        """
        nb_undetermined = 0
        # initialize dictionary characteristics
        total_spl_nb = pos_spl_nb + neg_spl_nb
        pos_probability = pos_spl_nb / total_spl_nb  # P(positive)
        neg_probability = neg_spl_nb / total_spl_nb  # P(negative)
        pos_dct_len = self._pos_dictionary.get_dictionary_length()  # number of words in positive_dictionary
        neg_dct_len = self._neg_dictionary.get_dictionary_length()  # number of words in negative_dictionary
        pos_dct_card = self._pos_dictionary.get_dictionary_card()  # sum of CARD(word) for every word of the dictionary
        neg_dct_card = self._neg_dictionary.get_dictionary_card()  # sum of CARD(word) for every word of the dictionary
        pos_dct = dict(self._pos_dictionary.get_dictionary())
        neg_dct = dict(self._neg_dictionary.get_dictionary())

        # ---------------- LAPLACE SMOOTHING TRUE ---------------
        if laplace_smoothing is True:
            for tweet in self._test_set[:, 0]:
                probability1 = pos_probability
                probability0 = neg_probability
                if type(tweet) == str:
                    for word in re.compile(r'\w+').findall(tweet):
                        probability1 *= (pos_dct.setdefault(word, 0) + 1) / (pos_dct_len + pos_dct_card)  # P(tweet = pos)
                        probability0 *= (neg_dct.setdefault(word, 0) + 1) / (neg_dct_len + neg_dct_card)  # P(tweet = neg)

                # decision : positive or negative tweet ?
                if probability0 < probability1:
                    sentiment = 1
                elif probability1 < probability0:
                    sentiment = 0
                else:
                    sentiment = 2
                    nb_undetermined += 1
                self._predictions.append(sentiment)

        # ---------------- LAPLACE SMOOTHING FALSE ---------------
        else:
            for tweet in self._test_set[:, 0]:
                probability1 = pos_probability
                probability0 = neg_probability
                if type(tweet) == str:
                    for word in re.compile(r'\w+').findall(tweet):
                        probability1 *= pos_dct.setdefault(word, 0) / pos_dct_card  # P(tweet = pos)
                        probability0 *= neg_dct.setdefault(word, 0) / neg_dct_card  # P(tweet = neg)

                # decision : positive or negative tweet ?
                if probability0 < probability1:
                    sentiment = 1
                elif probability1 < probability0:
                    sentiment = 0
                else:
                    sentiment = 2
                    nb_undetermined += 1
                self._predictions.append(sentiment)
        return nb_undetermined

    def compare_sentiments(self):
        """
        Compares the ground truth values of the training set to the predicted values for the target feature
        """
        test_y = [raw[-1] for raw in self._test_set]
        predicted_y = self._predictions

        if len(test_y) != len(predicted_y):
            print(
                f'predicted_y : length = {len(predicted_y)} and test_y : length = {len(test_y)} have not the same length ')
            exit()

        metrics = [None] * 4
        conf_matrix = [0, 0, 0, 0]

        for t, p in zip(test_y, predicted_y):
            if t == 1 and p:
                conf_matrix[TP] += 1
            elif t == 1 and not p:
                conf_matrix[FN] += 1
            elif t == 0 and p:
                conf_matrix[FP] += 1
            elif t == 0 and not p:
                conf_matrix[TN] += 1

        tp = conf_matrix[TP]
        fn = conf_matrix[FN]
        fp = conf_matrix[FP]
        tn = conf_matrix[TN]

        metrics[ACCURACY] = (tp + tn) / (tp + tn + fp + fn)
        metrics[PRECISION] = tp / (tp + fp)
        metrics[RECALL] = tp / (tp + fn)
        metrics[SPECIFICITY] = tn / (tn + fp)

        self._metrics = metrics
        self._conf_matrix = conf_matrix
        return metrics, conf_matrix

    def print_confusion_matrix(self):
        """
        displays the confusion matrix for this model
        """
        print("--- CONFUSION MATRIX ------------------------------------------")
        print(f'TP:{self._conf_matrix[TP]} | FN:{self._conf_matrix[FN]}')
        print(f'FP:{self._conf_matrix[FP]} | TN:{self._conf_matrix[TN]}')

    def plot_confusion_matrix(self):
        """
        displays the confusion matrix for this model
        """
        grid = np.array(self._conf_matrix).reshape(2, 2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reality")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        im = ax.imshow(grid, interpolation='none', aspect='auto')

        labels = ["TP", "FN", "FP", "TN"]

        for (j, i), label in np.ndenumerate(grid):
            ax.text(i, j, f'{labels[i + j]}:{label}', ha='center', va='center')

        fig.colorbar(im)
        plt.plot()

    def print_metrics(self):
        """
        displays metrics
        """
        print("--- METRICS ---------------------------------------------------")
        print(f'Accuracy: {self._metrics[ACCURACY]}')
        print(f'Precision: {self._metrics[PRECISION]}')
        print(f'Recall: {self._metrics[RECALL]}')
        print(f'Specificity: {self._metrics[SPECIFICITY]}')
