import re
#import nltk
#nltk.download('punkt')

"""
DICTIONARY CLASS :
From a list with many tweets, creates a dictionary :
 ________________________
|     0     |     1      |
|___________|____________|
|    WORD   | CARD(WORD) |
|___________|____________|
|    ...    |    ...     |
|___________|____________|
    * CARD(WORD) = counts how many times this word appears in our initial list.
"""


class Dictionary:
    """
    @attr   _data               data that we use to create the dictionary
    @attr   _dictionary         object created in this class
    """
    __slots__ = ["_data", "_dictionary"]

    def __init__(self, data):
        """
        Initializes a new dictionary with given data as an object "dict()".
        @param  data
        """
        self._data = data
        self._dictionary = dict()

    def create_dictionary(self):
        """
        Creates the positive or negative dictionary with words from training tweets.
        """
        my_dict = dict()
        for tweet in self._data[:, 0]:
            if type(tweet) == str:
                for word in re.compile(r'\w+').findall(tweet):
                    my_dict[word] = my_dict.setdefault(word, 0) + 1
        self._dictionary = my_dict

    def create_sized_dictionary(self, size):
        """
        Creates the dictionary with maximum size from words from training tweets.
        @param  size    maximal length of the dictionary
        """
        my_sized_dict = dict()
        actual_size = 0
        for tweet in self._data[:, 0]:
            if actual_size <= size:
                if type(tweet) == str:
                    for word in re.compile(r'\w+').findall(tweet):
                        my_sized_dict[word] = my_sized_dict.setdefault(word, 0) + 1
                        actual_size += 1
            else:
                break
        self._dictionary = my_sized_dict

    def get_dictionary(self):
        """
        Return the dictionary as an object dict()
        """
        return self._dictionary

    def get_dictionary_length(self):
        """
        Give the raw count of our dictionary.
        """
        return len(list(self._dictionary))

    def get_dictionary_card(self):
        """
        Give the sum of every word cardinal : total amount of word in the dictionary.
        """
        somme = 0
        for word in self._dictionary:
            somme += self._dictionary[word]
        return somme
