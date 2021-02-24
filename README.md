# Naive Bayes Sentiment Classifier
## INTRODUCTION
### Problem presentation
Given a database containing more than a million tweets, the problem was to create a model able to decide if a certain tweet would show a positive or a negative sentiment. For this purpose, the instruction was to create a Bayesian network to carry out this task.
In order to construct this model, it is necessary to create both training and testing sets. The first one will be used to create dictionary and to determine probabilities used in the algorithm. The second one will be used to test the model and determine if it is acceptable or not.

### Database
We have 2 databases :
- SentimentAnalysisDataset.csv, containing original tweets
- FinalStemmedSentimentAnalysisDataset.csv, where tweets have been processed with Lancaster Python Stemmer

To implement our model, we choose to use the second database because words are reduced to their root and all punctuation marks that could introduce noise have been removed. The .csv file contains four different columns indicating the identifier of the tweet, the tweet as a string, the date that the tweet was made and the label to which it belongs.

The file is rather equilibrate because it contains:
- 1'564'302 samples
- 781'650 positive tweets
- 782'651 negative tweets.

## BAYESIAN NETWORK STRUCTURE

### Structure of our program

#### Class READER
Handles the reading operations to give an usable database.
Attributes:
- folder: folder where all the files related to the database are located
Methods:
- read_data(filename: Read the file where the data is and stores the interesting data from the file in a numpy matrix: columns = tweetText, sentimentLabel ; rows = data samples
- Both columns tweetId and tweetDate are note useful for our model.

#### Class DATA
Create a dataset as a numpy array. Allows to create training and testing set for our model.

Attributes:
- data: matrix containing the data
- pos_data: matrix of positive samples
- neg_data: matrix of negative samples
- pos_spl_nb: number of positive samples in the training set
- neg_spl_nb: number of negative samples in the training set

Methods:
- create_sets_holdout(percent): given a dataset, constructs the training and test sets with holdout. "percent" represents proportion of the dataset that will become training set 5
- create_sets_cv(set_number, k): given a dataset, constructs a training and a testing set from the cross validation. set_number is the index of the k-fold that we use as testing set and k is the number of parts into what we divide the dataset.
- get_pos_spl_nb(): once the training and testing sets are created, returns how many positive samples are in the training set.
- get_neg_spl_nb(): once the training and testing sets are created, returns how many negative samples are in the training set.
- to_string(): represents the database as a string.


#### Class DICTIONARY

From a list with many tweets, creates a dictionary :
CARD(WORD) = counts how many times this word appears in our initial list.

Attributes:
- data: data that we use to create the dictionary
- dictionary: object created in this class
- 
Methods:
- create_dictionary(): creates the positive or negative dictionary with words from training tweets.
- create_sized_dictionary(size): creates the dictionary with maximum size from words from training tweets. Size parameter is the maximal cardinal of the dictionary
- get_dictionary(): return the dictionary as an object dict()
- get_dictionary_length(): give the raw count of our dictionary
- get_dictionary_card(): give the sum of every word cardinal : total amount of word in
the dictionary.

#### Class BAYES

Giving 2 dictionaries (one positive, the other negative) and a testing set, the Naïve Bayes algorithm will predict sentiments for each tweet of this testing set. For this purpose, probabilities must be calculated (cf. section "Calcul probabilities").

Attributes:
- pos_dictionary: dictionary containing words of positive samples
- neg_dictionary: dictionary containing words of negative samples
- test_set: testing set containing tweets for which we want to predict corresponding sentiment
- predictions: array of predicted sentiments for each tweet of the testing set
- metrics: list containing the evaluation metrics
- conf_matrix: array representing the confusion matrix

Methods:
- predict_sentiments(laplace_smoothing, pos_spl_nb, neg_spl_nb): given the number of positive and negative tweets in the training set, determine if a tweet is rather positive, negative or undetermined. "laplace_smoothing" gives the information whether or not Laplace Smoothing is used
- compare_sentiments(): compares the ground truth values of the training set to the predicted values for the target feature
- print_confusion_matrix() and plot_confusion_matrix(): displays the confusion matrix for this model
- print_metrics(): displays metrics : accuracy, precision, recall, specificity


### Userguide

To use this program, a file Main.py exists that allows to run every previous methods. For that purpose, the Constants.py file should be completed:
- FOLDER_PATH: give the path of the folder where files are
- FILENAME: name of the csv. file
- VALIDATION: validation method ('holdout' or 'crossvalidation')
- HOLDOUT_PERCENT: if validation method is holdout, it is necessary to choose a percentage to dispatch data between training and testing sets
- K: if validation method is cross-validation, it is necessary to choose a number k to divide the dataset in k parts
- SIZED_DCT: boolean that express whether or not we want to give our dictionaries a maximal size
- SIZE: if SIZED_DCT is True, give the maximal length that we want for our dictionaries
- LAPLACE_SMOOTHING: boolean to express whether or not we want to use Laplace Smoothing to calculate probabilities for the predictive algorithm
- ACCURACY, PRECISION, RECALL, SPECIFICITY: put metrics in a certain order
- TP, TN, FP, FN: put boxes of the confusion matrix in the right order


### Validation method
In this program, two validation methods are possible: holdout or cross-validation. While runing the algorithm, we can see that results of both methods are rather close. Moreover cross-validation requires a few minutes while holdout needs less than a minute.


## DICTIONARY IMPLEMENTATION

### Structure and content

A Bayesian Network used for a binary prediction requires two dictionaries: in our case, we need a dictionary for positive tweets and another one for negative tweets. After trying different types of structures (numpy arrays, lists, etc), we can notice that they are not efficient enough for such a quantity of data. Consequently, an alternative way of saving all such dictionaries was to use the dict() structure from python:

dictionary = {'is':39578, 'the':9638, 'new':1903'}

Keys are the words composing the dictionary, values are the number of times each word appears in its category.

### Mulitple dictionaries

We browse the training set of positive tweets, and for every word of every tweet, we put it in the dictionary: if it is already in the dictionary, we just add 1 to its value, otherwise we put it in the dictionary with a value of 1.

To browse all the tweets, we need to tokenize each tweet. First idea was to use nltk package with word_tokenize() method, but finally it considerably extended run time. An alternative is to use re package with compile().findall() method that gives greater times.

