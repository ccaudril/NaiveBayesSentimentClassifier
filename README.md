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
- read_data(filename: Read the file where the data is and stores the interesting data from the file in a numpy matrix:
    o columns = tweetText, sentimentLabel
    o rows = data samples
- Both columns tweetId and tweetDate are note useful for our model.
