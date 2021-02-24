# Folder path
FOLDER_PATH = "Database/"

# Filename
FILENAME = "FinalStemmedSentimentAnalysisDataset.csv"

# Validation method
VALIDATION = 'holdout'  # { 'holdout', 'crossvalidation' }

# Percent for holdout
HOLDOUT_PERCENT = 0.9

# K for cross-validation
K = 8

# Possibility to choose a size for the dictionary
SIZED_DCT = True
SIZE = 9000000

# Laplace Smoothing
LAPLACE_SMOOTHING = False

# Metrics indices
ACCURACY = 0
PRECISION = 1
RECALL = 2
SPECIFICITY = 3
TP = 0
TN = 3
FP = 2
FN = 1
