#!/usr/bin/python

from Reader import *
from Data import *
from Dictionary import *
from Bayes import *
from Constants import *

if __name__ == "__main__":

    print("BAYESIAN LEARNING \n")

    print("Creating READER class with data folder path...")
    reader = Reader(FOLDER_PATH)
    print("Reader class created")

    print("Reading data from  file...")
    raw_dataset = reader.read_data(FILENAME)
    print(f'Dataset data: shape: {raw_dataset.shape}\tsize: {raw_dataset.size} \n \n')

    print("Creating DATA class...")
    dataset = Data(raw_dataset)
    print(f'Dataset created')


    # --------- VALIDATION METHOD = HOLDOUT ----------

    if VALIDATION == 'holdout':
        print("Creating training and testing sets...")
        training_pos_set, training_neg_set, testing_set = dataset.create_sets_holdout(HOLDOUT_PERCENT)
        pos_spl_nb = dataset.get_pos_spl_nb()
        neg_spl_nb = dataset.get_neg_spl_nb()
        print(f'Dataset split between training and testing sets : {dataset.to_string()} \n \n')

        print("Creating 2 DICTIONARY classes...")
        dictionary1 = Dictionary(training_pos_set)
        dictionary0 = Dictionary(training_neg_set)
        print("Dictionary classes created")

        print("Creating and completing positive and negative dictionaries...")
        if SIZED_DCT is False:
            dictionary1.create_dictionary()
            print(f"Positive dictionary created")
            dictionary0.create_dictionary()
            print(f"Negative dictionary created \n \n")
        else:
            dictionary1.create_sized_dictionary(SIZE)
            print("Positive dictionary created")
            dictionary0.create_sized_dictionary(SIZE)
            print("Negative dictionary created \n \n")

        print("Creating BAYES class...")
        bayes = Bayes(dictionary1, dictionary0, testing_set)
        print("Bayes class created")

        print("Predicting sentiments for testing set...")
        nb_undetermined = bayes.predict_sentiments(LAPLACE_SMOOTHING, pos_spl_nb, neg_spl_nb)
        print("Prediction of sentiments for testing set done")
        print(f"Number of tweets with undetermined sentiments : {nb_undetermined}")

        print("Comparing sentiments from the dataset with predicted sentiments...")
        metrics, conf_matrix = bayes.compare_sentiments()

        bayes.print_metrics()
        bayes.print_confusion_matrix()
        bayes.plot_confusion_matrix()


    # --------- VALIDATION METHOD = CROSSVALIDATION ----------
    else:
        print("Proceeding with cross-validation algorithm")

        total_metrics = [0, 0, 0, 0]
        total_conf_matrix = [0, 0, 0, 0]

        for set_number in range(1, K+1):
            training_pos_set, training_neg_set, testing_set = dataset.create_sets_cv(set_number, K)
            pos_spl_nb = dataset.get_pos_spl_nb()
            neg_spl_nb = dataset.get_neg_spl_nb()
            print(f'Set number {set_number} created')

            dictionary1 = Dictionary(training_pos_set)
            dictionary0 = Dictionary(training_neg_set)
            if SIZED_DCT is False:
                dictionary1.create_dictionary()
                dictionary0.create_dictionary()
            else:
                dictionary1.create_sized_dictionary(SIZE)
                dictionary0.create_sized_dictionary(SIZE)
            print("Dictionaries created")

            bayes = Bayes(dictionary1, dictionary0, testing_set)
            nb_undetermined = bayes.predict_sentiments(LAPLACE_SMOOTHING, pos_spl_nb, neg_spl_nb)
            print(f'Prediction of sentiments for set {set_number} done')
            print(f'Number of tweets with undetermined sentiments : {nb_undetermined}')

            metrics, conf_matrix = bayes.compare_sentiments()

            for i in range(0, 4):
                total_metrics[i] += (metrics[i]/K)
                total_conf_matrix[i] += (conf_matrix[i]/K)

            bayes.print_metrics()
            bayes.print_confusion_matrix()
            bayes.plot_confusion_matrix()

        print(f'\n \n Cross-validation with {K}-fold metrics\' arithmetic mean results: \n')

        print("--- CONFUSION MATRIX ------------------------------------------")
        print(f'TP:{total_conf_matrix[TP]} | FN:{total_conf_matrix[FN]}')
        print(f'FP:{total_conf_matrix[FP]} | TN:{total_conf_matrix[TN]} \n \n')

        print("--- METRICS ---------------------------------------------------")
        print(f'Accuracy: {total_metrics[ACCURACY]}')
        print(f'Precision: {total_metrics[PRECISION]}')
        print(f'Recall: {total_metrics[RECALL]}')
        print(f'Specificity: {total_metrics[SPECIFICITY]}')