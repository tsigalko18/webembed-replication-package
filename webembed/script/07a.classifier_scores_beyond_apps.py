import csv
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    '''
    RQ1: configuration 1/3 BEYOND APPS
    Doc2Vec trained on DS + commoncrawl
    Classifiers trained on Labeled(DS)
    Classifiers tested on SS
    '''

    OUTPUT_CSV = True
    SAVE_MODELS = True
    filename = '../csv_results_table/rq1-beyond-apps.csv'

    # load Labeled(DS)
    df_labeled_ds = pd.read_csv('DS_threshold_set.csv')

    # load SS (all apps)
    df_ss = pd.read_csv('SS_threshold_set.csv')

    embedding_type = ['all', 'content', 'tags', 'content_tags', 'DOM_RTED', 'VISUAL_Hyst', 'VISUAL_PDiff']

    if OUTPUT_CSV:
        # create csv file to store the results
        if not os.path.exists(classifier_path):
            header = ['Setting', 'Model', 'Embedding', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1_0', 'F1_1']
            with open(classifier_path, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)

    for emb in embedding_type:
        print("embedding: %s" % emb)

        comparison_df = None
        if OUTPUT_CSV:
            comparison_df = pd.read_csv(classifier_path)

        names = [
            # "Dummy",
            # "Threshold",
            # "Nearest Neighbors",
            # "SVM RBF",
            # "Decision Tree",
            # "Gaussian Naive Bayes",
            # "Random Forest",
            # "Ensemble",
            # "Neural Network",
            "XGBoost"
        ]

        classifiers = [
            # DummyClassifier(strategy="stratified"),
            # "Threshold",
            # KNeighborsClassifier(),
            # SVC(),
            # DecisionTreeClassifier(),
            # GaussianNB(),
            # RandomForestClassifier(),
            # VotingClassifier(estimators=[('knn', KNeighborsClassifier()),
            #                              ('svm', SVC()),
            #                              ('dt', DecisionTreeClassifier()),
            #                              ('gnb', GaussianNB()),
            #                              ('rf', RandomForestClassifier())]),
            # MLPClassifier(max_iter=1000),
            GradientBoostingClassifier(n_estimators=100, max_depth=1)
        ]

        for name, model in zip(names, classifiers):

            if emb in {'DOM_RTED', 'VISUAL_Hyst', 'VISUAL_PDiff'}:
                feature = emb
            else:
                feature = 'doc2vec_distance_' + emb

            if name == "Threshold":

                # load Labeled(DS) as training set
                df_train = pd.read_csv('DS_threshold_set.csv')

                # load SS as test set (all apps)
                df_test = pd.read_csv('SS_threshold_set.csv')

                X_train = np.array(df_train[feature]).reshape(-1, 1)
                y_train = np.array(df_train['HUMAN_CLASSIFICATION'])

                X_test = np.array(df_test[feature]).reshape(-1, 1)
                y_test = np.array(df_test['HUMAN_CLASSIFICATION'])

                # 0, 1 = clones; 2 = distinct
                y_train[y_train == 1] = 0  # harmonize near-duplicates as 0's
                y_train[y_train == 2] = 1  # convert distinct as 1's

                y_test[y_test == 1] = 0  # harmonize near-duplicates as 0's
                y_test[y_test == 2] = 1  # convert distinct as 1's

                df_train = pd.DataFrame(list(zip(X_train, y_train)),
                                        columns=[feature,
                                                 'HUMAN_CLASSIFICATION'])

                # 0, 1 = clones; 2 = distinct
                df_clones = df_train.query("HUMAN_CLASSIFICATION != 2")
                df_clones = df_clones[feature].to_list()

                df_distinct = df_train.query("HUMAN_CLASSIFICATION == 2")
                df_distinct = df_distinct[feature].to_list()

                df_test = pd.DataFrame(list(zip(X_test, y_test)),
                                       columns=[feature,
                                                'HUMAN_CLASSIFICATION'])

                threshold = 0.8
                # 0, 1 = clones; 2 = distinct
                df_clones = df_test.query("HUMAN_CLASSIFICATION != 2")
                df_clones_test = df_clones[feature]
                tp = df_clones_test[df_clones_test > threshold].count()
                fn = len(df_clones_test) - tp

                df_distinct = df_test.query("HUMAN_CLASSIFICATION == 2")
                df_distinct_test = df_distinct[feature]
                fp = df_distinct_test[df_distinct_test > threshold].count()
                tn = len(df_distinct_test) - fp

                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_0 = 2 * ((precision * recall) / (precision + recall))
                f1_1 = 2 * ((precision * recall) / (precision + recall))
            else:
                # load Labeled(DS) as training set
                X_train = np.array(df_labeled_ds[feature]).reshape(-1, 1)
                y_train = np.array(df_labeled_ds['HUMAN_CLASSIFICATION'])

                # load SS as test set (all apps)
                X_test = np.array(df_ss[feature]).reshape(-1, 1)
                y_test = np.array(df_ss['HUMAN_CLASSIFICATION'])

                # 0, 1 = clones; 2 = distinct
                y_train[y_train == 1] = 0  # harmonize near-duplicates as 0's
                y_train[y_train == 2] = 1  # convert distinct as 1's

                y_test[y_test == 1] = 0  # harmonize near-duplicates as 0's
                y_test[y_test == 2] = 1  # convert distinct as 1's

                # fit the classifier
                model = model.fit(X_train, y_train)

                # save the classifier
                if SAVE_MODELS:
                    classifier_path = '../trained_classifiers/beyond-apps-' + \
                                      name.replace(" ", "-").replace("_", "-").lower() + \
                               '-' + \
                                      feature.replace(" ", "-").replace("_", "-").lower() + \
                               '.sav'
                    pickle.dump(model, open(classifier_path, 'wb'))
                    # another = pickle.load(open(filename, 'rb'))
                    # print()

                # predict the scores
                y_pred = model.predict(X_test)

                # compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1_0 = f1_score(y_test, y_pred, pos_label=0)
                f1_1 = f1_score(y_test, y_pred, pos_label=1)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

            print(f'{name}, '
                  f'accuracy: {accuracy}, '
                  f'precision: {precision}, '
                  f'recall: {recall}, '
                  f'f1_0: {f1_0}, '
                  f'f1_1: {f1_1}')

            if OUTPUT_CSV:
                a = ''
                if emb == 'content':
                    a = 'Content only'
                elif emb == 'tags':
                    a = 'Tags only'
                elif emb == 'content_tags':
                    a = 'Content and tags'
                elif emb == 'all':
                    a = "Ensemble"
                elif emb == 'DOM_RTED':
                    a = 'DOM_RTED'
                elif emb == 'VISUAL_PDiff':
                    a = 'VISUAL_PDiff'
                elif emb == 'VISUAL_Hyst':
                    a = 'VISUAL_Hyst'
                else:
                    print('nope')

                d1 = pd.DataFrame(
                    {'Model': ['DS_' + emb + '_' + 'modelsize100' + 'epoch31'],
                     'Embedding': [a],
                     'Classifier': [name],
                     'Accuracy': [accuracy],
                     'Precision': [precision],
                     'Recall': [recall],
                     'F1_0': [f1_0],
                     'F1_1': [f1_1]})

                comparison_df = pd.concat([comparison_df, d1])

        if OUTPUT_CSV:
            comparison_df.to_csv(filename, index=False)
