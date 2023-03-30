import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    Compute accuracy for threshold based classifier
    '''

    embedding_type = ['content', 'tags', 'content_tags']

    for emb in embedding_type:
        df_train = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\DS_threshold_set.csv')
        df_test = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')

        X_train = df_train['doc2vec_distance_' + emb]
        y_train = df_train['HUMAN_CLASSIFICATION']
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

        df_train = pd.DataFrame(list(zip(X_train, y_train)),
                                columns=['doc2vec_distance_' + emb, 'HUMAN_CLASSIFICATION'])

        # 0, 1 = clones; 2 = distinct
        df_clones = df_train.query("HUMAN_CLASSIFICATION != 2")
        df_clones = df_clones['doc2vec_distance_' + emb].to_list()

        df_distinct = df_train.query("HUMAN_CLASSIFICATION == 2")
        df_distinct = df_distinct['doc2vec_distance_' + emb].to_list()

        X_test = df_test['doc2vec_distance_' + emb]
        y_test = df_test['HUMAN_CLASSIFICATION']
        df_test = pd.DataFrame(list(zip(X_test, y_test)),
                               columns=['doc2vec_distance_' + emb, 'HUMAN_CLASSIFICATION'])

        for threshold in np.linspace(0, 1, 11):
            # 0, 1 = clones; 2 = distinct
            df_clones = df_test.query("HUMAN_CLASSIFICATION != 2")
            df_clones_test = df_clones['doc2vec_distance_' + emb]
            tp = df_clones_test[df_clones_test > threshold].count()
            fn = len(df_clones_test) - tp

            df_distinct = df_test.query("HUMAN_CLASSIFICATION == 2")
            df_distinct_test = df_distinct['doc2vec_distance_' + emb]
            fp = df_distinct_test[df_distinct_test > threshold].count()
            tn = len(df_distinct_test) - fp

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            print(emb + "\t" + str(round(threshold, 1)) + "\t" + str(round(accuracy, 4)))
