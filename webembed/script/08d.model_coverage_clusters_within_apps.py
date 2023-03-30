import csv
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd
from natsort import natsorted


def is_clone(model, distance):
    try:
        prediction = model.predict(np.array(distance).reshape(1, -1))  # 0 = near-duplicates, 1 = distinct
    except ValueError:
        prediction = [0]

    if prediction == [0]:
        return True
    else:
        return False


APPS = ['addressbook', 'claroline', 'dimeshift', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix', 'ppma']

CLASSIFIERS = {
    'addressbook': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'claroline': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'dimeshift': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'mantisbt': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'mrbs': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'pagekit': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'petclinic': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'phoenix': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],

    'ppma': ["path-to-trained-webembed-classifier",
                    "path-to-trained-rted-classifier",
                    "path-to-trained-pdiff-classifier"],
}

OUTPUT_CSV = True
filename = 'csv_results_table/rq3-within-apps.csv'

if __name__ == '__main__':
    os.chdir("..")

    SETTINGS = ["within-apps-"]
    APPS = ['mantisbt']

    if OUTPUT_CSV:
        # create csv file to store the results
        if not os.path.exists(filename):
            header = ['Setting', 'App', 'Feature', 'Classifier', 'Precision', 'Recall', 'F1']
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)

    for app in APPS:
        print(app)
        comparison_df = None
        for feature in np.arange(3):
            classifier = CLASSIFIERS[app][feature]
            print(classifier)

            if OUTPUT_CSV:
                comparison_df = pd.read_csv(filename)

            column = None
            if 'dom-rted' in classifier:
                column = 'dom-rted'.upper()
            elif 'visual-hyst' in classifier:
                column = 'VISUAL_Hyst'
            elif 'visual-pdiff' in classifier:
                column = 'VISUAL-PDiff'
            elif 'doc2vec-distance-content' in classifier:
                column = 'doc2vec-distance-content'
            elif 'doc2vec-distance-tags' in classifier:
                column = 'doc2vec-distance-tags'
            elif 'doc2vec-distance-content-tags' in classifier:
                column = 'doc2vec-distance-content-tags'
            elif 'doc2vec-distance-all' in classifier:
                column = 'doc2vec-distance-all'

            ss = pd.read_csv('script/SS_threshold_set.csv',
                             usecols=['appname', 'state1', 'state2', column.replace('-', '_')])
            ss = ss.query("appname == @app")
            ss = ss.drop(['appname'], axis=1)

            model = None
            try:
                model = pickle.load(open(classifier, 'rb'))
            except FileNotFoundError:
                print("Cannot find classifier %s" % classifier)
                exit()
            # except pickle.UnpicklingError:
            #     # print(CLASSIFIER_USED)
            #     exit()

            # convert distances to similarities
            ss[column.replace('-', '_')] = ss[column.replace('-', '_')].map(lambda dist: is_clone(model, dist))

            tuples = [tuple(x) for x in ss.to_numpy()]

            lis = tuples

            items = natsorted(set.union(set([item[0] for item in lis]), set([item[1] for item in lis])))

            value = dict(zip(items, range(len(items))))
            dist_matrix = np.zeros((len(items), len(items)))

            for i in range(len(lis)):
                # upper triangle
                dist_matrix[value[lis[i][0]], value[lis[i][1]]] = lis[i][2]
                # lower triangle
                dist_matrix[value[lis[i][1]], value[lis[i][0]]] = lis[i][2]

            new_ss = pd.DataFrame(dist_matrix, columns=items, index=items)
            new_ss.to_csv('script/SS_as_distance_matrix.csv')

            dictionary = {}
            for index, row in new_ss.iterrows():
                clones = []
                sel = new_ss.loc[new_ss[index] == 1.0]
                clones.append(sel[index].keys().tolist())
                dictionary[index] = clones

            with open('output/' + app + '.json', 'r') as f:
                data = json.load(f)

            number_in_common = 0
            number_gt = 0
            number_d2v = len(dictionary.keys())

            for cluster in data:
                value = data[cluster]
                # print(cluster + ' -> ' + str(value))

                key = value[0]  # I treat the first item of the cluster as key
                value.remove(key)

                if len(value) == 0:  # empty cluster
                    pass
                else:
                    # print(value)
                    pairs = list(itertools.combinations(value, 2))
                    # print(pairs)
                    number_gt += len(pairs)
                    for pair in pairs:
                        state1 = pair[0]
                        state2 = pair[1]
                        if state2 in dictionary[state1][0]:
                            number_d2v += 1
                            number_in_common += 1
                        else:
                            number_d2v += 1
                # print(clones)
                # print(dict[key])

            print("number of pairs in ground truth: %d" % number_gt)
            print("number of pairs in common: %d" % number_in_common)
            print("number of pairs %s: %d" % (column, number_d2v))

            precision = number_in_common / number_d2v
            print("precision: %.2f" % precision)
            recall = number_in_common / number_gt
            print("recall: %.2f" % recall)
            try:
                f1 = (2 * ((precision * recall) / (precision + recall)))
            except ZeroDivisionError:
                f1 = 0
            print("f1: %.2f" % f1)

            if OUTPUT_CSV:
                d1 = pd.DataFrame(
                    {'Setting': SETTINGS[0][:-1],
                     'App': app,
                     'Feature': column,
                     'Classifier': classifier,
                     'Precision': [precision],
                     'Recall': [recall],
                     'F1': [f1]})
                comparison_df = pd.concat([comparison_df, d1])
                comparison_df.to_csv(filename, index=False)
