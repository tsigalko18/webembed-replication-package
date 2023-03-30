from os import mkdir
from re import A
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count

all_html_path = '..\\dataset\\GroundTruthModels'

model_content = Doc2Vec.load(
    '..\\trained_model\\FULL\\content_model_train_setsize300epoch5.doc2vec.model')
model_tags = Doc2Vec.load(
    '..\\trained_model\\FULL\\tags_model_train_setsize300epoch5.doc2vec.model')
model_content_tags = Doc2Vec.load(
    '..\\trained_model\\FULL\\content_tags_model_train_setsize300epoch5.doc2vec.model')


df = pd.read_csv('..\\dataset\\sets\\treshold_setGS.csv')
df['answer'] = [int(row != 2) for row in df['HUMAN_CLASSIFICATION']]
df = df.dropna(subset=['state1_content', 'state2_content', 'state1_tags',
                       'state2_tags', 'state1_content_tags', 'state2_content_tags'])


def load_and_create_embedding(metadata, model):
    with open(join(all_html_path, metadata['path'])) as fp:
        data = json.load(fp)
    return model.infer_vector(data).reshape(1, -1)


def calculate_states_similarity(inpt, s=0):
    try:
        (_, row) = inpt
        if s == 'all':
            # print('all')
            metadata_content1 = json.loads(row['state1_content'])
            metadata_content2 = json.loads(row['state2_content'])
            emb1_content = load_and_create_embedding(
                metadata_content1, model_content)
            emb2_content = load_and_create_embedding(
                metadata_content2, model_content)

            metadata_tags1 = json.loads(row['state1_tags'])
            metadata_tags2 = json.loads(row['state2_tags'])
            emb1_tags = load_and_create_embedding(metadata_tags1, model_tags)
            emb2_tags = load_and_create_embedding(metadata_tags2, model_tags)

            metadata_content_tags1 = json.loads(row['state1_content_tags'])
            metadata_content_tags2 = json.loads(row['state2_content_tags'])
            emb1_content_tags = load_and_create_embedding(
                metadata_content_tags1, model_content_tags)
            emb2_content_tags = load_and_create_embedding(
                metadata_content_tags2, model_content_tags)

            cos_sim_content = cosine_similarity(emb1_content, emb2_content)
            cos_sim_tags = cosine_similarity(emb1_tags, emb2_tags)
            cos_sim_content_tags = cosine_similarity(
                emb1_content_tags, emb2_content_tags)

            final_sim = np.array(
                [cos_sim_content[0, 0], cos_sim_tags[0, 0], cos_sim_content_tags[0, 0]])

            return final_sim, row['answer']

        if s == 'content':
            metadata_content1 = json.loads(row['state1_content'])
            metadata_content2 = json.loads(row['state2_content'])
            emb1_content = load_and_create_embedding(
                metadata_content1, model_content)
            emb2_content = load_and_create_embedding(
                metadata_content2, model_content)

            cos_sim_content = cosine_similarity(emb1_content, emb2_content)

            final_sim = np.array([cos_sim_content[0, 0]])

            return final_sim, row['answer']
        if s == 'tags':
            metadata_tags1 = json.loads(row['state1_tags'])
            metadata_tags2 = json.loads(row['state2_tags'])
            emb1_tags = load_and_create_embedding(metadata_tags1, model_tags)
            emb2_tags = load_and_create_embedding(metadata_tags2, model_tags)

            cos_sim_tags = cosine_similarity(emb1_tags, emb2_tags)

            final_sim = np.array([cos_sim_tags[0, 0]])

            return final_sim, row['answer']

        if s == 'content_tags':
            metadata_content_tags1 = json.loads(row['state1_content_tags'])
            metadata_content_tags2 = json.loads(row['state2_content_tags'])
            emb1_content_tags = load_and_create_embedding(
                metadata_content_tags1, model_content_tags)
            emb2_content_tags = load_and_create_embedding(
                metadata_content_tags2, model_content_tags)

            cos_sim_content_tags = cosine_similarity(
                emb1_content_tags, emb2_content_tags)

            final_sim = np.array([cos_sim_content_tags[0, 0]])

            return final_sim, row['answer']

    except Exception as e:
        print(e, row)
        return None


if __name__ == '__main__':
    apps = ['addressbook', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix', 'ppma']
    # TODO per below: tags, content, tags content all
    # apps = ['claroline', 'dimeshift','mantisbt', 'mrbs','pagekit', 'petclinic','phoenix', 'ppma']
    # apps = ['addressbook', 'claroline', 'dimeshift','mantisbt', 'mrbs','pagekit', 'petclinic','phoenix', 'ppma']
    # apps = ['addressbook','mantisbt', 'mrbs','pagekit', 'petclinic','phoenix', 'ppma']
    for app_name in apps:
        args = ['content', 'tags', 'content_tags', 'all']
        # args = ['all']
        df2 = df[df['appname'] == app_name]
        # df2 = df
        print('df\n', df2.head())
        print('df shape: ', df2.shape)
        # GroundTruthModel
        # comparison_df = pd.read_csv('../csv_results_table/full300_5_GS.csv')
        # comparison_df = pd.read_csv('../csv_results_table/small100_20_GS.csv')
        # comparison_df = pd.read_csv('../csv_results_table/verysmall30_40_GS.csv')

        # Single apps
        try:
            comparison_df = pd.read_csv(f'..\\csv_results_table\\full300_5_{app_name}.csv')
        except:
            columns = ['Embedding', 'Classifier', 'Precision', 'Accuracy', 'Recall', 'F1 (clone as positive class)',
                       'F1 (distinct as positive class)']
            comparison_df = pd.DataFrame(columns=columns)

        print('comp_df\n', comparison_df.head())
        for arg in args:
            X = []
            y = []
            pbar = tqdm(total=df2.shape[0])
            print(f'app name:{app_name}')
            print(arg, 'first loop')
            for inpt in df2.iterrows():
                ret_val = calculate_states_similarity(inpt, arg)
                if ret_val is None:
                    continue
                X.append(ret_val[0])
                y.append(ret_val[1])
                pbar.update()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=16)

            names = [
                "Nearest Neighbors",
                "Linear SVM",
                "RBF SVM",
                # "Gaussian Process",
                "Decision Tree",
                "Random Forest",
                "Neural Net",
                "AdaBoost",
                "Naive Bayes",
                "QDA",
            ]

            classifiers = [
                KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                # GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),
            ]
            print(arg, 'second loop')
            for name, model in zip(names, classifiers):
                # model = KNeighborsClassifier(n_neighbors=3)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                # compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                # positive = equal class (1)
                f1 = f1_score(y_test, y_pred, pos_label=1)
                # psoitive = distinct class (0)
                f2 = f1_score(y_test, y_pred, pos_label=0)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                print(f'{name}, accuracy:{accuracy},f1: {f1},f2: {f2}, precision: {precision}, recall: {recall}')

                a = ''
                if arg == 'content':
                    a = 'Content only'
                elif arg == 'tags':
                    a = 'Tags only'
                elif arg == 'content_tags':
                    a = 'Content and tags'
                elif arg == 'all':
                    a = "Ensemble"
                else:
                    print('nope')
                d1 = pd.DataFrame(
                    {'Embedding': [a], 'Classifier': [name], 'Precision': [precision], 'Accuracy': [accuracy],
                     'Recall': [recall],
                     'F1 (clone as positive class)': [f1], 'F1 (distinct as positive class)': [f2]})
                comparison_df = pd.concat([comparison_df, d1])
            # comparison_df.to_csv('../csv_results_table/full300_5_GS.csv',index=False)
            # comparison_df.to_csv('../csv_results_table/small100_20.csv',index=False)
            # comparison_df.to_csv('../csv_results_table/verysmall30_40.csv',index=False)

            # app scores
            comparison_df.to_csv(f'..\\csv_results_table\\full300_5_{app_name}.csv', index=False)

    #     # tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    #     # print(tn, fp, fn, tp)
