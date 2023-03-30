import csv
import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from numpy import arange, argmax
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from script.utils import compute_embeddings


def to_labels(pos_prob, threshold):
    return (pos_prob >= threshold).astype('int')


df = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\DS_threshold_set.csv')
df['answer'] = [int(row != 2) for row in df['HUMAN_CLASSIFICATION']]
df = df.dropna(subset=['state1_content', 'state2_content', 'state1_tags',
                       'state2_tags', 'state1_content_tags', 'state2_content_tags'])

if __name__ == '__main__':

    trained_models_path = 'D:\\doc2vec\\trained_model\\'
    # embedding_type = ['content', 'tags', 'content_tags']
    embedding_type = ['content_tags']
    vector_size = ['modelsize500']
    # epochs = range(1, 51)
    epochs = [50]

    for ep in epochs:
        for emb in embedding_type:
            for vs in vector_size:

                if not os.path.exists(r'..\\csv_results_table\\' + 'DS_' + vs + '.csv'):
                    header = ['Model', 'Embedding', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1']
                    with open('..\\csv_results_table\\' + 'DS_' + vs + '.csv', 'w', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        # write the header
                        writer.writerow(header)

                print("epoch: %s\tembedding: %s" % (str(ep), emb))
                name = trained_models_path + 'DS_' + emb + '_' + vs + 'epoch' + str(ep) + '.doc2vec.model'
                model = Doc2Vec.load(name)

                comparison_df = pd.read_csv('..\\csv_results_table\\' + 'DS_' + vs + '.csv')

                X = []
                y = []
                pbar = tqdm(total=df.shape[0])
                for inp in df.iterrows():
                    ret_val = compute_embeddings(inp, model, emb, compute_similarity=True)
                    if ret_val is None:
                        continue
                    X.append(ret_val[0])
                    y.append(ret_val[1])
                    pbar.update()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                names = [
                    # "Threshold Classifier",
                    "Nearest Neighbors",
                    "SVM RBF",
                    "Decision Tree",
                    "Gaussian Naive Bayes",
                    "Random Forest",
                    "Ensemble",
                    "Neural Net"
                ]

                classifiers = [
                    KNeighborsClassifier(),
                    SVC(),
                    DecisionTreeClassifier(),
                    GaussianNB(),
                    RandomForestClassifier(),
                    VotingClassifier(estimators=[('knn', KNeighborsClassifier()),
                                                 ('svm', SVC()),
                                                 ('dt', DecisionTreeClassifier()),
                                                 ('gnb', GaussianNB()),
                                                 ('rf', RandomForestClassifier())]),
                    MLPClassifier()
                ]

                for name, model in zip(names, classifiers):

                    # fit the classifier
                    model = model.fit(X_train, y_train)

                    # predict the scores
                    y_pred = model.predict(X_test)

                    # compute metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    print(f'{name}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}')

                    a = ''
                    if emb == 'content':
                        a = 'Content only'
                    elif emb == 'tags':
                        a = 'Tags only'
                    elif emb == 'content_tags':
                        a = 'Content and tags'
                    elif emb == 'all':
                        a = "Ensemble"
                    else:
                        print('nope')

                    d1 = pd.DataFrame(
                        {'Model': ['DS_' + emb + '_' + vs + 'epoch' + str(ep)], 'Embedding': [a],
                         'Classifier': [name], 'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall],
                         'F1': [f1]})

                    comparison_df = pd.concat([comparison_df, d1])

                comparison_df.to_csv(
                    '..\\csv_results_table\\' + 'DS_' + vs + '.csv',
                    index=False)
