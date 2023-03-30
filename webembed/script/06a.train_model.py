"""
This script trains all Doc2Vec models
"""

from multiprocessing import cpu_count
from os import mkdir

import gensim
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


class MonitorCallback(CallbackAny2Vec):
    def __init__(self, epochs, vector_size, output_name):
        self.pbar = tqdm(total=epochs)
        self.epoch_number = 0
        self.vector_size = vector_size
        self.output_name = output_name

    def on_epoch_end(self, model):
        self.epoch_number += 1
        if self.epoch_number % 1 == 0:
            model.save(self.output_name + f'size{self.vector_size}epoch{self.epoch_number}.doc2vec.model')
        self.pbar.update()


def train_model(train_model_set_path, output_trained_model_path, vector_size=100, epochs=10):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=4, epochs=epochs, workers=cpu_count())
    model.build_vocab(corpus_file=train_model_set_path)

    monitor = MonitorCallback(model.epochs, vector_size, output_trained_model_path)

    model.train(corpus_file=train_model_set_path, total_examples=model.corpus_count,
                total_words=model.corpus_total_words, epochs=model.epochs, callbacks=[monitor])

    model.save(output_trained_model_path + f'size{vector_size}epoch{epochs}.doc2vec.model')


try:
    mkdir('D:\\doc2vec\\trained_model')
except:
    pass

# print('Training Doc2Vec models on the entire corpus of DS + commoncrawl (content)')
# train_model(train_model_set_path='D:\\doc2vec\\dataset\\train_model_corpus\\DS_content_model_train_set.line_sentence',
#             output_trained_model_path='D:\\doc2vec\\trained_model\\DS_content_model')
#
# print('Training Doc2Vec models on the entire corpus of DS + commoncrawl (tags)')
# train_model(train_model_set_path='D:\\doc2vec\\dataset\\train_model_corpus\\DS_tags_model_train_set.line_sentence',
#             output_trained_model_path='D:\\doc2vec\\trained_model\\DS_tags_model')

# print('Training Doc2Vec models on the entire corpus of DS + commoncrawl (content + tags) vector_size=500, epochs=50')
# train_model(
#     train_model_set_path='D:\\doc2vec\\dataset\\train_model_corpus\\DS_content_tags_model_train_set.line_sentence',
#     output_trained_model_path='D:\\doc2vec\\trained_model\\DS_content_tags_model',
#     vector_size=500, epochs=50)

print('Training Doc2Vec models on the entire corpus of DS + commoncrawl (content + tags) vector_size=1000, epochs=50')
train_model(
    train_model_set_path='D:\\doc2vec\\dataset\\train_model_corpus\\DS_content_tags_model_train_set.line_sentence',
    output_trained_model_path='D:\\doc2vec\\trained_model\\DS_content_tags_model',
    vector_size=1000, epochs=50)
