"""
This script generates the line sentences for the set all_html (DS + commoncrawls)
"""

from ast import excepthandler
from os.path import join
from os import mkdir, remove
import json
from random import random

from tqdm import tqdm

all_html_path = 'D:\\doc2vec\\dataset\\all_html'
sets_path = 'D:\\doc2vec\\dataset\\training_sets'
output_corpus = 'D:\\doc2vec\\dataset\\train_model_corpus'

content_model_train_set = join(sets_path, 'DS_content_model_train_set.json')
tags_model_train_set = join(sets_path, 'DS_tags_model_train_set.json')
content_tags_model_train_set = join(sets_path, 'DS_content_tags_model_train_set.json')

DELETE_OLD_FILE = False

try:
    mkdir(output_corpus)
except:
    pass


# TODO: what is this line sentence?
def generate_line_sentences(set_path, output):
    try:
        if DELETE_OLD_FILE:
            remove(output)
    except:
        pass
    with open(set_path, 'r') as fp:
        set_metadata = json.load(fp)
    # extract data
    print("generating line sentences for %s" % output)
    for app_name in tqdm(set_metadata):
        print("processing app %s" % app_name)
        for file_metadata in tqdm(set_metadata[app_name]):
            file_path = join(all_html_path, file_metadata['path'])
            # print("processing file %s" % file_path)
            try:
                with open(file_path, 'r', encoding="utf-8") as fp:
                    data = json.load(fp)
                # write each doc (content, tag, content + tag) as a single line
                with open(output, 'a+', encoding="utf-8") as fp:
                    fp.write(' '.join(data))
                    fp.write('\n')
            except Exception as e:
                print(app_name, file_path, e)
        fp.close()


content_model_train_set_path = join(output_corpus, 'DS_content_model_train_set.line_sentence')
tags_model_train_set_path = join(output_corpus, 'DS_tags_model_train_set.line_sentence')
content_tags_model_train_set_path = join(output_corpus, 'DS_content_tags_model_train_set.line_sentence')

# all docs
# generate_line_sentences(content_model_train_set, content_model_train_set_path)
# generate_line_sentences(tags_model_train_set, tags_model_train_set_path)
# generate_line_sentences(content_tags_model_train_set, content_tags_model_train_set_path)


# only pick some docs
def generate_small_corpus(probability, input_path, output_path):
    with open(input_path, 'r') as fp:
        try:
            # remove because later append to file _. avoid appending to old file content
            remove(output_path)
        except:
            pass
        print("generating corpus for %s with probability %s" % (output_path, str(probability)))
        for line in fp:
            if random() < probability:
                with open(output_path, 'a+') as op:
                    op.write(line)
                    op.write('\n')
        op.close()


# TODO: what is this probability line selection?
probability_line_selection = 0.05
generate_small_corpus(probability_line_selection, content_model_train_set_path,
                      join(output_corpus, 'DS_content_model_train_set_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, tags_model_train_set_path,
                      join(output_corpus, 'DS_tags_model_train_set_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, content_tags_model_train_set_path,
                      join(output_corpus, 'DS_content_tags_model_train_set_SMALL.line_sentence'))

probability_line_selection = 0.005
generate_small_corpus(probability_line_selection, content_model_train_set_path,
                      join(output_corpus, 'DS_content_model_train_set_VERY_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, tags_model_train_set_path,
                      join(output_corpus, 'DS_tags_model_train_set_VERY_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, content_tags_model_train_set_path,
                      join(output_corpus, 'DS_content_tags_model_train_set_VERY_SMALL.line_sentence'))
