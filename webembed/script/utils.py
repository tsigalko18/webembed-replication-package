import json
import pickle
from os.path import join

import gensim
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from bs4.element import NavigableString, Comment

from abstract_function_python.main import doc2vec_model_content_tags, doc2vec_model_tags, doc2vec_model_content

all_html_path = 'D:\\doc2vec\\dataset\\all_html'


def compute_embeddings(inp, trained_models, s='all', compute_similarity=False):
    try:
        (_, row) = inp
        final_sim = None
        if s == 'all':
            # create content embeddings
            metadata_content1 = json.loads(row['state1_content'])
            metadata_content2 = json.loads(row['state2_content'])
            emb1_content = load_and_create_embedding(metadata_content1, trained_models[0])
            emb2_content = load_and_create_embedding(metadata_content2, trained_models[0])

            # create tags embeddings
            metadata_tags1 = json.loads(row['state1_tags'])
            metadata_tags2 = json.loads(row['state2_tags'])
            emb1_tags = load_and_create_embedding(metadata_tags1, trained_models[1])
            emb2_tags = load_and_create_embedding(metadata_tags2, trained_models[1])

            # create content + tags embeddings
            metadata_content_tags1 = json.loads(row['state1_content_tags'])
            metadata_content_tags2 = json.loads(row['state2_content_tags'])
            emb1_content_tags = load_and_create_embedding(metadata_content_tags1, trained_models[2])
            emb2_content_tags = load_and_create_embedding(metadata_content_tags2, trained_models[2])

            if compute_similarity:
                # calculate the similarity between embeddings
                emb_page1 = np.hstack((emb1_content, emb1_tags, emb1_content_tags))
                emb_page2 = np.hstack((emb2_content, emb2_tags, emb2_content_tags))
                cos_sim_all = cosine_similarity(emb_page1, emb_page2)
                final_sim = np.array([cos_sim_all[0, 0]])
            else:
                print("feature not yet implemented")
                exit()

            return final_sim, row['answer']

        if s == 'content':
            metadata_content1 = json.loads(row['state1_content'])
            metadata_content2 = json.loads(row['state2_content'])

            # create content embeddings
            emb1_content = load_and_create_embedding(metadata_content1, trained_models[0])
            emb2_content = load_and_create_embedding(metadata_content2, trained_models[0])

            if compute_similarity:
                # calculate the similarity between embeddings
                cos_sim_content = cosine_similarity(emb1_content, emb2_content)
                final_sim = np.array([cos_sim_content[0, 0]])
            else:
                print("feature not yet implemented")
                exit()

            return final_sim, row['answer']

        if s == 'tags':
            metadata_tags1 = json.loads(row['state1_tags'])
            metadata_tags2 = json.loads(row['state2_tags'])

            # create tags embeddings
            emb1_tags = load_and_create_embedding(metadata_tags1, trained_models[0])
            emb2_tags = load_and_create_embedding(metadata_tags2, trained_models[0])

            if compute_similarity:
                # calculate the similarity between embeddings
                cos_sim_tags = cosine_similarity(emb1_tags, emb2_tags)
                final_sim = np.array([cos_sim_tags[0, 0]])
            else:
                print("feature not yet implemented")
                exit()

            return final_sim, row['answer']

        if s == 'content_tags':
            metadata_content_tags1 = json.loads(row['state1_content_tags'])
            metadata_content_tags2 = json.loads(row['state2_content_tags'])

            # create content+tags embeddings
            emb1_content_tags = load_and_create_embedding(metadata_content_tags1, trained_models[0])
            emb2_content_tags = load_and_create_embedding(metadata_content_tags2, trained_models[0])

            if compute_similarity:
                # calculate the similarity between embeddings
                cos_sim_content_tags = cosine_similarity(emb1_content_tags, emb2_content_tags)
                final_sim = np.array([cos_sim_content_tags[0, 0]])
            else:
                print("feature not yet implemented")
                exit()

            return final_sim, row['answer']

    except Exception as e:
        print(e, row)
        return None


def load_and_create_embedding(metadata, model):
    with open(join(all_html_path, metadata['path'])) as fp:
        data = json.load(fp)
    return model.infer_vector(data).reshape(1, -1)
