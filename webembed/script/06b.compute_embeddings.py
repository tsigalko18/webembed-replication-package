import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm

from script.utils import compute_embeddings

if __name__ == '__main__':
    '''
    Compute embeddings for the sets DS and/or SS
    '''

    trained_models_path = '../trained_model/'
    vector_size = ['modelsize100']
    epochs = 31  # this is the best model in terms of accuracy from 07.classifier_scores_DS
    compute_similarity = True

    # embedding_type = ['content', 'tags', 'content_tags', 'all]
    embedding_type = ['all']

    dataset = ["SS"]  # "DS", "SS"

    df = None
    if "DS" in dataset:
        df = pd.read_csv('DS_threshold_set.csv')

        for emb in embedding_type:
            print("computing embedding: %s\tsimilarity: %s" % (emb, str(compute_similarity)))

            models = []
            if emb == 'all':
                # load Doc2Vec content model
                model_content = Doc2Vec.load(trained_models_path +
                                             'DS_' +
                                             'content' +
                                             '_' +
                                             vector_size[0] +
                                             'epoch' +
                                             str(epochs) +
                                             '.doc2vec.model')
                # load Doc2Vec tags model
                model_tags = Doc2Vec.load(trained_models_path +
                                          'DS_' +
                                          'tags' +
                                          '_' +
                                          vector_size[0] +
                                          'epoch' +
                                          str(epochs) +
                                          '.doc2vec.model')
                # load Doc2Vec content + tags model
                model_content_tags = Doc2Vec.load(trained_models_path +
                                                  'DS_' +
                                                  'tags' +
                                                  '_' +
                                                  vector_size[0] +
                                                  'epoch' +
                                                  str(epochs) +
                                                  '.doc2vec.model')

                models.append(model_content)
                models.append(model_tags)
                models.append(model_content_tags)
            else:
                model = Doc2Vec.load(trained_models_path +
                                     'DS_' +
                                     emb +
                                     '_' +
                                     vector_size[0] +
                                     'epoch' +
                                     str(epochs) +
                                     '.doc2vec.model')
                models.append(model)

            embeddings = []
            pbar = tqdm(total=df.shape[0])
            for inp in df.iterrows():
                ret_val = compute_embeddings(inp, models, emb, compute_similarity=compute_similarity)
                if ret_val is None:
                    continue
                embeddings.append(ret_val[0][0])
                pbar.update()

            df['doc2vec_distance_' + emb] = embeddings
            df.to_csv('DS_threshold_set.csv')

    elif "SS" in dataset:
        df = pd.read_csv('SS_threshold_set.csv')
        apps = ['pagekit', 'petclinic', 'phoenix', 'ppma']
        # apps = ['addressbook']
        for app in apps:
            df_temp = df[df['appname'] == app]
            print("app %s" % app)
            for emb in embedding_type:
                print("computing embedding: %s\tsimilarity: %s" % (emb, str(compute_similarity)))

                models = []
                if emb == 'all':
                    # load Doc2Vec content model
                    model_content = Doc2Vec.load(trained_models_path +
                                                 'DS_' +
                                                 'content' +
                                                 '_' +
                                                 vector_size[0] +
                                                 'epoch' +
                                                 str(epochs) +
                                                 '.doc2vec.model')
                    # load Doc2Vec tags model
                    model_tags = Doc2Vec.load(trained_models_path +
                                              'DS_' +
                                              'tags' +
                                              '_' +
                                              vector_size[0] +
                                              'epoch' +
                                              str(epochs) +
                                              '.doc2vec.model')
                    # load Doc2Vec content + tags model
                    model_content_tags = Doc2Vec.load(trained_models_path +
                                                      'DS_' +
                                                      'tags' +
                                                      '_' +
                                                      vector_size[0] +
                                                      'epoch' +
                                                      str(epochs) +
                                                      '.doc2vec.model')

                    models.append(model_content)
                    models.append(model_tags)
                    models.append(model_content_tags)
                else:
                    model = Doc2Vec.load(trained_models_path +
                                         'DS_' +
                                         emb +
                                         '_' +
                                         vector_size[0] +
                                         'epoch' +
                                         str(epochs) +
                                         '.doc2vec.model')
                    models.append(model)

                embeddings = []
                pbar = tqdm(total=df_temp.shape[0])
                for inp in df_temp.iterrows():
                    ret_val = compute_embeddings(inp, models, emb, compute_similarity=compute_similarity)
                    if ret_val is None:
                        continue
                    embeddings.append(ret_val[0][0])
                    pbar.update()

                with open(app + '_' + emb + '.npy', 'wb') as f:
                    np.save(f, np.array(embeddings))
                f.close()

                # df['doc2vec_distance_' + emb] = embeddings
                # if "DS" in dataset:
                #     df.to_csv('D:\\doc2vec\\dataset\\training_sets\\DS_threshold_set.csv')
                # elif "SS" in dataset:
                #     df.to_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')
