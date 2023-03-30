# # abstraction function that computes similarity
# import pickle
#
# from abstract_function_python.main import FEATURE, CLASSIFIER_PATH
# from script.utils import get_distance_from_embeddings
#
#
# def word2vec_equals(obj1, obj2):
#     dist = get_distance_from_embeddings(obj1, obj2, feature=FEATURE)
#     dist = dist.reshape(1, -1)
#
#     model = pickle.load(open(CLASSIFIER_PATH, 'rb'))
#     word2vec = model.predict(dist)
#     return word2vec
