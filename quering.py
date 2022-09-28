from utils import *
from feature_extraction import *
from tree_construction import *
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np


def compute_tfidf_query(object_idxs, prediction_names, data, idf_database):
    n_objects = max(object_idxs)+1  # total number of objects: 50
    unique_predictions = sorted(set(prediction_names))

    num_features_each_object = data.shape[1]  # 2000
    f_list_all = []
    for obj_idx in range(n_objects):
        start_idx = (obj_idx)*num_features_each_object
        end_idx = (obj_idx+1)*num_features_each_object
        obj_predictions = prediction_names[start_idx:end_idx]
        f_list = []
        for pred in unique_predictions:
            count = obj_predictions.count(pred)
            f_list.append(count)
        f_list_all.append(f_list)
    f = np.array(f_list_all)  # number of occurrences of word vi in object oj, shape: (50, 64)
    F = np.array(list(map(lambda x: x.shape[0], data))).reshape(-1,1)  # total number of visual words in each object with value 2000 repeated 50 times, shape (50,1)
    tf = f/F  # (50, 64)
    print(tf.shape)
    print(idf_database.shape)
    print((tf*idf_database).shape)

    return tf*idf_database


def recall_rate(w_d, w_q):
    for q in w_q:
        # q shape (16)
        q = np.repeat(q.reshape(-1,1), w_d.shape[0], axis=1).T # q shape (50, 16)
        s = np.absolute(w_d-q)
        sum_s = np.sum(s, 1)
        print(sum_s)


def main():
    stacked_client_dict = np.load("stacked_client_dict.npy", allow_pickle=True)
    stacked_server_dict = np.load("stacked_server_dict.npy", allow_pickle=True)

    num_features_each_object = 2000
    b = 4
    depth = 3

    server_features = combine_feature(stacked_server_dict, num_features_each_object)  # (50, 2000, 128)
    client_features = combine_feature(stacked_client_dict, num_features_each_object)  # (50, 2000, 128)
    print(client_features.shape)
    # print(client_features)

    # build the tree
    model_list, name_list = hi_kmeans(server_features, b, depth)
    print(len(model_list), len(name_list), len(set(name_list)), name_list)

    # get idf from the database
    object_idxs_all_database, prediction_names_all_database = get_prediction_names_all(server_features, model_list, name_list, depth)  # (100000, 3)
    w_d, idf_database = compute_tfidf(object_idxs_all_database, prediction_names_all_database, server_features)

    # get tf from the query
    object_idxs_all, prediction_names_all = get_prediction_names_all(client_features, model_list, name_list, depth)  # (100000, 3)
    print(len(object_idxs_all))  # 100000 items with object indices
    print(len(prediction_names_all))  # # 100000 items with model names, e.g., 'm_0_1_2_0', 'm_0_3_1_3'
    print(prediction_names_all[:10])

    w_q = compute_tfidf_query(object_idxs_all, prediction_names_all, client_features, idf_database)
    recall_rate(w_d, w_q)








if __name__ == '__main__':
    main()
