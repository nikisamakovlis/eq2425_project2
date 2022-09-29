from tree_construction import *
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, top_k_accuracy_score


def compute_tfidf_query(object_idxs, prediction_names, data, idf_database, prediction_set_database):
    n_objects = max(object_idxs)+1  # total number of objects: 50
    num_features_each_object = data.shape[1]  # 2000

    f_list_all = []
    for obj_idx in range(n_objects):
        start_idx = (obj_idx)*num_features_each_object
        end_idx = (obj_idx+1)*num_features_each_object
        obj_predictions = prediction_names[start_idx:end_idx]
        f_list = []
        for pred in prediction_set_database:
            count = obj_predictions.count(pred)
            f_list.append(count)
        f_list_all.append(f_list)
    f = np.array(f_list_all)  # number of occurrences of word vi in object oj, shape: (50, 64)
    F = np.array(list(map(lambda x: x.shape[0], data))).reshape(-1, 1)  # total number of visual words in each object with value 2000 repeated 50 times, shape (50,1)
    tf = f/F  # (50, 64)
    # print(tf.shape)  # (50, 15165)
    # print(idf_database.shape)  # (50, 15165)
    # print((tf*idf_database).shape)
    return tf*idf_database

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def recall_rate(w_d, w_q):
    q_list = range(w_q.shape[0])
    top5_list = []
    top1_list = []
    # top5_distances_list = []
    for idx_q in range(w_q.shape[0]):
        q = w_q[idx_q]
        # q shape (16)
        q = np.repeat(q.reshape(-1, 1), w_d.shape[0], axis=1).T  # q shape (50, 16)
        s = np.absolute(w_d-q)
        sum_s = np.sum(s, 1)  # sum_s shape 50

        sorted_indices = np.argsort(sum_s)
        one_min_idx = sorted_indices[:1][0]  # take the top1 item
        five_min_idx = sorted_indices[:5]

        onehot = np.zeros((5, w_q.shape[0]))
        onehot[np.arange(len(five_min_idx)), five_min_idx] = 1  # get one-hot
        five_min_idx_multihot = np.sum(onehot, 0)  # convert one-hot to multi-hot
        top1_list.append(one_min_idx)
        top5_list.append(five_min_idx_multihot)

        # top5_distance = []
        # for idx in five_min_idx:
        #     top5_distance.append(sum_s[idx])
        # top5_distances_list.append(top5_distance)

    top1_recall = recall_score(q_list, top1_list, average='micro')
    top1_acc = accuracy_score(q_list, top1_list)
    top5_acc = top_k_accuracy_score(q_list, np.stack(top5_list, axis=0), k=5)

    return top1_recall, top1_acc, top5_acc


def main():
    stacked_client_dict = np.load("stacked_client_dict.npy", allow_pickle=True)
    stacked_server_dict = np.load("stacked_server_dict.npy", allow_pickle=True)

    # num_features_each_object = 2000
    # b = 5
    # depth = 7
    # (a) Build three vocabulary trees by varying the settings as: b = 4, depth = 3;
    # b = 4, depth = 5
    # and b = 5,depth = 7.
    # For these three trees, report the average top-1 and top-5 recall rates over 50 objects
    num_features_each_object_database = 4000
    config_list = [[num_features_each_object_database,4,3], [num_features_each_object_database,4,5], [num_features_each_object_database,5,7],
                   [int(num_features_each_object_database*0.9),5,7], [int(num_features_each_object_database*0.7),5,7], [int(num_features_each_object_database*0.5),5,7]]
    for config in config_list:
        num_features_each_object_query, b, depth = config

        print(f'Starting the query for a tree with num_features_each_object_query: {num_features_each_object_query}, b:{b}, depth: {depth}')
        server_features = combine_feature(stacked_server_dict, num_features_each_object_database)  # (50, 2000, 128)
        client_features = combine_feature(stacked_client_dict, num_features_each_object_query)  # (50, 2000, 128)
        print(f'Combining server and client features: finished - with server/client shape: {server_features.shape} and {client_features.shape}')

        # build the tree
        model_list, name_list = hi_kmeans(server_features, b, depth)
        print(f'Building the Hi_KMeans: finished - with the number of models/names: {len(model_list)} and {len(name_list)}')

        # get tf-idf from the database
        object_idxs_all_database, prediction_names_all_database = get_prediction_names_all(server_features, model_list, name_list, depth)  # (100000, 3)
        w_d, idf_database = compute_tfidf(object_idxs_all_database, prediction_names_all_database, server_features)
        print(f'Calculating tf-idf from the database: finished - with the number of leaf nodes: {len(set(prediction_names_all_database))}')

        # get tf-idf from from the query
        object_idxs_all, prediction_names_all = get_prediction_names_all(client_features, model_list, name_list, depth)  # (100000, 3)
        w_q = compute_tfidf_query(object_idxs_all, prediction_names_all, client_features, idf_database, sorted(set(prediction_names_all_database)))
        print(f'Calculating tf-idf from the query: finished - with the number of unique predictions: {len(set(prediction_names_all))}')

        # get recall rate
        top1_recall, top1_acc, top5_acc = recall_rate(w_d, w_q)
        print(f'top1_recall: {top1_recall}')
        print(f'top1_acc: {top1_acc}')
        print(f'top5_acc: {top5_acc}', '\n', '\n', '\n')


if __name__ == '__main__':
    main()
