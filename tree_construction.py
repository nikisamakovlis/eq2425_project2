from utils import *
from feature_extraction import *
from sklearn.cluster import KMeans

def get_prediction_each_feature(feature, name, maxdepth, model_list, name_list, prediction_list=[]):
    # data: (num_obj, num_features_each_object, 128)
    if maxdepth - (len(name.split('_'))-1) >= 1:
        thislevel_idx = name_list.index(name)
        thislevel_model = model_list[thislevel_idx]
        leaf_idx = thislevel_model.predict(feature)[0]
        nextlevel_name = name + f'_{leaf_idx}'
        prediction_list.append(nextlevel_name)
        get_prediction_each_feature(feature, nextlevel_name, maxdepth, model_list, name_list, prediction_list)
        return prediction_list
    else:
        prediction_list


def get_prediction_names_all(data, model_list, name_list, maxdepth):
    object_idxs_all = []
    prediction_names_all = []
    for obj_idx in range(data.shape[0]):
        obj = data[obj_idx,:,:]
        for obj_feature in obj:
            feature = obj_feature.reshape(1,-1)
            prediction_list = get_prediction_each_feature(feature, f'm_0', maxdepth, model_list, name_list, prediction_list=[])
            object_idxs_all.append(obj_idx)
            prediction_names_all.append(prediction_list[-1])  # only append the last item into the list

    return object_idxs_all, prediction_names_all


def recursive_kmeans(data, idx, b, depth, name, i=0, model_list=[], name_list=[]):
    if data[idx].shape[0] >= b and depth > 1:
        model = KMeans(n_clusters=b, random_state=0).fit(data[idx])
        name = name+f'_{i}'
        model_list.append(model)
        name_list.append(name)

        for i in range(b):
            idx_b = [idx[l] for l, ll in enumerate(model.labels_) if ll == i]
            recursive_kmeans(data, idx_b, b, depth-1, name, i, model_list, name_list)
        return model_list, name_list
    else:
        return model_list, name_list


def hi_kmeans(data, b, depth):
    data = np.concatenate(data, axis=0)
    return recursive_kmeans(data, np.arange(data.shape[0]), b, depth, name=f'm', i=0, model_list=[], name_list=[])


def combine_feature(feature_dict, num_features_each_object):
    combined_features = []
    for key in range(1, 51, 1):
        feature_per_obj = feature_dict.item().get(key)[:num_features_each_object,:]
        # feature_per_obj = feature_dict[key]
        combined_features.append(feature_per_obj)

    return np.array(combined_features)  # dimension: (num_obj, num_features_each_object, 128) --> (50, 2000, 128)


def main():
    # To build the Vocabulary Tree
    stacked_server_dict = np.load("stacked_server_dict.npy", allow_pickle=True)
    stacked_client_dict = np.load("stacked_client_dict.npy", allow_pickle=True)

    num_features_each_object = 2000
    b = 4
    depth = 4

    # for example, when b==4 and depth==4 --> 21 models and
    # name_list: ['m_0', --> depth 4
    # 'm_0_0',  --> depth 3
    #   'm_0_0_0', 'm_0_0_1', 'm_0_0_2', 'm_0_0_3', --> depth 2
    # 'm_0_1',
    #   'm_0_1_0', 'm_0_1_1', 'm_0_1_2', 'm_0_1_3',
    # 'm_0_2',
    #   'm_0_2_0', 'm_0_2_1', 'm_0_2_2', 'm_0_2_3',
    # 'm_0_3',
    #   'm_0_3_0', 'm_0_3_1', 'm_0_3_2', 'm_0_3_3']
    server_features = combine_feature(stacked_server_dict, num_features_each_object) # (50, 2000, 128)
    print(server_features.shape)
    model_list, name_list = hi_kmeans(server_features, b, depth)
    print(len(model_list), len(name_list), len(set(name_list)), name_list)

    object_idxs_all, prediction_names_all = get_prediction_names_all(server_features, model_list, name_list, depth)  # (100000, 3)
    print(len(object_idxs_all))  # 100000 items with indices
    print(len(prediction_names_all))  # # 100000 items with model names, e.g., 'm_0_1_2_0', 'm_0_3_1_3'
    print(prediction_names_all[:10])

    # 3a) In order to query by SIFT features, what information should be stored in each node in the vocabulary tree?
    # internal nodes: clustering center
    # all nodes except for leaves should store the knn model
    # when one server feature comes in, the path along down the tree until the leaf node + object index should be stored

    # (b) Based on the TF-IDF score, decide what additional information you need to store in
    # the leaf nodes of the tree (Hint: The leaf nodes can be seen as visual vocabularies).
    # leaves: local (SIFT) features indexed, The leaf nodes of this tree contain a "bag" of sift descriptors
    # TODO: TF-IDF score


if __name__ == '__main__':
    main()