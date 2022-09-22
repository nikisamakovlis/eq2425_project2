
def combine_feature(feature_dict):
    combined_features = []
    for key in feature_dict.keys():
        feature_per_obj = feature_dict[key]
        combined_features.append(feature_per_obj)
    combined_features = np.concatenate(combined_features, axis=0)
    return combined_features # dimension: total_num_features, 128

def hi_kmeans(data, b, depth):
    pass


def main():
    # To build the Vocabulary Tree
    client_dict, server_dict = read_images('Data2')
    stacked_server_dict, num_server_features = group_features(server_dict, obj_num_max=50, img_num_max=5,
                                                              img_name_pre='')
    stacked_client_dict, num_client_features = group_features(client_dict, obj_num_max=50, img_num_max=1,
                                                              img_name_pre='t')

    server_features = combine_feature(stacked_server_dict)
    print(server_features.shape)


    # 3a) In order to query by SIFT features, what information should be stored in each node in the vocabulary tree?
    # internal nodes: clustering center

    # (b) Based on the TF-IDF score, decide what additional information you need to store in
    # the leaf nodes of the tree (Hint: The leaf nodes can be seen as visual vocabularies).
    # leaves: local (SIFT) features indexed, The leaf nodes of this tree contain a "bag" of sift descriptors




    pass


if __name__ == '__main__':
    main()