
def hi_kmeans(data, b, depth):
    pass


def main():
    # 3a) In order to query by SIFT features, what information should be stored in each node in the vocabulary tree?
    # internal nodes: clustering center

    # (b) Based on the TF-IDF score, decide what additional information you need to store in
    # the leaf nodes of the tree (Hint: The leaf nodes can be seen as visual vocabularies).
    # leaves: local (SIFT) features indexed, The leaf nodes of this tree contain a "bag" of sift descriptors

    pass


if __name__ == '__main__':
    main()