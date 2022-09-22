from utils import *
import numpy as np


def kp_detector_sift(img, n_features=0, contrast_threshold=0.04, edge_threshold=10, if_plot=False):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold,
                                       edgeThreshold=edge_threshold)
    kp, des = sift.detectAndCompute(img, None)  # keypoints and descriptors
    if if_plot:
        kp_img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('kp_img', kp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # print(len(des))
    return des


def group_features(data_dict, obj_num_max, img_num_max, img_name_pre='', if_plot=False):
    stacked_dict = {}
    num_features = []
    for obj_num in range(1, obj_num_max+1, 1):
        des_object = []
        for img_num in range(1, img_num_max+1, 1):
            obj_key = f'{obj_num}_{img_name_pre}{img_num}'
            if obj_key in data_dict.keys():
                server_img = data_dict[obj_key]
                # if_plot = True if img_num == 1 and obj_num <= 2 else False
                server_des = kp_detector_sift(server_img, n_features=5000, if_plot=if_plot)
                des_object.append(server_des)
        feature_object = np.concatenate(des_object, axis=0)
        num_features.append(feature_object.shape[0])
        stacked_dict[obj_num] = feature_object
    return stacked_dict, num_features


def main():
    client_dict, server_dict = read_images('Data2')

    # part2-a
    stacked_server_dict, num_server_features = group_features(server_dict, obj_num_max=50, img_num_max=5,
                                                              img_name_pre='')
    print(sum(num_server_features)/len(num_server_features))  # 14864.0

    # part2-b
    stacked_client_dict, num_client_features = group_features(client_dict, obj_num_max=50, img_num_max=1,
                                                              img_name_pre='t')
    print(sum(num_client_features) / len(num_client_features))  # 4984.46


if __name__ == '__main__':
    main()


