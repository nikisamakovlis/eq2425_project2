from utils import *


def kp_detector_sift(img, n_features=0, contrast_threshold=0.185, edge_threshold=145):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold,
                                       edgeThreshold=edge_threshold)
    kp, des = sift.detectAndCompute(img, None)  # keypoints and descriptors
    print(len(des))
    return des


def main():
    client_dict, server_dict = read_images('Data2')
    stacked_dict = {}
    for obj_num in range(1,51,1):
        # destructor
        for img_num in range(1,4,1):
            obj_key =f'{obj_num}_{img_num}'
            server_img = server_dict[obj_key]
            server_des = kp_detector_sift(server_img, contrast_threshold=0.09, edge_threshold=15)












if __name__ == '__main__':
    main()


