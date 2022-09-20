import os
import cv2


def read_images(root_folder):
    client_dict = {}
    server_dict = {}
    for root, folder, files in os.walk(os.path.join(root_folder, 'client')):
        for name in files:
            img_path = os.path.join(root, name)
            if img_path.endswith('.JPG'):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
                obj_num = name.split('.JPG')[0].split('obj')[-1]
                client_dict[obj_num] = img

    for root, folder, files in os.walk(os.path.join(root_folder, 'server')):
        for name in files:
            img_path = os.path.join(root, name)
            if img_path.endswith('.JPG'):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
                obj_num = name.split('.JPG')[0].split('obj')[-1]
                server_dict[obj_num] = img
    print(client_dict.keys(), server_dict.keys())
    return client_dict, server_dict
