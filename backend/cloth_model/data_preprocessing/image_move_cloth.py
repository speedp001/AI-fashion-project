####### org_dataset을 train, valid dataset으로 분할하는 코드

import os
import shutil

org_data_folder_path = "./cloth_org_dataset/"

dataset_folder_path = "./cloth_dataset/"

#train or val folder path
train_folder_path = os.path.join(dataset_folder_path, "train")
val_folder_path = os.path.join(dataset_folder_path, "valid")


#train, val, test folder create
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)

subdirectorys = [subdir for subdir in os.listdir(org_data_folder_path) if os.path.isdir(os.path.join(org_data_folder_path, subdir)) and subdir != ".DS_Store"]
# mac환경에서 생기는 .DS_Store과 디렉토리여부를 판단하여 조건에 맞는 하위 디렉토리만 반환(리스트 형태)
# print(subdirectorys)
# ['both', 'fw', 'ss']

for subdirectory in subdirectorys :
    
    org_folder_path = os.path.join(org_data_folder_path, subdirectory)
    # print(org_folder_path)
    
    """
    ./cloth_org_dataset/both
    ./cloth_org_dataset/fw
    ./cloth_org_dataset/ss
    """
    
    #label folder path
    train_label_folder_path = os.path.join(train_folder_path, subdirectory)
    val_label_folder_path = os.path.join(val_folder_path, subdirectory)
    
    labels = [label for label in os.listdir(org_folder_path) if os.path.isdir(os.path.join(org_folder_path, label)) and label != ".DS_Store"]
    # both, ss, fw의 하위 디렉토리의 목록들을 순회하여 labels에 리스트 요소로 담는다.
    # print(labels)
    
    # """
    # ['skirt', 'denim_pants', 'sport_pants', 'sweatshirt', 'long-sleeved_T-shirt', 'onepiece', 'blouse', 'long_pants', 'hooded_zipup', 'cargo_pants', 'leggings', 'cardigan', 'hooded', 'shirt']
    # ['jacket', 'coat', 'knitwear', 'padding']
    # ['short_pants', 'polo_shirt', 'short-sleeved_T-shirt']
    # """
    
    for label in labels :
        
        # 기존 데이터셋의 전체 경로
        org_folder_full_path = os.path.join(org_folder_path, label)
        # print(org_folder_full_path)
        
        # 이미지들을 list화
        images = os.listdir(org_folder_full_path)
        
        #label folder path
        train_label_folder_path_full = os.path.join(train_label_folder_path, label)
        val_label_folder_path_full = os.path.join(val_label_folder_path, label)
        
        os.makedirs(train_label_folder_path_full, exist_ok=True)
        os.makedirs(val_label_folder_path_full, exist_ok=True)
    
    
    
        # train_split_index는 90% 지점
        # Split the images into train, val, test sets
        train_split_index = int(len(images) * 0.9)

        # Move images to the train folder
        for image in images[:train_split_index]:
            src_path = os.path.join(org_folder_full_path, image)
            dst_path = os.path.join(train_label_folder_path_full, image)
            shutil.copyfile(src_path, dst_path)

        # Move images to the validation folder
        for image in images[train_split_index:]:
            src_path = os.path.join(org_folder_full_path, image)
            dst_path = os.path.join(val_label_folder_path_full, image)
            shutil.copyfile(src_path, dst_path)

