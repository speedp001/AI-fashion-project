import os
import cv2
import glob
from torch.utils.data import Dataset


# 커스텀 데이터 셋 정의
class MyDataset(Dataset):
    
    def __init__(self, directory_data, transforms=None):
        self.directory_data = glob.glob(os.path.join(directory_data, "*", "*", "*.png"))
        self.transforms = transforms
        self.label_dictionary = self.create_label_dict()
        
    # 라벨 딕셔너리 생성 함수
    def create_label_dict(self):
        
        label_dictionary = {}
        
        for filepath in self.directory_data:
            label = os.path.basename(os.path.dirname(os.path.dirname(filepath))) + "_" + os.path.basename(os.path.dirname(filepath))
            
            if label not in label_dictionary:
                label_dictionary[label] = len(label_dictionary)
        
        # sorted() 함수를 사용하면 기본적으로 문자열 키(key)를 알파벳 순서대로 정렬
        sorted_keys = sorted(label_dictionary.keys())
        label_dictionary = {key: idx for idx, key in enumerate(sorted_keys)}

        return label_dictionary
    
    
    
    def __getitem__(self, item) :
        
        # print(self.label_dictionary)
        
        image_filepath = self.directory_data[item]
        
        img = cv2.imread(image_filepath)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = label = os.path.basename(os.path.dirname(os.path.dirname(image_filepath))) + "_" + os.path.basename(os.path.dirname(image_filepath))
        # print(label)
        
        label_idx = self.label_dictionary[label]
        # print(label, label_idx)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
            
        return image, label_idx
        
        
     
    def __len__(self):
        return len(self.directory_data)
    
    

# # test run
# test = MyDataset("./backend/cloth_model/cloth_dataset/valid/", transforms=None)

# # for i in test:
# #     print(i)

# print(test.label_dictionary)


    

