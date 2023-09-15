####### app.py에서 호출할 Item, Color, Style 분류기 파일

import cv2
import json
import torch
import numpy as np
import torch.nn as nn
from rembg import remove
import torchvision.models as models
import albumentations as A

from reference import style_dict, color_dict
from sklearn.cluster import KMeans
from albumentations.pytorch import ToTensorV2






####### Item 및 Style 분류기 
class ItemClassifier:
    # 인스턴스 변수
    def __init__(self):
        self.device = self.gpu()
        self.model = self.initialize_model()
        # self.item_dictionary = self.create_item_dictionary()
        self.style_dictionary = style_dict

    # gpu 환경설정
    def gpu(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using {device}")
            
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using {device}")
    
        else: 
            device = torch.device("cpu")
            print(f"using {device}")
            
        return device
            
    # 모델 정의 함수
    def initialize_model(self):
        model = models.efficientnet_b0()
        model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        model.classifier[1] = nn.Linear(1280, out_features=21)

        checkpoint = torch.load("./cloth_model/weight/model_best.pt", map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        
        return model
    
    
    # 사용자 이미지 배경제거 함수
    def rembg(self, img):
        # 이미지 데이터를 바이너리에서 이미지로 디코딩
        org_image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

        # 배경을 흰색으로 변경 -> bgcolor=(b, g, r, a)
        color_rembg_img = remove(org_image, only_mask=True)
        item_rembg_img = remove(org_image, bgcolor=(255, 255, 255, 255))

        return item_rembg_img, color_rembg_img
    
    
    # 이미지 전처리 함수
    def image_preprocessing(self, image_data):
        image_transform = A.Compose([
            A.Resize(width=480, height=640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
        if image_transform is not None:
            img = image_transform(image=image_data)['image']

        img = img.unsqueeze(0).to(self.device)
        return img

    # 아이템 예측 함수
    def item_predict(self, image_data):
        image_tensor = self.image_preprocessing(image_data)

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(image_tensor)
            _, predicted_label_idx = outputs.max(1)
            # predicted_item = self.item_dictionary[predicted_label_idx.item()]
            
            # 정수를 스트링으로 변환하고 두 자리로 포맷팅
            predicted_item_str = f'{predicted_label_idx.item():02}'

        return predicted_item_str

    # 스타일 예측 함수
    def style_predict(self, style):
        predicted_style_str = self.style_dictionary[style]
        
        return predicted_style_str
    






####### Color 분류기
class ColorClassifier:
    # 인스턴스 변수
    def __init__(self):
        # Read JSON file
        with open("./hex_map.json", "r") as j:
            self.hex_dict = json.load(j)
        
        self.color_dictionary = color_dict
    
    # rgb -> hex code
    def rgb_to_hex(self, rgb):  # convert rgb array to hex code
        return "{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    # pxs 변환
    def cvt216(self, pxs):  # convert rgb into the value in 216
        st = [0, 51, 102, 153, 204, 255]
        p0 = min(st, key=lambda x: abs(x - pxs[0]))
        p1 = min(st, key=lambda x: abs(x - pxs[1]))
        p2 = min(st, key=lambda x: abs(x - pxs[2]))

        return np.array((p0, p1, p2))

    # Kmeans 적용 함수
    def dom_with_Kmeans(self, img_list, k=3):  # use kmeans
        pixels = img_list

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        """
        ex) colors -> 세 개의 중심 색상
        [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255]     # Blue
        ]
        """
        
        # Get the labels (which cluster each pfixel belongs to)
        labels = kmeans.labels_
        """
        ex) labels -> 클러스터링 한 3개의 중심색상으로 각 픽셀 라벨정보
        [0, 0, 1, 2, 1, 2]
        """
        
        # Count the frequency of each label
        label_counts = np.bincount(labels)
        
        labels = np.argsort(label_counts)[::-1]

        dom_counts = [label_counts[i] for i in labels[:3]]
        total = sum(dom_counts)
        # Each cluster's rate
        dom_counts_rate = [i / total for i in dom_counts]
        
        # Top3 colors
        dom_colors = [colors[i] for i in labels[:3]]

        return dom_colors, dom_counts_rate

    # 색상 예측 함수
    def color_predict(self, image_data, mask_data): # image_data 에 mask img만 들어왔음 acensia
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        height, width, _ = image_data.shape
        print(image_data.shape)
        cropped_list = np.array(
            [
                image_data[i][j]
                for i in range(height)
                for j in range(width)
                if mask_data[i][j] > 100
            ]
        )
        
        colors, counts_rate = self.dom_with_Kmeans(cropped_list)
        fst, snd, trd = colors[:3]
        # print(counts_rate)
        print(fst)
        # print(snd)
        # print(trd)

        fst_cvt216 = self.cvt216(fst)
        snd_cvt216 = self.cvt216(snd)
        trd_cvt216 = self.cvt216(trd)
        p1, p2, p3 = (
            self.hex_dict[self.rgb_to_hex(fst_cvt216)],
            self.hex_dict[self.rgb_to_hex(snd_cvt216)],
            self.hex_dict[self.rgb_to_hex(trd_cvt216)],
        )
        
        predicted_color_str = self.color_dictionary[p1]
        print(self.rgb_to_hex(fst_cvt216))
        return predicted_color_str
