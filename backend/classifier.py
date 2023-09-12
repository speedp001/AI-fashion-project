import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import albumentations as A

from sklearn.cluster import KMeans
from albumentations.pytorch import ToTensorV2







class ItemClassifier:
    
    def __init__(self):
        self.device = self.gpu()
        self.model = self.initialize_model()
        # self.item_dictionary = self.create_item_dictionary()
        self.style_dictionary = self.create_style_dictionary()


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
            

    def initialize_model(self):
        model = models.efficientnet_b0()
        model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        model.classifier[1] = nn.Linear(1280, out_features=21)

        checkpoint = torch.load("./cloth_model/weight/model_best.pt", map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        
        return model


    # def create_item_dictionary(self):
        
    #     # Create label_dictionary -> 참고용
    #     item_dict = {
    #         'both_blouse': 0,
    #         'both_cardigan': 1, 
    #         'both_cargo_pants': 2, 
    #         'both_denim_pants': 3, 
    #         'both_hooded': 4, 
    #         'both_hooded_zipup': 5, 
    #         'both_leggings': 6, 
    #         'both_long-sleeved_T-shirt': 7,
    #         'both_long_pants': 8,
    #         'both_onepiece': 9,
    #         'both_shirt': 10, 
    #         'both_skirt': 11, 
    #         'both_sport_pants': 12, 
    #         'both_sweatshirt': 13, 
    #         'fw_coat': 14, 
    #         'fw_jacket': 15, 
    #         'fw_knitwear': 16, 
    #         'fw_padding': 17, 
    #         'ss_polo_shirt': 18, 
    #         'ss_short-sleeved_T-shirt': 19, 
    #         'ss_short_pants': 20
    #         }
        
        
    #     reverse_item_dict = {v: k for k, v in item_dict.items()}
    #     return reverse_item_dict
    
    def create_style_dictionary(self):
        style_dict = {'casual': '0', 'dandy': 1, 'formal': '2', 'girlish': '3', 'gorpcore': '4', 'retro': '5', 'romantic': '6', 'sports': '7', 'street': '8'}
        
        return style_dict


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

    
    def style_predict(self): # argument 없음 acensia
        predicted_style_str = self.item_dictionary[self.style_dictionary.item()]
        
        return predicted_style_str
    







class ColorClassifier:
    
    def __init__(self):
        # Read JSON file
        with open("./hex_map.json", "r") as j:
            self.hex_dict = json.load(j)
        
        self.color_dictionary = self.create_color_dictionary()
        
    def create_color_dictionary(self):
        color_dict = {
            "beige": '00', 
            "black": '01', 
            "blue": '02', 
            "brown": '03', 
            "burgundy": '04', 
            "gray": '05', 
            "green": '06', 
            "khaki": '07', 
            "lightgreen": '08', 
            "lightpurple": '09', 
            "mint": '10', 
            "navy": '11', 
            "orange": '12', 
            "pink": '13', 
            "purple": '14', 
            "red": '15', 
            "skyblue": '16', 
            "teal": '17', 
            "white": '18', 
            "yellow": '19'
        }
        
        return color_dict
    

    def rgb_to_hex(self, rgb):  # convert rgb array to hex code
        return "{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


    def cvt216(self, pxs):  # convert rgb into the value in 216
        st = [0, 51, 102, 153, 204, 255]
        p0 = min(st, key=lambda x: abs(x - pxs[0]))
        p1 = min(st, key=lambda x: abs(x - pxs[1]))
        p2 = min(st, key=lambda x: abs(x - pxs[2]))

        return np.array((p0, p1, p2))


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
        dom_counts_rate = [i / total * 100 for i in dom_counts] # generator * int 안된다는 오류 acensia
        
        # Top3 colors
        dom_colors = [colors[i] for i in labels[:3]]

        return dom_colors, dom_counts_rate


    def color_predict(self, image_data, mask_data): # image_data 에 mask img만 들어왔음 acensia
        height, width, _ = image_data.shape
        print(image_data.shape)
        cropped_list = np.array(
            [
                image_data[i][j]
                for i in range(height)
                for j in range(width)
                if mask_data[i][j] > 200 # mask 허용 범위 변경 acensia
            ]
        )
        
        colors, counts_rate = self.dom_with_Kmeans(cropped_list)
        fst, snd, trd = colors[:3]
        # print(counts_rate)
        # print(fst)
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

        return predicted_color_str
