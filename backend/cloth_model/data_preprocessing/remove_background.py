####### 원본 이미지를 배경을 제거한 이미지로 변경 후 저장

import cv2
import os
import glob
from rembg import remove
from tqdm import tqdm



# 작업을 진행할 디렉토리 경로
path = "./"
org_image = glob.glob(os.path.join(path, "*.jpg"))

count = 1

# tqdm을 사용하여 파일 전처리 진행 상황 시각화
for i in tqdm(org_image):
    
    # 디렉토리 이름을 파일 이름으로 사용
    dir_name = os.path.basename(os.path.dirname(i))
    
    # 파일 이름에서 확장자 .jpg 제거
    filename = os.path.splitext(os.path.basename(i))[0]
    
    input = cv2.imread(i)
    # 배경제거 뒤에 배경은 투명도 0에 흰색으로 처리
    output = remove(input, bgcolor=(255,255,255,255))

    output_dir = f"./filtered/{dir_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 다음 디렉토리 회귀 시에 count 초기화 작업
    if i != org_image[-1] and os.path.dirname(i) != os.path.dirname(org_image[org_image.index(i) + 1]):
        count = 1
    
    output_path = os.path.join(output_dir, f"{filename}_white_bg({count}).png")
    cv2.imwrite(output_path, output)
    
    count += 1

print("Done!")






# org_image = "/Users/sang-yun/Desktop/Cloth_Model/cloth_org_dataset/both/cardigan/cardigan(1).png"
# filename = os.path.splitext(os.path.basename(org_image))[0]



# input = cv2.imread(org_image)
# output = remove(input, bgcolor=(255,255,255,255))


# output_dir = "./"

# output_path = os.path.join(output_dir, f"{filename}_white_bg.png")
# cv2.imwrite(output_path, output)

# output_array = cv2.imread('cardigan(1)_white_bg.png')
# print(output_array)
