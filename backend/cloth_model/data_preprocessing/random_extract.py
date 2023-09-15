####### 데이터 셋에서 550개를 랜덤으로 추출하는 코드

import os
import random
import shutil

# 원본 이미지 파일이 있는 디렉토리 경로
source_directory = "./"

# 이미지를 저장할 새로운 디렉토리 경로
destination_directory = "./"

# 원하는 이미지 개수
num_images_to_select = 550

# 원본 디렉토리에서 모든 이미지 파일 가져오기
image_files = [f for f in os.listdir(source_directory) if f.endswith(".png")]

# 이미지 파일을 무작위로 선택
selected_images = random.sample(image_files, num_images_to_select)

# 새로운 이미지 디렉토리가 존재하지 않으면 생성
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# 새로운 디렉토리에 선택된 이미지 복사
for image_file in selected_images:
    source_path = os.path.join(source_directory, image_file)
    destination_path = os.path.join(destination_directory, image_file)
    shutil.copy(source_path, destination_path)

print(f"{num_images_to_select} 개의 이미지를 선택하여 복사했습니다.")
