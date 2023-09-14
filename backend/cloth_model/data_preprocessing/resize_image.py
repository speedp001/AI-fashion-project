####### 이미지 크기 변경

import os
import cv2

input_dir = './'  # 이미지가 들어있는 디렉토리 경로

output_dir = './'  # 크기 변경된 이미지를 저장할 디렉토리 경로
os.makedirs(output_dir, exist_ok=True)

desired_size = (640, 640)  # 변경할 크기

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  # 이미지 파일인지 확인
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)  # 이미지 로드
        
        resized_image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)  # 크기 변경
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_image)  # 크기 변경된 이미지 저장
        print(f"Resized {filename} and saved to {output_path}")

print("Done resizing images!")
