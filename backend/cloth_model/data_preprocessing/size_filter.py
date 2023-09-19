####### 지정한 사이즈 이상의 이미지들만 저장하는 코드

import os
import cv2
import glob



# 사이즈를 검사할 디렉토리 지정
path = "./"
input_dir = glob.glob(os.path.join(path, "*", "*.png"))

min_width = 480
min_height = 640

for image_path in input_dir :
        image = cv2.imread(image_path)  # 이미지 로드
        
        # 디렉토리 이름을 파일 이름으로 사용
        dir_name = os.path.basename(os.path.dirname(image_path))
        
        # 파일 이름에서 확장자 .jpg 제거
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        output_dir = f"./filtered/{dir_name}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        height, width = image.shape[:2]
        if width >= min_width and height >= min_height :
            output_path = os.path.join(output_dir, f"{filename}.png")
            cv2.imwrite(output_path, image)  # 이미지 저장
            print(f"Saved {filename} to {output_path}")

print("Done")























# input_dir = './org/'  # 디렉토리 경로 설정
# output_dir = './org_size_filtered/'  # 변경된 이미지 저장할 디렉토리 경로
# os.makedirs(output_dir, exist_ok=True)

# min_width = 480
# min_height = 640

# for filename in os.listdir(input_dir) :
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일인지 확인
#         image_path = os.path.join(input_dir, filename)
#         image = cv2.imread(image_path)  # 이미지 로드

#         height, width = image.shape[:2]
#         if width >= min_width and height >= min_height :
#             output_path = os.path.join(output_dir, filename)
#             cv2.imwrite(output_path, image)  # 이미지 저장
#             print(f"Saved {filename} to {output_path}")

# print("Done processing images!")
