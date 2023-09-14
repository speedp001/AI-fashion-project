####### 기존 이미지 확장자를 .png로 바꿔주는 코드

from PIL import Image
import os
from tqdm import tqdm

# 변환할 폴더의 경로를 설정
base_dir = './'

# "Filtered" 폴더 생성
filtered_folder_path = os.path.join(base_dir, "fw_filtered")

# base_dir 안의 모든 서브폴더를 순회합니다.
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    # ".DS_Store" 파일이거나 "Filtered" 폴더와 같은 이름의 폴더가 이미 존재하는 경우 무시
    if folder_name.startswith('.') or folder_path == filtered_folder_path:
        continue


    # 폴더명으로 된 하위 폴더를 "Filtered" 폴더 내에 생성
    filtered_subfolder_path = os.path.join(filtered_folder_path, folder_name)
    os.makedirs(filtered_subfolder_path, exist_ok=True)

    # tqdm을 사용하여 진행 상황 표시
    progress_bar = tqdm(os.listdir(folder_path), desc=f"폴더 {folder_name}")

    # 서브폴더 내의 이미지 파일들을 순회합니다.
    for i, filename in enumerate(progress_bar):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)

            # 이미지 열기
            img = Image.open(img_path)

            # 새로운 파일명 생성 (폴더명(인덱스).png)
            new_filename = f"{folder_name}({i + 1}).png"
            new_img_path = os.path.join(filtered_subfolder_path, new_filename)

            # 이미지를 .png 형식으로 저장
            img.save(new_img_path, 'PNG')

            # 진행 상황 업데이트
            progress_bar.set_postfix(변환_파일=new_filename)

print("모든 작업이 완료되었습니다.")

