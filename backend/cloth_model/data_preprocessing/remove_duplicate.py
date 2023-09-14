####### 디렉토리 내에서 중복된 사진을 찾아 삭제해주는 코드

import os
import hashlib

# 중복 이미지를 검사하고 삭제할 폴더를 지정
folder_path = './'

def image_hash(file_path):
    with open(file_path, "rb") as f:
        image_data = f.read()
        return hashlib.md5(image_data).hexdigest()

# 중복 검사 함수
def find_and_remove_duplicate_images(folder_path):
    image_hashes = {}
    duplicate_count = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                # 이미지 파일의 해시 값을 계산합니다.
                img_hash = image_hash(file_path)

                # 이미 해시가 있는 이미지면 삭제합니다.
                if img_hash in image_hashes:
                    print(f"중복된 이미지 발견: {file_path}")
                    os.remove(file_path)
                    duplicate_count += 1
                else:
                    image_hashes[img_hash] = file_path

    print(f"총 {duplicate_count}개의 중복 이미지를 삭제했습니다.")




if __name__ == "__main__":
    find_and_remove_duplicate_images(folder_path)
