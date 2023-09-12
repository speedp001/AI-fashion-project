# Flask 애플리케이션에서 사용하는 메인 파일 (app.py)

import io
import cv2
import numpy as np
import threading
import bcrypt # 암호 해싱 지원 라이브러리

from pymongo import MongoClient
from rembg import remove
from PIL import Image
from flask import Flask, request, jsonify, redirect, make_response
from classifier import ItemClassifier, ColorClassifier





app = Flask(__name__)

####### MyClassifier 클래스 인스턴스 생성
item_classifier = ItemClassifier()
color_classifier = ColorClassifier()


####### DB connection
client = MongoClient("mongodb+srv://sudo:sudo@atlascluster.e7pmjep.mongodb.net/")
user = client["user"]
user_info = user.info
app = Flask(__name__)


####### 사용자 이미지 배경제거 함수
def rembg(img):
    # 이미지 데이터를 바이너리에서 이미지로 디코딩
    org_image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    # BGR에서 RGB로 변환
    # org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

    # 배경을 흰색으로 변경 -> bgcolor=(b, g, r, a)
    color_rembg_img = remove(org_image, only_mask=True)
    item_rembg_img = remove(org_image, bgcolor=(255, 255, 255, 255))

    return item_rembg_img, color_rembg_img


####### Image upload
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # 사용자 정보 불러오기
        email = request.files["email"].read().decode("utf-8")
        style = request.files["style"].read().decode("utf-8")
        image = request.files["image"].read()
        
        # DB에서 email로 gender 조회
        gender = "0" if user_info.find_one({"email": email})["gender"] == "남성" else 1
        
        # 사용자 이미지
        img_byte = io.BytesIO(image).getvalue()
        nparr = np.frombuffer(img_byte, np.uint8)
        item_rembg_img, color_rembg_img,  = rembg(nparr)
        cv2.imwrite("./color_rembg_img.png", color_rembg_img)
        cv2.imwrite("./item_rembg_img.png", item_rembg_img)
        print(email, style, gender)
        print(style.split(","))
        print(item_rembg_img.shape)
        print(color_rembg_img.shape)

        # 멀티스레딩
        try:
            # 멀티 스레딩으로 클래스 내부 함수 실행
            thread1 = threading.Thread(target=item_classifier.item_predict, args=(item_rembg_img[:,:,:3],))
            thread2 = threading.Thread(target=color_classifier.color_predict, args=(item_rembg_img[:,:,:3], color_rembg_img))
            thread3 = threading.Thread(target=item_classifier.style_predict, args=(style,))

            # 스레드 시작
            thread1.start()
            thread2.start()
            thread3.start()
            # thread4.start()

            # 스레드가 종료될 때까지 대기
            thread1.join()
            thread2.join()
            thread3.join()
            # thread4.join()

            # 스레드 반환 결과
            predicted_label1 = thread1.result  # item info
            predicted_label2 = thread2.result  # color info
            predicted_label3 = thread3.result  # style info

            # 0(gender) 1(style) 13(color) 16(item) 6자리 'searchcode'
            search_code = gender + predicted_label3 + predicted_label2 + predicted_label1
            print(search_code)

        except Exception as e:
            # 오류 처리 및 오류 코드 반환 -> 서버 treading 문제
            error_message = f"Thread error: {str(e)}"
            return jsonify({'error': error_message}), 500

        # DB에서 1,2,3,4 정보 조합해서 정보 조회
        try:
            pass
            #json형태로 200코드와 조회 이미지를 딕셔너리형태로 클라이언트한테 반환해준다.
            # return jsonify({'recommend_image': image }), 200
        
        except Exception as e:
            # 오류 처리 및 오류 코드 반환 -> 서버 treading 문제
            error_message = f"DB error: {str(e)}"
            return jsonify({'error': error_message}), 500
        
        

    except Exception as e:
        # 오류 처리 및 오류 코드 반환 -> 클라이언트 이미지 형식 문제, http통신 문제
        error_message = f"User error: {str(e)}"
        return jsonify({'error': error_message}), 400


####### Sign-up
@app.route("/sign-up", methods=["POST"])
def sign_up():
    info = request.json  # => front
    username = info["username"]
    email = info["email"]
    pw = info["pw"]
    gender = 0 if info["gender"] == "남성" else 1
    info["gender"] = gender
    hashed_pw = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())
    existing_user = user_info.find_one({"username": username, "email": email})
    if existing_user:
        return {
            "msg": "User with the same username and id already exists. Try another one."
        }
    else:
        info["pw"] = hashed_pw.decode("utf-8")
        user_info.insert_one(info)
        res = {"msg": "User registration has been successfully done."}
        return jsonify(res), 200


####### Login
@app.route("/login", methods=["POST"])  
def login():
    info = request.json
    email = info["email"]
    pw = info["pw"]
    user = user_info.find_one({"email": email})
    if user:
        if bcrypt.checkpw(pw.encode("utf-8"), user["pw"].encode("utf-8")):
            username = user["username"]
            return jsonify({"msg": "Sign-in successful!"}), 200
    else:
        return jsonify({"message": "이메일 주소와 비밀번호를 확인해주세요."}), 401


####### Logout
@app.route("/logout")
def logout():
    res = make_response(redirect("http://localhost:8501"))
    res.delete_cookie("user_token")
    return res
    
    
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
