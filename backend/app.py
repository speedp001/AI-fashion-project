# Flask 애플리케이션에서 사용하는 메인 파일 (app.py)

import io
import cv2
import numpy as np
import threading
import bcrypt # 암호 해싱 지원 라이브러리

from pymongo import MongoClient
from rembg import remove
from flask import Flask, request, jsonify, redirect, make_response
from concurrent.futures import ThreadPoolExecutor
from classifier import ItemClassifier, ColorClassifier


app = Flask(__name__)

####### MyClassifier 클래스 인스턴스 생성
item_classifier = ItemClassifier()
color_classifier = ColorClassifier()


####### DB connection
client = MongoClient("mongodb+srv://sudo:sudo@atlascluster.e7pmjep.mongodb.net/")
user = client["user"]
user_info = user.info


####### 사용자 이미지 배경제거 함수
def rembg(img):
    # 이미지 데이터를 바이너리에서 이미지로 디코딩
    org_image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

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
        gender = "0" if user_info.find_one({"email": email})["gender"] == 0 else "1"
        
        # 사용자 upload 이미지
        img_byte = io.BytesIO(image).getvalue()
        img_array = np.frombuffer(img_byte, np.uint8)
        item_rembg_img, color_rembg_img,  = rembg(img_array)

        # Item, Color, Style 판단
        try:
            # 스레드 풀 생성
            with ThreadPoolExecutor() as executor:
                # 함수들을 제출하고 결과를 얻음
                predicted_label1 = executor.submit(item_classifier.item_predict, item_rembg_img[:,:,:3]).result()
                predicted_label2 = executor.submit(color_classifier.color_predict, item_rembg_img[:,:,:3], color_rembg_img).result()
                predicted_label3 = executor.submit(item_classifier.style_predict, style).result()

            # 스레드 풀 종료
            executor.shutdown()

            # search code 생성
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
    signup_info = request.json  # => front
    username = signup_info["username"]
    email = signup_info["email"]
    pw = signup_info["pw"]
    gender = 0 if signup_info["gender"] == "남성" else 1
    signup_info["gender"] = gender
    hashed_pw = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())
    existing_user = user_info.find_one({"username": username, "email": email, "gender": gender})
    if existing_user:
        res = {"msg": "User with the same username and id already exists. Try another one."}
        return jsonify(res), 402
    else:
        pw = hashed_pw.decode("utf-8")
        user_info.insert_one(signup_info)
        res = {"msg": f"{username}님 회원가입을 축하드립니다."}
        return jsonify(res), 200


####### Login
@app.route("/login", methods=["POST"])  
def login():
    login_info = request.json
    login_email = login_info["email"]
    login_pw = login_info["pw"]
    
    # email 정보로 찾은 user document
    user_document = user_info.find_one({"email": login_email})
    
    # if user는 해당 이메일 주소와 일치하는 사용자가 데이터베이스에서 찾아졌을 때 True가 되고, 사용자가 찾아지지 않은 경우 False
    if user_document:
        if bcrypt.checkpw(login_pw.encode("utf-8"), user_document["pw"].encode("utf-8")):
            username = user_document["username"]
            return jsonify({"msg": f"환영합니다. {username}님"}), 200
    else:
        return jsonify({"msg": "이메일 주소와 비밀번호를 확인해주세요."}), 401


####### Logout
@app.route("/logout")
def logout():
    res = make_response(redirect("http://localhost:8501"))
    res.delete_cookie("user_token")
    return res
    
    
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
