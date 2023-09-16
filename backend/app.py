# Flask 애플리케이션에서 사용하는 메인 파일 (app.py)

import io
import random
import string
import numpy as np
import bcrypt  # 암호 해싱 지원 라이브러리

from pymongo import MongoClient
from rembg import remove
from flask_mail import Mail, Message
from flask import Flask, request, jsonify, redirect, make_response
from concurrent.futures import ThreadPoolExecutor
from db_search import DB_search
from classifier import ItemClassifier, ColorClassifier


# Flask 인스턴스 정의
app = Flask(__name__)


# MyClassifier 클래스 인스턴스 생성
item_classifier = ItemClassifier()
color_classifier = ColorClassifier()


# Mail 인스턴스 생성
mail = Mail(app)
email_verification_codes = {}  # 인증코드 저장 딕셔너리

app.config["MAIL_SERVER"] = "smtp.gmail.com"  # 이메일 호스트 서버 설정
app.config["MAIL_PORT"] = 587  # 이메일 호스트 포트 설정 (일반적으로 587 또는 465)
app.config["MAIL_USE_TLS"] = True  # TLS(Transport Layer Security) 사용 여부 설정
app.config["MAIL_USERNAME"] = "kdhwi92@gmail.com"  # 관리자 이메일 계정
app.config["MAIL_PASSWORD"] = "kgnfjnorrakfrwzq"  # 관리자 이메일 비밀번호


# DB connection
client = MongoClient(
    "mongodb+srv://sudo:sudo@atlascluster.e7pmjep.mongodb.net/")
user = client["user"]
user_info = user.info


# Sign-up
@app.route("/sign_up", methods=["POST"])
def sign_up():
    signup_data = request.json  # front
    username = signup_data["username"]  # 사용자 이름
    email = signup_data["email"]  # 가입 이메일
    pw = signup_data["pw"]  # 가입 비밀번호
    gender = 0 if signup_data["gender"] == "Male" else 1  # 성별
    signup_data["gender"] = gender

    # 비밀번호 해쉬화
    hashed_pw = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())

    # 가입 조건 확인
    if user_info.find_one({"username": username}):
        res = {"msg": "이미 사용중인 닉네임입니다. "}
        return jsonify(res), 400
    elif user_info.find_one({"email": email}):
        res = {"msg": "이미 가입된 이메일입니다."}
        return jsonify(res), 400
    else:
        signup_data["pw"] = hashed_pw.decode("utf-8")
        user_info.insert_one(signup_data)
        res = {"msg": f"{username}님 회원가입을 축하드립니다."}
        return jsonify(res), 200


# Check username
@app.route("/check_username", methods=["POST"])
def check_username():
    username = request.form["username"]
    # 'username' 필드로 중복을 확인
    if user_info.find_one({"username": username}):
        return jsonify({"available": False})
    return jsonify({"available": True})


# Send email code
@app.route("/send_code", methods=["POST"])
def send_code():
    signup_id = request.form["id"]
    # 'id' 필드로 중복을 확인
    if user_info.find_one({"id": signup_id}):
        return jsonify({"available": False})

    # 이메일 인증 코드 생성 함수
    def generate_verification_code():
        # 4자리 숫자로 된 랜덤 코드 생성
        return "".join(random.choices(string.digits, k=4))

    # 이메일 보내기 함수
    def send_email(signup_id, verification_code):
        msg = Message("이메일 인증 코드", sender="help@example.com",
                      recipients=[signup_id])  # recipients 받을 사람의 목록
        msg.body = f"인증 코드: {verification_code}"
        mail.send(msg)

    # 이메일 인증 코드 생성
    verification_code = generate_verification_code()

    # 이메일 보내기 함수 호출
    send_email(signup_id, verification_code)

    # email_verification_codes 딕셔너리에 저장
    email_verification_codes[signup_id] = verification_code
    # ex) {signup_id : "1234"}

    return jsonify(
        {
            "available": True,
            "message": "이메일로 인증 코드가 전송되었습니다.",
            "verification_code": verification_code,
        }
    )


# Verify email code
@app.route("/verify", methods=["POST"])
def verify():
    signup_id = request.form["signup_id"]
    entered_code = request.form["verification_code"]  # form에 작성한 4자리 인증코드
    verification_code = email_verification_codes.get(
        signup_id)  # email_verification_codes에 저장된 4자리 인증코드

    # 저장된 인증코드와 입력 인증코드의 일치 확인
    if not verification_code:
        return jsonify({"message": "인증 코드를 요청하지 않았거나 유효하지 않습니다."}), 400

    elif verification_code == entered_code:
        return jsonify({"message": "인증 코드가 유효합니다. 이메일이 성공적으로 인증되었습니다."}), 200

    else:
        return jsonify({"message": "인증 코드가 유효하지 않습니다. 다시 확인하세요."}), 400


# Login
@app.route("/login", methods=["POST"])
def login():
    login_info = request.json
    login_email = login_info["email"]
    login_pw = login_info["pw"]

    # email 정보로 찾은 user document
    user_document = user_info.find_one({"email": login_email})

    # if user는 해당 이메일 주소와 일치하는 사용자가 데이터베이스에서 찾았을 때 True가 되고, 사용자가 찾아지지 않은 경우 False
    if user_document:
        if bcrypt.checkpw(login_pw.encode("utf-8"), user_document["pw"].encode("utf-8")):
            username = user_document["username"]
            return jsonify({"msg": f"환영합니다. {username}님"}), 200
    else:
        return jsonify({"msg": "이메일 주소와 비밀번호를 확인해주세요."}), 401


# Image upload
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # 사용자 정보 불러오기
        email = request.files["email"].read().decode("utf-8")
        style = request.files["style"].read().decode("utf-8")
        image = request.files["image"].read()

        # DB에서 email로 gender 조회
        gender = "0" if user_info.find_one({"email": email})[
            "gender"] == 0 else "1"

        # 사용자 upload 이미지
        img_byte = io.BytesIO(image).getvalue()
        img_array = np.frombuffer(img_byte, np.uint8)
        # print(img_array)
        item_rembg_img, color_rembg_img = item_classifier.rembg(img_array)
        print(item_rembg_img, color_rembg_img)

        # Item, Color, Style 판단
        try:
            # 스레드 풀 생성
            with ThreadPoolExecutor() as executor:
                # 함수들을 제출하고 결과를 얻음
                predicted_label1 = executor.submit(
                    item_classifier.item_predict, item_rembg_img[:, :, :3]).result()
                predicted_label2 = executor.submit(
                    color_classifier.color_predict, item_rembg_img[:, :, :3], color_rembg_img).result()
                predicted_label3 = executor.submit(
                    item_classifier.style_predict, style).result()

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
            result = DB_search(client, search_code)
            # json형태로 200코드와 조회 이미지를 딕셔너리형태로 클라이언트한테 반환해준다.
            return jsonify(result), 200

        except Exception as e:
            # 오류 처리 및 오류 코드 반환 -> 서버 treading 문제
            error_message = f"DB error: {str(e)}"
            print(error_message)
            return jsonify({'error': error_message}), 500

    except Exception as e:
        # 오류 처리 및 오류 코드 반환 -> 클라이언트 이미지 형식 문제, http통신 문제
        error_message = f"User error: {str(e)}"
        return jsonify({'error': error_message}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
