from flask import Flask, request, jsonify
from flask_mail import Mail, Message
import random
import string

app = Flask(__name__)

# Flask-Mail 설정
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # 이메일 호스트 서버 설정
app.config['MAIL_PORT'] = 587  # 이메일 호스트 포트 설정 (일반적으로 587 또는 465)
app.config['MAIL_USE_TLS'] = True  # TLS(Transport Layer Security) 사용 여부 설정
app.config['MAIL_USERNAME'] = 'kdhwi92@gmail.com'  # 이메일 계정
app.config['MAIL_PASSWORD'] = 'kgnfjnorrakfrwzq'  # 이메일 비밀번호

# Mail 인스턴스 생성
mail = Mail(app)

# 가상의 데이터베이스로 사용할 딕셔너리
users_db = {}
email_verification_codes = {}

# 이메일 형식 검증 함수
def is_valid_email(email):
    # 간단한 이메일 형식 검증 로직을 사용하거나 정교한 이메일 형식 검증 라이브러리를 사용할 수 있습니다.
    # 이 예제에서는 간단한 형식을 사용합니다.
    return "@" in email

# 이메일 인증 코드 생성 함수
def generate_verification_code():
    # 4자리 숫자로 된 랜덤 코드 생성
    return ''.join(random.choices(string.digits, k=4))

# 이메일 보내기 함수
def send_email(email, verification_code):
    msg = Message('이메일 인증 코드', sender='your_email@example.com', recipients=[email])
    msg.body = f'인증 코드: {verification_code}'
    mail.send(msg)

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')

    if not is_valid_email(email):
        return jsonify({"message": "유효한 이메일 주소를 입력하세요."}), 400

    if email in users_db:
        return jsonify({"message": "이미 가입된 이메일(ID)입니다."}), 400

    # 이메일 인증 코드 생성
    verification_code = generate_verification_code()

    # 이메일 인증 코드를 이메일로 보냄
    send_email(email, verification_code)

    # 이메일 인증 코드를 딕셔너리에 저장
    email_verification_codes[email] = verification_code

    return jsonify({"message": "이메일로 인증 코드가 전송되었습니다.", "verification_code": verification_code})

@app.route('/verify', methods=['POST'])
def verify():
    email = request.form.get('email')
    entered_code = request.form.get('verification_code')

    stored_verification_code = email_verification_codes.get(email)

    if not stored_verification_code:
        return jsonify({"message": "인증 코드가 만료되었습니다. 새로 인증 코드를 요청하세요."}), 400

    if stored_verification_code == entered_code:
        return jsonify({"message": "이메일이 성공적으로 인증되었습니다."})
    else:
        return jsonify({"message": "잘못된 인증 코드입니다."}), 400
    
@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    verification_code = request.form.get('verification_code')
    password = request.form.get('password')

    # 이메일 및 인증 코드 유효성 검사
    if not is_valid_email(email):
        return jsonify({"message": "유효한 이메일 주소를 입력하세요."}), 400

    stored_verification_code = email_verification_codes.get(email)

    if not stored_verification_code or stored_verification_code != verification_code:
        return jsonify({"message": "잘못된 인증 코드입니다."}), 400

    if email in users_db:
        return jsonify({"message": "이미 가입된 이메일(ID)입니다."}), 400

    # 비밀번호 유효성 검사 (추가적인 검사 필요)
    if len(password) < 6:
        return jsonify({"message": "비밀번호는 최소 6자 이상이어야 합니다."}), 400

    # 사용자 등록 (가상의 데이터베이스에 추가)
    users_db[email] = {"email": email, "password": password}

    return jsonify({"message": "회원가입이 성공적으로 완료되었습니다."})


if __name__ == '__main__':
    app.run(debug=True)
