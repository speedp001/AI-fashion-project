import streamlit as st
import re
import requests
from email_validator import validate_email, EmailNotValidError


def signup_section():
    st.success("회원가입 세션", icon="✅")
    if st.session_state["sign_up"]:
        st.markdown("<h1 style='font-size: 32px;'>회원가입</h1>", unsafe_allow_html=True)

        def is_valid_username(username):
            pattern = r"^[a-zA-Z0-9]{1,20}$"
            return bool(re.match(pattern, username))

        def is_username_available(username):
            # 백엔드로 username 중복 여부를 확인하는 요청을 보냅니다.
            response = requests.post(
                "http://localhost:5000/check_username",
                data={"username": signup_username},
            )
            if response.status_code == 200:
                return response.json()["available"]
            else:
                # 요청 실패 시 기본적으로 False를 반환하도록 처리
                return False

        col1, col2 = st.columns(2)  # 컬럼 레이아웃을 생성합니다.
        with col1:
            signup_username = st.text_input("닉네임")
        message_container_username = st.empty()  # 빈 요소를 생성하여 메시지를 표시할 준비를 합니다.
        with col2:
            st.write(
                "<div style='height: 28px;'></div>", unsafe_allow_html=True
            )  # 행 정렬 맞춰주기 위한 노력...
            if st.button("사용하기"):
                if is_valid_username(signup_username):
                    if is_username_available(signup_username):
                        message_container_username.success("사용 가능한 username입니다.")
                    else:
                        message_container_username.warning(
                            "이미 존재하는 username입니다. 다른 username을 선택해주세요."
                        )
                else:
                    message_container_username.error(
                        "유효하지 않은 username입니다. 영문 대소문자와 숫자만 허용됩니다."
                    )

        def is_id_available(id):
            # 백엔드로 username 중복 여부를 확인하는 요청을 보냅니다.
            response = requests.post(
                "http://localhost:5000/send_code", data={"id": signup_id}
            )
            if response.status_code == 200:
                return response.json()["available"]
            else:
                # 요청 실패 시 기본적으로 False를 반환하도록 처리
                return False

        col3, col4 = st.columns(2)  # 컬럼 레이아웃을 생성합니다.
        with col3:
            signup_id = st.text_input("이메일(ID)")
        message_container_id = st.empty()  # 빈 요소를 생성하여 메시지를 표시할 준비를 합니다.
        with col4:
            st.write(
                "<div style='height: 30px;'></div>", unsafe_allow_html=True
            )  # 행 정렬 맞춰주기 위한 노력...
            # 인증 버튼을 클릭하면 백엔드로 데이터를 전송하고 인증 코드를 요청
            if st.button("인증 코드 요청"):
                try:
                    # 이메일 유효성 검사
                    valid_email = validate_email(signup_id)
                    print("전송")
                    if is_id_available(signup_id):
                        message_container_id.success("이메일로 인증 코드가 전송되었습니다.")
                    else:
                        message_container_id.warning("이미 가입된 이메일(ID)입니다.")
                except EmailNotValidError:
                    message_container_id.error("유효한 이메일 주소를 입력하세요.")

        verification_code = st.text_input("인증 코드(4자리 숫자)", "")
        # 입력한 내용이 바뀔 때마다 확인
        if verification_code:
            response = requests.post(
                "http://localhost:5000/verify",
                data={"signup_id": signup_id, "verification_code": verification_code},
            )
            if response.status_code == 200:
                st.success("인증 코드가 유효합니다. 이메일이 성공적으로 인증되었습니다.")
            elif response.status_code == 400:
                st.error("인증 코드가 유효하지 않습니다. 다시 확인하세요.")

        col5, col6 = st.columns(2)  # 컬럼 레이아웃을 생성합니다.
        # 비밀번호와 비밀번호 확인 필드
        with col5:
            signup_pw = st.text_input("비밀번호", type="password", key="signup_pw")
            # 비밀번호 안내문구
            if signup_pw:
                # 비밀번호가 요구사항을 충족하지 않을 경우 메시지 표시
                if not re.match(
                    r"^(?=.*[a-zA-Z0-9])(?=.*[@#$%^&+=!])(?=.{6,})", signup_pw
                ):
                    st.error("비밀번호는 6자리 이상의 영문과 특수문자 조합을 사용해야 합니다.")
                else:
                    st.empty()  # 아무것도 표시하지 않음
        with col6:
            signup_confirm_pw = st.text_input(
                "비밀번호 확인", type="password", key="signup_confirm_pw"
            )
        # 비밀번호 일치 여부 확인 및 메시지 표시
        if signup_pw and signup_confirm_pw:
            if signup_pw != signup_confirm_pw:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                st.success("비밀번호가 일치합니다.")
        else:
            st.empty()  # 아무것도 표시하지 않음

        signup_gender = st.selectbox("성별", ["Male", "Female"])

        # 회원가입 버튼을 누르면 백엔드에 회원 정보 저장 및 로그인 상태 변경
        signup_button = st.button(":man-woman-girl-boy:회원가입")
        if signup_button:
            # 입력된 정보 가져오기
            signup_data = {
                "username": signup_username,
                "email": signup_id,
                "pw": signup_pw,
                "gender": signup_gender,
            }

            # 모든 입력값이 비어있지 않은지 확인
            if all(signup_data.values()):
                # 백엔드로 회원가입 데이터 전송 및 응답 처리
                response = requests.post(
                    "http://localhost:5000/sign_up", json=signup_data
                )

                if response.status_code == 200:
                    st.success("회원가입이 완료되었습니다.")
                    # 회원가입 성공 시 로그인 상태 변경
                    st.session_state["sign_up"] = False
                    st.experimental_rerun()

                elif response.status_code == 400:
                    # 중복된 username 또는 id일 경우
                    error_message = response.json()["message"]
                    st.error(error_message)
                else:
                    st.error("회원가입 중 오류가 발생하였습니다.")

                    # 로그인 상태 변경 후 로그인 세션으로 자동 전환 가능
            # (중복 등의 오류 처리도 여기에서 가능)
            else:
                st.warning("모든 필드를 입력하세요.")

        if st.button(":rewind: 돌아가기"):
            st.session_state["sign_up"] = False
            st.experimental_rerun()
