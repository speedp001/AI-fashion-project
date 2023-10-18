import streamlit as st
import requests
from io import BytesIO
import mimetypes


def login_section():
    left_column, right_column = st.columns(2)

    # 로그인 상태마다 다른 UI 적용하기
    if st.session_state["logged_in"]:
        with left_column:
            st.subheader(":robot_face:   AI패션 추천 서비스", divider="grey")  # ln1

            options = [
                "casual",
                "dandy",
                "formal",
                "girlish",
                "gorpcore",
                "retro",
                "romantic",
                "sports",
                "street",
            ]
            selected_options = st.multiselect(
                " :heavy_check_mark: 원하는 스타일을 하나만 선택해주세요",
                options,
                max_selections=1,
                key="K",
            )

            uploaded_file = st.file_uploader(
                ":heavy_check_mark: 아래에서 이미지를 업로드 하세요. :camera:",
                type=["jpg", "jpeg", "png"],
            )
            if selected_options:
                st.session_state["selected"] = selected_options

            if uploaded_file:
                ext = mimetypes.guess_extension(uploaded_file.type)
                if ext not in [".jpg", ".jpeg", ".png"]:
                    st.error(
                        ":ballot_box_with_check: 업로드 파일을 다시 확인해주시고, 의류 이미지를 업로드 해주세요."
                    )
                else:
                    file_stream = BytesIO(uploaded_file.read())
                    st.session_state["uploaded_file"] = file_stream

            if st.session_state["uploaded_file"]:
                st.image(
                    st.session_state["uploaded_file"],
                    caption="업로드된 이미지",
                    use_column_width=True,
                )

            if st.session_state["uploaded_file"] and st.session_state["selected"]:

                def upload_request():
                    flask_server_url = "http://localhost:5000/upload"
                    file_bytes = st.session_state["uploaded_file"].getvalue()
                    files = {
                        "email": st.session_state["email"],
                        "style": "".join(st.session_state["selected"]),
                        "image": file_bytes,
                    }
                    st.session_state["flask_upload_url"] = flask_server_url
                    st.session_state["request_form"] = files
                    st.session_state["loading"] = True
                    st.session_state["current_page"] = "loading"
                    st.session_state["needs_rerun"] = True

                st.button(":postbox: AI에게 이미지 보내기", on_click=upload_request)
                        

            st.subheader(" ", divider="grey")
            if st.button(":x:로그아웃"):
                st.session_state["logged_in"] = False
                st.session_state["email"] = ""
                st.experimental_rerun()
    else:
        left_column.subheader(":robot_face:   AI패션 추천 서비스", divider="grey")
        email = left_column.text_input(":e-mail: 이메일 주소", key="unique_login_email")
        password = left_column.text_input(
            ":closed_lock_with_key:비밀번호", type="password", key="login_password"
        )

        if left_column.button(":key:  로그인"):
            try:
                response = requests.post(
                    "http://localhost:5000/login",
                    json={"email": email, "pw": password},
                )
                if response.status_code == 200:
                    left_column.success("로그인 성공")

                    # 로그인 성공 시, 세션 상태를 업데이트합니다.
                    st.session_state["logged_in"] = True
                    st.session_state["email"] = email  # 사용자가 입력한 이메일. 여기 함수에서만 정의됨

                    st.experimental_rerun()
                else:
                    left_column.error("로그인 실패")
            except Exception as e:
                st.error(f"서버와 통신 중 문제가 발생했습니다: {e}")

        if left_column.button(":man-woman-girl-boy:회원가입"):
            st.session_state["sign_up"] = True
            st.experimental_rerun()

    if st.session_state["logged_in"]:
        right_column.image(
            "./front_images/upload_session_image.jpg", use_column_width=True
        )
    else:
        right_column.image("./front_images/main_image.jpg", use_column_width=True)


# if __name__ == '__main__':
#     login_section()
