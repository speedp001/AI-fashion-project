import streamlit as st
import time
import requests

def result_backend():
    if not st.session_state["flask_upload_url"]:
        print("No url")
    if not st.session_state["request_form"]:
        print("No files")

    response = requests.post(
        st.session_state["flask_upload_url"],
        files=st.session_state["request_form"],
    )
    if response.status_code == 200:
        st.session_state["current_page"] = "result"
        st.session_state["result"] = response.json()
        st.session_state["loading"] = False
        print("Go to result")
        st.experimental_rerun()
    else:
        error_message = response.json().get("error")
        st.error(f"이미지 전송 실패: {error_message}")
        st.session_state["loading"] = False
        st.error("업로드 페이지로 복귀합니다....")
        time.sleep(3)
        st.session_state["current_page"] = "image_upload"
        st.experimental_rerun()
        


def loading_session():
    st.subheader("AI패션 추천 서비스", divider="grey")
    if st.session_state["loading"]:
        st.image("./front_images/loading_ai_6.gif", use_column_width=True)

        st.write("AI가 열심히 분석중입니다.")
        st.write("잠시만 기다려주세요...")

        st.subheader(" ", divider="grey")

    if st.button(":rewind: 이미지 다시 올리기"):
        # st.session_state["loading"] = False
        st.session_state["current_page"] = "image_upload"
        st.experimental_rerun()

    result_backend()


# if __name__ == "__main__":
#     if "loading" not in st.session_state:
#         st.session_state["loading"] = True
#     loading_session()
