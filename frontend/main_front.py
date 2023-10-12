import streamlit as st
from module.login_upload_module import login_section
from module.signup_module import signup_section
from module.result_module import result_session
from module.loading_module import loading_session

from module.session_manager import session_state_init


session_state_init()

# 유저 정보 상태
if st.session_state.get("logged_in", False):
    st.markdown(
        f"<div style='text-align: right; font-size: 12px;'>로그인 유저: {st.session_state.get('email','이메일 없음')}</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div style='text-align: right; font-size: 12px;'>로그인 유저: Guest(비회원)</div>",
        unsafe_allow_html=True,
    )

# 메인 앱 로직
if not st.session_state["logged_in"]:  # 비 로그인상태
    if st.session_state["sign_up"]:  # 회원가입 폼
        signup_section()
    else:  # 로그인 폼
        login_section()

else:  # Logged in
    if st.session_state["current_page"] == "login":
        login_section()
    elif st.session_state["current_page"] == "image_upload":
        login_section()  # from upload
    elif st.session_state["current_page"] == "loading":  # Loading session
        loading_session()
    elif st.session_state["current_page"] == "result":  # Result session
        result_session()
    else:  # Default to image upload section
        login_section()  # from upload
