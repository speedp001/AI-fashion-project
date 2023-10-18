#init main
import streamlit as st


def session_state_init():
    # Initialize session states if not already done

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "loading" not in st.session_state:
        st.session_state["loading"] = False
    if "result" not in st.session_state:
        st.session_state["result"] = False
    if "sign_up" not in st.session_state:
        st.session_state["sign_up"] = False
    if "fail" not in st.session_state:
        st.session_state["fail"] = False
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "login"
    if "user" not in st.session_state:
        st.session_state["user"] = ""


    # Upload session
    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None
    if "selected" not in st.session_state:
        st.session_state["selected"] = None
        
    # Result session
    if "result" not in st.session_state:
        st.session_state["result"] = None