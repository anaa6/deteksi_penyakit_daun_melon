import streamlit as st
import os
import database as db
from ui_functions import show_login_page, show_register_page, show_main_app_page, show_history_page, show_about_app_page

st.set_page_config(layout="wide", page_title="Deteksi Penyakit Daun Melon")

def initialize_app():
    db.init_db()

initialize_app()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'username' not in st.session_state:
    st.session_state.username = None

if 'uploaded_image_data' not in st.session_state:
    st.session_state.uploaded_image_data = None
if 'uploaded_file_hash' not in st.session_state:
    st.session_state.uploaded_file_hash = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'processed_image_for_display_upload' not in st.session_state:
    st.session_state.processed_image_for_display_upload = None
if 'detection_results_summary_upload' not in st.session_state:
    st.session_state.detection_results_summary_upload = "Tidak ada deteksi."
if 'detection_highest_confidence_upload' not in st.session_state:
    st.session_state.detection_highest_confidence_upload = 0.0
if 'last_upload_conf_slider_value' not in st.session_state:
    st.session_state.last_upload_conf_slider_value = 0.50
if 'last_processed_upload_conf' not in st.session_state:
    st.session_state.last_processed_upload_conf = 0.0
if 'last_saved_upload_hash' not in st.session_state:
    st.session_state.last_saved_upload_hash = None
if 'last_saved_upload_conf_for_hash' not in st.session_state:
    st.session_state.last_saved_upload_conf_for_hash = 0.0

if 'confidences_list_upload' not in st.session_state:
    st.session_state.confidences_list_upload = []
if 'detected_class_names_upload' not in st.session_state:
    st.session_state.detected_class_names_upload = []

if 'current_detection_info' not in st.session_state:
    st.session_state.current_detection_info = {"diseases": [], "avg_confidence": 0.0, "keterangan": "Menunggu aktivasi webcam..."}

if not st.session_state.logged_in:
    st.sidebar.info("Silakan login untuk mengakses aplikasi.")
else:
    st.sidebar.title(f"Halo, **{st.session_state.username}**!")
    if st.sidebar.button("Logout", key="sidebar_logout_btn"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.session_state.username = None
        st.session_state.uploaded_image_data = None
        st.session_state.uploaded_file_hash = None
        st.session_state.processed_image_for_display_upload = None
        st.session_state.detection_results_summary_upload = "Tidak ada deteksi."
        st.session_state.detection_highest_confidence_upload = 0.0
        st.session_state.last_upload_conf_slider_value = 0.50
        st.session_state.last_processed_upload_conf = 0.0
        st.session_state.last_saved_upload_hash = None
        st.session_state.last_saved_upload_conf_for_hash = 0.0
        st.session_state.confidences_list_upload = []
        st.session_state.detected_class_names_upload = []
        st.session_state.current_detection_info = {"diseases": [], "avg_confidence": 0.0, "keterangan": "Menunggu aktivasi webcam."}
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.title("Menu Aplikasi")

    if st.sidebar.button("Deteksi Penyakit", key="sidebar_nav_detect"):
        st.session_state.page = "main_app"
        st.rerun()

    if st.sidebar.button("Riwayat Deteksi", key="sidebar_nav_history"):
        st.session_state.page = "history"
        st.rerun()

    if st.sidebar.button("Info Aplikasi", key="sidebar_nav_about_app"):
        st.session_state.page = "about_app"
        st.rerun()

if st.session_state.logged_in:
    if st.session_state.page == 'main_app':
        show_main_app_page()
    elif st.session_state.page == 'history':
        show_history_page()
    elif st.session_state.page == 'about_app':
        show_about_app_page()
    else:
        st.session_state.page = 'main_app'
        show_main_app_page()
else:
    if st.session_state.page == 'login':
        show_login_page()
    elif st.session_state.page == 'register':
        show_register_page()
    else:
        st.session_state.page = 'login'
        show_login_page()