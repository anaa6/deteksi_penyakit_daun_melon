import streamlit as st
from ultralytics import YOLO
import os 

MODEL_PATH = "best.pt"

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan 'app.py'.")
        return None

    try:
        model = YOLO(MODEL_PATH)
        model.to('cpu') 
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO. Pastikan file '{MODEL_PATH}' ada di direktori yang sama dan pustaka 'ultralytics' terinstal dengan benar. Detail: {e}")
        st.exception(e) 
        return None

yolo_model = load_yolo_model()