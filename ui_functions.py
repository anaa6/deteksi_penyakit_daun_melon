import streamlit as st
import cv2
import io
import os
import uuid
import pandas as pd
from PIL import Image
import numpy as np
import time
import database as db
import queue
import base64 

from model_load import load_yolo_model
from webcam_processor import MelonDiseaseProcessor, RTC_CONFIGURATION
from streamlit_webrtc import webrtc_streamer, WebRtcMode

@st.cache_resource
def get_yolo_model():
    return load_yolo_model()

yolo_model = get_yolo_model()

def _image_to_base64(image_np_rgb: np.ndarray) -> str | None:
    try:
        pil_img = Image.fromarray(image_np_rgb)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        st.error(f"Gagal mengonversi gambar ke base64: {e}")
        return None

def _process_image_with_model(uploaded_image_data: bytes, confidence_threshold: float):
    if not yolo_model:
        return None, "Error: Model tidak tersedia.", 0.0, [], []

    image_pil = Image.open(io.BytesIO(uploaded_image_data))
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    results = yolo_model(image_pil, conf=confidence_threshold, verbose=False)

    plotted_image_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    detections_summary_list = []
    confidences_list = []
    detected_class_names = []
    highest_confidence = 0.0

    if results[0].boxes:
        for box in results[0].boxes: 
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = yolo_model.names[cls]

            detections_summary_list.append(f"{name} ({conf:.2f})")
            confidences_list.append(conf)
            detected_class_names.append(name)
            if conf > highest_confidence:
                highest_confidence = conf

    detection_summary = ", ".join(detections_summary_list) if detections_summary_list else "Tidak ada deteksi yang melewati ambang batas."

    return plotted_image_rgb, detection_summary, highest_confidence, detected_class_names, confidences_list

def show_login_page():
    st.title("Aplikasi Deteksi Penyakit Daun Melon")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if db.verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username atau Password salah.")

    st.markdown("---")
    st.info("Belum punya akun? Klik tombol di bawah untuk mendaftar.")
    if st.button("Daftar Akun Baru"):
        st.session_state.page = 'register'
        st.rerun()

def show_register_page():
    st.title("Aplikasi Deteksi Penyakit Daun Melon")
    st.subheader("Daftar Akun Baru")

    with st.form("register_form"):
        new_username = st.text_input("Username Baru")
        new_password = st.text_input("Password Baru", type="password")
        confirm_password = st.text_input("Konfirmasi Password", type="password")
        register_button = st.form_submit_button("Daftar")

        if register_button:
            if not new_username or not new_password:
                st.error("Username dan Password tidak boleh kosong.")
            elif new_password != confirm_password:
                st.error("Konfirmasi password tidak cocok.")
            else:
                if db.add_user(new_username, new_password):
                    st.success("Akun berhasil dibuat! Silakan login.")
                    st.session_state.page = 'login'
                    st.rerun()
                else:
                    st.error("Username sudah terdaftar. Gunakan username lain.")
    st.markdown("---")
    st.info("Sudah punya akun? Klik tombol di bawah untuk login.")
    if st.button("Kembali ke Login"):
        st.session_state.page = 'login'
        st.rerun()

def show_history_page():
    st.title(f"Riwayat Deteksi")

    st.write("Berikut adalah riwayat deteksi penyakit yang telah Anda lakukan:")
    st.markdown("---")

    history_records = db.get_detection_history(st.session_state.username)

    if history_records:
        for record in history_records:
            record_id = record['id']
            st.write(f"**Waktu:** {record['timestamp']} | **Penyakit:** {record['disease_name']} | **Kepercayaan:** {record['confidence']:.2f}")

            col_img, col_delete_btn = st.columns([2, 1]) 
            with col_img:
                image_base64_data = record['image_path'] 
                
                if image_base64_data: 
                    try:
                        decoded_img_bytes = base64.b64decode(image_base64_data)
                        img = Image.open(io.BytesIO(decoded_img_bytes))
                        st.image(img, caption="Gambar Hasil Deteksi", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Gagal memuat gambar dari database: {e}")
                        st.info("Tidak ada gambar deteksi tersedia (Error decoding).") 
                else:
                    st.info("Tidak ada gambar deteksi tersedia.") 

            with col_delete_btn:
                st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) 
                if st.button("🗑️ Hapus", key=f"delete_{record_id}"):
                    if db.delete_detection_record(record_id):
                        st.success(f"Catatan riwayat berhasil dihapus.")
                    else:
                        st.error("Gagal menghapus catatan riwayat.")
                    time.sleep(1.5) 
                    st.rerun() 
            st.markdown("---") 

    else:
        st.info("Anda belum memiliki riwayat deteksi.")

def run_webcam_detection():
    st.info("Arahkan webcam Anda ke daun melon untuk deteksi langsung.")

    status_placeholder = st.empty()

    webrtc_ctx = None
    if yolo_model:
        webrtc_ctx = webrtc_streamer(
            key="melon-disease",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: MelonDiseaseProcessor(yolo_model), 
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.error("Model YOLO belum dimuat, tidak bisa memulai webcam deteksi.")
        status_placeholder.error("Model tidak tersedia.")
        return 
    
    if webrtc_ctx and webrtc_ctx.video_processor:
        try:
            while True:
                detection_info_from_queue = webrtc_ctx.video_processor.out_queue.get_nowait()
                if detection_info_from_queue is not None:
                    st.session_state['current_detection_info'] = detection_info_from_queue
        except queue.Empty:
            pass

    if webrtc_ctx and webrtc_ctx.state.playing:
        status_placeholder.success("Webcam aktif dan mendeteksi!")

    else:
        status_placeholder.info("Webcam belum aktif. Klik tombol 'Start' di bawah video untuk memulai deteksi.")
        st.session_state['current_detection_info'] = {
            "diseases": [],
            "avg_confidence": 0.0,
            "keterangan": "Menunggu aktivasi webcam." 
        }

    st.markdown("---")

def _reset_upload_state():
    st.session_state.uploaded_image_data = None
    st.session_state.uploaded_file_hash = None
    st.session_state.uploaded_file_name = None
    st.session_state.processed_image_for_display_upload = None
    st.session_state.detection_results_summary_upload = "Tidak ada deteksi."
    st.session_state.detection_highest_confidence_upload = 0.0
    st.session_state.detected_class_names_upload = []
    st.session_state.confidences_list_upload = []
    st.session_state.last_processed_upload_conf = 0.0
    st.session_state.last_saved_upload_hash = None
    st.session_state.last_saved_upload_conf_for_hash = 0.0
    
    if 'last_upload_conf_slider_value' not in st.session_state:
        st.session_state.last_upload_conf_slider_value = 0.50

def _render_upload_section():
    if 'uploaded_file_hash' not in st.session_state:
        _reset_upload_state()

    uploaded_file = st.file_uploader("Pilih file gambar daun melon", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_file_hash = uuid.uuid5(uuid.NAMESPACE_URL, file_bytes).hex

        if st.session_state.uploaded_file_hash != current_file_hash:
            _reset_upload_state()
            st.session_state.uploaded_image_data = file_bytes
            st.session_state.uploaded_file_hash = current_file_hash
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.last_uploaded_file_id_processed = current_file_hash 
            st.rerun() 
    elif st.session_state.uploaded_file_hash is not None and uploaded_file is None:
        _reset_upload_state()
        st.rerun() 

    confidence_threshold_upload = st.slider(
        "Ambang Batas Kepercayaan (Confidence Threshold)",
        min_value=0.01,
        max_value=1.0,
        value=st.session_state.last_upload_conf_slider_value,
        step=0.01,
        key="upload_conf_slider",
        help="Atur nilai minimum kepercayaan agar deteksi ditampilkan pada gambar yang diunggah."
    )
    st.session_state.last_upload_conf_slider_value = confidence_threshold_upload 

    st.markdown("---")

    if st.session_state.uploaded_image_data is not None:
        if (st.session_state.processed_image_for_display_upload is None or
            st.session_state.last_processed_upload_conf != confidence_threshold_upload):

            with st.spinner('Memproses deteksi penyakit...'):
                processed_img, summary, highest_conf, detected_class_names, confidences_list_from_processing = \
                    _process_image_with_model(st.session_state.uploaded_image_data, confidence_threshold_upload)

                if processed_img is None: 
                    st.error(summary) 
                else:
                    st.session_state.processed_image_for_display_upload = processed_img
                    st.session_state.detection_results_summary_upload = summary
                    st.session_state.detection_highest_confidence_upload = highest_conf
                    st.session_state.detected_class_names_upload = detected_class_names
                    st.session_state.confidences_list_upload = confidences_list_from_processing
                    st.session_state.last_processed_upload_conf = confidence_threshold_upload

        if st.session_state.username and st.session_state.processed_image_for_display_upload is not None:
            if (st.session_state.last_saved_upload_hash != st.session_state.uploaded_file_hash or
                st.session_state.last_saved_upload_conf_for_hash != confidence_threshold_upload):

                image_base64_for_db = _image_to_base64(st.session_state.processed_image_for_display_upload)

                if image_base64_for_db: 
                    confidence_to_save = st.session_state.detection_highest_confidence_upload 
                    disease_names_for_db = ", ".join(list(set(st.session_state.detected_class_names_upload))) if st.session_state.detected_class_names_upload else "Tidak Terdeteksi"

                    if db.add_detection_record(st.session_state.username, disease_names_for_db, confidence_to_save, image_base64_for_db):
                        st.success("Deteksi berhasil disimpan ke riwayat!") 
                        st.session_state.last_saved_upload_hash = st.session_state.uploaded_file_hash
                        st.session_state.last_saved_upload_conf_for_hash = confidence_threshold_upload
                    else:
                        st.error("Gagal menyimpan catatan deteksi ke database.")
                else:
                    st.error("Gagal mengonversi gambar untuk penyimpanan ke database.") 
        
        st.subheader("Perbandingan Gambar Asli dan Hasil Deteksi:")
        col1, col2 = st.columns(2, gap="small")

        with col1:
            if st.session_state.uploaded_image_data is not None:
                try:
                    st.image(st.session_state.uploaded_image_data, caption='Gambar Asli', use_container_width=True)
                except Exception as e:
                    st.error("Gagal menampilkan gambar asli.")
            else:
                st.info("Gambar asli belum tersedia.")
        with col2:
            if st.session_state.processed_image_for_display_upload is not None:
                try:
                    st.image(st.session_state.processed_image_for_display_upload, caption='Hasil Deteksi', use_container_width=True)
                except Exception as e:
                    st.error("Gagal menampilkan gambar hasil deteksi.")
            else:
                st.info("Gambar hasil deteksi belum tersedia atau sedang diproses.")

        st.subheader("Ringkasan Hasil Deteksi:")
        detected_classes = st.session_state.get('detected_class_names_upload', [])

        if "daun sehat" in detected_classes:
            if len(detected_classes) == 1:
                st.write(f"✅ Daun melon terlihat **Sehat**")
            else:
                other_diseases = [d for d in detected_classes if d != "daun sehat"]
                st.write(f"❗ Penyakit Terdeteksi: **{', '.join(other_diseases)}**")
                st.warning("Perhatian: 'Daun sehat' juga terdeteksi, mungkin ada ambiguitas atau tumpang tindih.")
        elif not detected_classes and st.session_state.detection_results_summary_upload == "Tidak ada deteksi yang melewati ambang batas.":
            st.write(f"❓ Tidak ada deteksi yang valid atau dikenali dengan ambang batas saat ini.")
        elif detected_classes:
            st.write(f"❗ Penyakit Terdeteksi: **{', '.join(detected_classes)}**")
        else:
            st.write("Tidak ada deteksi yang valid.") 

        st.write(f"Kepercayaan Tertinggi: {st.session_state.detection_highest_confidence_upload:.2f}")

    else:
        st.info("Mohon unggah file gambar daun melon untuk memulai deteksi.")

def show_main_app_page():
    st.title(f"Selamat Datang di Halaman Deteksi, {st.session_state.username}!")

    if 'main_detection_mode' not in st.session_state:
        st.session_state.main_detection_mode = "Unggah Gambar"

    detection_mode = st.radio(
        "Pilih metode deteksi:",
        ("Unggah Gambar", "Gunakan Webcam"),
        horizontal=True,
        key="main_detection_mode"
    )
    st.markdown("---")

    if detection_mode == "Unggah Gambar":
        _render_upload_section()
    elif detection_mode == "Gunakan Webcam":
        run_webcam_detection() 

def show_about_app_page():
    st.columns([4, 3, 4])[1].title("Info Aplikasi")
    st.markdown("---")

    st.markdown(
        """
        ### Deteksi Penyakit Daun Melon
        Aplikasi ini dirancang untuk membantu Anda **mendeteksi dini penyakit pada daun tanaman melon** dengan mudah melalui analisis gambar.
        """
    )
    st.markdown("---")

    st.markdown(
        """
        ### Panduan Penggunaan Aplikasi
        Anda akan menemukan dua fitur utama dalam aplikasi ini:
        1.  **Deteksi Penyakit:**
            * Pilih tab **"Unggah Gambar"** jika Anda ingin menganalisis foto daun melon yang sudah ada di perangkat Anda. Unggah gambar, dan sistem akan menampilkan hasil deteksi beserta gambarnya. Anda bisa menyesuaikan Ambang Batas Kepercayaan (Confidence Threshold) untuk melihat hasil dengan akurasi yang berbeda.
            * Pilih tab **"Gunakan Webcam"** untuk mendeteksi langsung daun melon melalui kamera perangkat Anda. Pastikan Anda memberikan izin akses kamera. Aplikasi akan menampilkan deteksi secara *real-time*.
            * Hasil deteksi akan menunjukkan jenis penyakit yang teridentifikasi (jika ada) dan tingkat keyakinan (confidence) model terhadap deteksi tersebut.
        2.  **Riwayat Deteksi:**
            * Kunjungi halaman **"Riwayat Deteksi"** untuk melihat semua catatan deteksi yang pernah Anda lakukan. Setiap catatan mencakup tanggal, jenis penyakit, tingkat kepercayaan, dan gambar yang dianalisis.
            * Anda juga bisa menghapus catatan riwayat yang tidak lagi diperlukan.
        """
    )
    st.markdown("---")

    st.markdown(
        """
        ### Catatan Penting
        **Penyakit yang saat ini dapat dideteksi oleh aplikasi ini adalah Downy Mildew dan Cucumber Mosaic Virus (CMV).**
        """
    )