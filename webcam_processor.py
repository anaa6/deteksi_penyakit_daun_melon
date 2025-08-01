# webcam_processor.py
import cv2
import numpy as np
import av
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration
import queue 

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class MelonDiseaseProcessor(VideoProcessorBase):

    CONFIDENCE_THRESHOLD = 0.50
    PROCESS_INTERVAL = 5 
    INFERENCE_IMG_SIZE = 480 

    def __init__(self, model_instance):
        self.model = model_instance
        self.frame_count = 0
        self.last_annotated_frame_bgr = None

    def _prepare_image_for_inference(self, img_bgr: np.ndarray) -> Image.Image:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.INFERENCE_IMG_SIZE and (img_rgb.shape[0] != self.INFERENCE_IMG_SIZE or img_rgb.shape[1] != self.INFERENCE_IMG_SIZE):
            return Image.fromarray(cv2.resize(img_rgb, (self.INFERENCE_IMG_SIZE, self.INFERENCE_IMG_SIZE), interpolation=cv2.INTER_AREA))
        return Image.fromarray(img_rgb)

    def _process_detections_and_annotate(self, results, original_frame_bgr: np.ndarray) -> np.ndarray:
        frame_with_annotations = original_frame_bgr.copy()

        if results and results[0].boxes:
            orig_h, orig_w = original_frame_bgr.shape[:2]
            inferred_w, inferred_h = self.INFERENCE_IMG_SIZE, self.INFERENCE_IMG_SIZE
            scale_x, scale_y = orig_w / inferred_w, orig_h / inferred_h

            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= self.CONFIDENCE_THRESHOLD:
                    name = self.model.names[int(box.cls[0])]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
                    x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)

                    cv2.rectangle(frame_with_annotations, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
                    cv2.putText(frame_with_annotations, f"{name} {conf:.2f}", (x1_orig, y1_orig - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_with_annotations

    def recv(self, frame: av.VideoFrame):
        self.frame_count += 1
        img_bgr = frame.to_ndarray(format="bgr24")
        frame_to_return = img_bgr.copy() 
        
        if self.frame_count % self.PROCESS_INTERVAL == 0:
            try:
                pil_image = self._prepare_image_for_inference(img_bgr)
                results = self.model.predict(pil_image, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
                frame_to_return = self._process_detections_and_annotate(results, img_bgr)
                self.last_annotated_frame_bgr = frame_to_return.copy()
            except Exception as e:
                # Menampilkan pesan error di video jika terjadi kesalahan
                cv2.putText(frame_to_return, f"ERROR: {e}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.last_annotated_frame_bgr = frame_to_return.copy() 
            
        elif self.last_annotated_frame_bgr is not None:
            frame_to_return = self.last_annotated_frame_bgr

        return av.VideoFrame.from_ndarray(frame_to_return, format="bgr24")
