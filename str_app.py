import io
import time
import requests
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from lpr.app.trackers.sort import Sort
from dotenv import load_dotenv
import os
import base64

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

class App:
    title = 'License Plate Detection and Recognition'

    def __init__(self):
        self.api_base = os.getenv("API_BASE")
        self.external_cameras = {}
        self.tracker = Sort()
        self.fetch_external_cameras()

    def process_remote_images(self, image_bytes=None, camera_id="0"):
        payload = {}
        if camera_id in self.external_cameras:
            response = requests.post(f"{self.api_base}/detect/camera?camera_id={camera_id}", json=payload)
        else:
            if image_bytes:
                image_base64 = base64.b64encode(image_bytes).decode()
                payload = {"image_base64": image_base64}
            response = requests.post(f"{self.api_base}/detect/image", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Image processing failed: {response.text}")

    def fetch_external_cameras(self):
        try:
            response = requests.get(f"{self.api_base}/cameras")
            response.raise_for_status()
            camera_list = response.json()
            if isinstance(camera_list, list):
                for cam in camera_list:
                    if isinstance(cam, dict) and 'id' in cam:
                        self.external_cameras[cam['id']] = cam
        except:
            self.external_cameras = {}

    def build_rtsp_url(self, camera_id):
        if camera_id in self.external_cameras:
            cam_data = self.external_cameras[camera_id]
            user = cam_data.get('rtspUser')
            password = cam_data.get('rtspPass')
            host = cam_data.get('rtspHost')
            port = cam_data.get('rtspPort', '554')
            path = cam_data.get('rtspPath', '/')
            if all([user, password, host, port, path]):
                rtsp_url = f"rtsp://{user}:{password}@{host}:{port}{path}"
                if path.lower().endswith('/streaming/channels'):
                    rtsp_url = f"{rtsp_url}/101"
                else:
                    rtsp_url = f"{rtsp_url}?channel=1&subtype=0"
                return rtsp_url
        return None

def main():
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False

    st.title(App.title)
    input_method = st.radio('Input Method', ['Upload Image', 'Camera', 'RTSP Camera'])
    uploaded_file = None
    selected_camera = None
    camera_id = "0"
    app = App()

    if input_method == 'Upload Image':
        uploaded_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg', 'bmp'])
    elif input_method == 'Camera':
        uploaded_file = st.camera_input('Take Photo')
    elif input_method == 'RTSP Camera':
        if not app.external_cameras:
            st.error("No external cameras available")
            return
        camera_display_map = {f"Camera {cam_id} - {cam_data.get('parkingName', f'ID {cam_id}')}": cam_id for cam_id, cam_data in app.external_cameras.items()}
        camera_options_display = list(camera_display_map.keys())
        selected_camera_display = st.selectbox('Select Camera', camera_options_display)
        selected_camera = selected_camera_display
        camera_id = camera_display_map[selected_camera_display]

    if st.button('Detect and Recognize License Plates'):
        if input_method == 'RTSP Camera':
            if not selected_camera:
                st.error('Please select a camera.')
                return
            processing_result = app.process_remote_images(image_bytes=None, camera_id=camera_id)
            processed_plates = processing_result.get('processed_plates', [])
            frame_base64 = processing_result.get('frame')
            if frame_base64:
                frame_bytes = base64.b64decode(frame_base64)
                image = Image.open(io.BytesIO(frame_bytes))
            else:
                image = None
        else:
            if not uploaded_file:
                st.error('Please provide an image first.')
                return
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
            image_bytes = image_bytes.getvalue()
            processing_result = app.process_remote_images(image_bytes, camera_id)
            processed_plates = processing_result.get('processed_plates', [])

        if not processed_plates:
            st.warning('No license plates detected.')
            return
        st.success(f'Detected {len(processed_plates)} license plate(s).')
        cols = st.columns(min(len(processed_plates), 3))
        for i, plate_data in enumerate(processed_plates):
            recognized_text = plate_data['detectedText']
            with cols[i % 3]:
                st.subheader(f'Plate {plate_data["id"]} Recognition')
                st.image(image, caption=f'Recognized: {recognized_text}', width='stretch')
                st.write(f'Confidence: {plate_data["confidence"]:.2f}')

    if input_method == 'RTSP Camera' and selected_camera:
        if st.button('Start Monitoring'):
            st.session_state.monitoring = True
        if st.button('Stop Monitoring'):
            st.session_state.monitoring = False

        frame_placeholder = st.empty()
        detection_placeholder = st.empty()
        count_placeholder = st.empty()
        plate_count_threshold = 1
        plate_detection_counts = {}
        confirmed_plates = set()

        if st.session_state.monitoring:
            rtsp_url = app.build_rtsp_url(camera_id)
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                st.error(f'Failed to open RTSP stream. URL: {rtsp_url}')
                st.session_state.monitoring = False
            else:
                last_detection_time = 0
                detection_interval = 5
                while st.session_state.monitoring:
                    ret, frame = cap.read()
                    if not ret:
                        st.error('Failed to read frame.')
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels='RGB')
                    image = Image.fromarray(frame_rgb)
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='JPEG')
                    image_bytes = img_bytes.getvalue()

                    current_time = time.time()
                    if current_time - last_detection_time >= detection_interval:
                        last_detection_time = current_time
                        try:
                            result = app.process_remote_images(image_bytes=image_bytes, camera_id=camera_id)
                            processed_plates = result.get('processed_plates', [])
                            if processed_plates:
                                detection_placeholder.subheader(f'Detected {len(processed_plates)} License Plate(s)')
                                plates_to_show = []
                                for plate_data in processed_plates:
                                    recognized_text = plate_data['detectedText']
                                    if recognized_text not in plate_detection_counts:
                                        plate_detection_counts[recognized_text] = []
                                    plate_detection_counts[recognized_text].append(current_time)
                                    plate_detection_counts[recognized_text] = [t for t in plate_detection_counts[recognized_text] if current_time - t < 300]
                                    if len(plate_detection_counts[recognized_text]) >= plate_count_threshold and recognized_text not in confirmed_plates:
                                        confirmed_plates.add(recognized_text)
                                        plates_to_show.append(plate_data)
                                if plates_to_show:
                                    cols = st.columns(min(len(plates_to_show), 3))
                                    for i, plate_data in enumerate(plates_to_show):
                                        recognized_text = plate_data['detectedText']
                                        with cols[i % 3]:
                                            st.image(image, caption=f'Recognized: {recognized_text}', width='stretch')
                                            st.write(f'Confidence: {plate_data["confidence"]:.2f}')
                                count_info = " | ".join([f"{text}: {len(times)}" for text, times in plate_detection_counts.items() if len(times) > 0])
                                count_placeholder.info(f"Detection counts (need {plate_count_threshold}+ to confirm): {count_info}")
                            else:
                                detection_placeholder.empty()
                                count_placeholder.empty()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    time.sleep(0.03)
                cap.release()

if __name__ == '__main__':
    main()
