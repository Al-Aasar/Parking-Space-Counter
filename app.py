import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


st.markdown("<h1 style='text-align: center;'>Parking Space Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Detect Empty and Occupied Spots using YOLOv8</h3>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:

    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':

        st.subheader("Image Analysis")
        
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        results = model(image_np)
        
        empty_count = 0
        occupied_count = 0
        
        for r in results:
            detections = r.boxes.cls.cpu().numpy()
            empty_count = np.sum(detections == 0)
            occupied_count = np.sum(detections == 1)
            annotated_image = r.plot()
            
        st.image(annotated_image, caption=f'Empty: {empty_count} | Occupied: {occupied_count}', use_column_width=True)
        
    elif file_type == 'video':

        st.subheader("Video Analysis")
        st.warning("Video processing might be slow depending on the video length and model size.")


        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        video_path = "temp_video.mp4"

        empty_count = 0
        occupied_count = 0

        frame_placeholder = st.empty()
        
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for r in results:
                detections = r.boxes.cls.cpu().numpy()
                empty_count = np.sum(detections == 0)
                occupied_count = np.sum(detections == 1)
                annotated_frame = r.plot()

            text_counters = f'Empty: {empty_count} | Occupied: {occupied_count}'
            cv2.putText(annotated_frame, text_counters, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame_placeholder.image(annotated_frame, channels='BGR', use_column_width=True, caption=text_counters)
            
        cap.release()
        st.success("Video processing complete!")
