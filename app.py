import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Set the title and header of the app
st.title('Parking Space Detection')
st.header('Detect Empty and Occupied Spots using YOLOv8')

# Load the trained model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    # Check if the uploaded file is an image or a video
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        # Process the image
        st.subheader("Image Analysis")
        
        # Read the image using PIL and convert to OpenCV format
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Perform detection on the image
        results = model(image_np)
        
        # Initialize counters
        empty_count = 0
        occupied_count = 0
        
        for r in results:
            detections = r.boxes.cls.cpu().numpy()
            empty_count = np.sum(detections == 0)
            occupied_count = np.sum(detections == 1)
            annotated_image = r.plot()
            
        # Display the annotated image and counters
        st.image(annotated_image, caption=f'Empty: {empty_count} | Occupied: {occupied_count}', use_column_width=True)
        
    elif file_type == 'video':
        # Process the video
        st.subheader("Video Analysis")
        st.warning("Video processing might be slow depending on the video length and model size.")

        # Save the video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        video_path = "temp_video.mp4"

        # Initialize counters
        empty_count = 0
        occupied_count = 0

        # Create a placeholder to display the video frames
        frame_placeholder = st.empty()
        
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection on the frame
            results = model(frame)
            
            # Count the empty and occupied spots
            for r in results:
                detections = r.boxes.cls.cpu().numpy()
                empty_count = np.sum(detections == 0)
                occupied_count = np.sum(detections == 1)
                annotated_frame = r.plot()

            # Display the annotated frame
            text_counters = f'Empty: {empty_count} | Occupied: {occupied_count}'
            cv2.putText(annotated_frame, text_counters, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame_placeholder.image(annotated_frame, channels='BGR', use_column_width=True, caption=text_counters)
            
        cap.release()
        st.success("Video processing complete!")