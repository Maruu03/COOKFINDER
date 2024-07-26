from ultralytics import YOLO
import time
import streamlit as st
import cv2
import numpy as np
from pytube import YouTube

import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    #display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True
    #if display_tracker == 'Yes'
    if is_display_tracker:
        #tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        tracker_type = "bytetrack.yaml"
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Video detectado',
                   channels="BGR",
                   use_column_width=True
                   )


def play_webcam(confidence, model):
    is_display_tracker, tracker = display_tracker_options()

    # Inicializar la webcam
    video_stream = cv2.VideoCapture(0)
    ret, frame = video_stream.read()

    if not ret:
        video_stream = cv2.VideoCapture(1)

    if not video_stream.isOpened():
        st.error("No se pudo abrir la cámara web.")
        return

    st_frame = st.empty()
    capture_button = st.button("Capturar")

    while not capture_button:
        ret, frame = video_stream.read()

        if not ret:
            st.error("Error al capturar el fotograma de la webcam.")
            break

        if frame is not None:
            # Detección de ingredientes en el fotograma actual
            _display_detected_frames(confidence, model, st_frame, frame, is_display_tracker, tracker)

            # Capturar una imagen con los ingredientes detectados si se hace clic en el botón "Capturar"
            if capture_button:
                captured_frame = frame.copy()
                res = model.predict(frame, conf=confidence)
                for detection in res[0].xyxy:
                    box = detection[0:4].int().cpu().numpy()
                    cv2.rectangle(captured_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                st.image(captured_frame, channels="BGR", caption="Ingredientes detectados")
                break

    # Liberar recursos
    video_stream.release()

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Escoja un video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detectar los objetos del video'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
