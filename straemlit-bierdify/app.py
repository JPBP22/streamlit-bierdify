import streamlit as st
import os
import cv2
import pandas as pd
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import tempfile
from csv_processing import process_csv
from model_predictions import load_model as load_prediction_model, make_predictions  # Renamed load_model
from model_predictions import model_path as model_path
from trimmer import trim_video
from ground_truth import compare_predictions


# Initialize session state for processed files
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'processed_csv' not in st.session_state:
    st.session_state.processed_csv = None
if 'csv_output_path' not in st.session_state:
    st.session_state.csv_output_path = None
if 'trimmed_video' not in st.session_state:
    st.session_state.trimmed_video = None

# Load the NAS Pose Large model with COCO pretrained weights
@st.cache_resource
def load_model():
    model_type = "yolo_nas_pose_l"
    yolo_nas_pose = models.get(model_type, pretrained_weights="coco_pose")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_nas_pose.to(device)
    return yolo_nas_pose, device

yolo_nas_pose, device = load_model()


def process_single_frame(image, conf):
    result = yolo_nas_pose.predict(image, conf=conf, batch_size=32)
    processed_frame = [image_prediction.draw()[:,:,::-1] for image_prediction in result._images_prediction_lst]
    return processed_frame[0], result


def create_csv(pose_data_list, csv_path):
    all_data = []
    max_bboxes = 0
    desired_columns = 18  # Define the number of columns you want

    for frame_index, frame_data in enumerate(pose_data_list):
        frame_bboxes = len(frame_data._images_prediction_lst)
        max_bboxes = max(max_bboxes, frame_bboxes)

        for bbox_id, pose in enumerate(frame_data._images_prediction_lst):
            pose_coords = [coord for point in pose.prediction.poses for coord in point]
            pose_row = [frame_index] + pose_coords
            pose_row = pose_row[:desired_columns]
            all_data.append(pose_row)

    if all_data:
        column_names = ['frame_index']
        num_features_per_bbox = (desired_columns - 1) // max_bboxes
        for bbox_id in range(max_bboxes):
            column_names += [f'bbox_{bbox_id}_coord_{i}' for i in range(num_features_per_bbox)]

        df = pd.DataFrame(all_data, columns=column_names)
        df.to_csv(csv_path, index=False)
    else:
        print(f"No data to write to CSV for {csv_path}")

def create_video_from_frames(frames, output_filename, fps=30.0):
    frame_height, frame_width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Using H.264 codec
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)

    out.release()

def process_video(video_path, output_video_path, csv_output_path, device, conf=0.4, progress_bar=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    all_pose_data = []
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, pose_data = process_single_frame(frame, conf)
        frames.append(processed_frame)
        all_pose_data.append(pose_data)

        if progress_bar:
            progress_bar.progress((frame_idx + 1) / total_frames)

    cap.release()
    create_video_from_frames(frames, output_video_path)
    create_csv(all_pose_data, csv_output_path)

    print(f'Processed video saved as {output_video_path}')
    print(f'CSV file saved as {csv_output_path}')

# Streamlit UI
st.title("Video Pose Estimation with YOLO NAS Pose")

def get_stable_temp_path():
    if 'temp_path' not in st.session_state:
        st.session_state.temp_path = tempfile.mkdtemp()
    return st.session_state.temp_path

# Initialize session state
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

# File uploader and process button
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="file_uploader")


# Insert the ground truth CSV uploader here
ground_truth_file = st.file_uploader("Upload Ground Truth CSV", type=["csv"], key="ground_truth_uploader")

if ground_truth_file is not None:
    ground_truth_path = os.path.join(get_stable_temp_path(), "ground_truth.csv")
    with open(ground_truth_path, 'wb') as f:
        f.write(ground_truth_file.read())
    st.session_state['ground_truth_path'] = ground_truth_path

process_button = st.button("Process Video")

if process_button and uploaded_file is not None and not st.session_state.get('video_processed', False):
    st.session_state['stable_temp_path'] = get_stable_temp_path()

    temp_video_path = os.path.join(st.session_state['stable_temp_path'], "temp_video.mp4")
    with open(temp_video_path, 'wb') as f:
        f.write(uploaded_file.read())

    st.session_state['output_video_path'] = os.path.join(st.session_state['stable_temp_path'], 'output_video.mp4')
    st.session_state['csv_output_path'] = os.path.join(st.session_state['stable_temp_path'], 'output_data.csv')

    progress_text = "Processing video. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    process_video(temp_video_path, st.session_state['output_video_path'], st.session_state['csv_output_path'], device, conf=0.4, progress_bar=my_bar)

    my_bar.empty()

    with open(st.session_state['output_video_path'], 'rb') as file:
        st.session_state.processed_video = file.read()
    with open(st.session_state['csv_output_path'], 'rb') as file:
        st.session_state.processed_csv = file.read()

    st.session_state.video_processed = True

# Display processed video and download buttons
if st.session_state.get('video_processed', False):
    # Display the processed video if available
    if st.session_state.get('processed_video'):
        st.video(st.session_state['processed_video'])

        # Provide a download button for the processed video
        st.download_button(label="Download Processed Video", 
                           data=st.session_state['processed_video'], 
                           file_name="processed_video.mp4", 
                           mime="video/mp4")

    # Display the CSV download button if available
    if st.session_state.get('processed_csv'):
        st.download_button(label="Download Processed CSV", 
                           data=st.session_state['processed_csv'], 
                           file_name="pose_data.csv", 
                           mime="text/csv")

# Button to make predictions
make_predictions_button = st.button("Make Predictions")

if make_predictions_button:
    if 'csv_output_path' in st.session_state and st.session_state['csv_output_path']:
        processed_df = process_csv(st.session_state['csv_output_path'])
        prediction_model = load_prediction_model(model_path)  # Update with your model's path
        predictions = make_predictions(prediction_model, processed_df)
        processed_df['predictions'] = predictions

        # Save the DataFrame with predictions
        predictions_csv_path = os.path.join(st.session_state['stable_temp_path'], 'processed_predictions.csv')
        processed_df.to_csv(predictions_csv_path, index=False)

        # Read the predictions CSV for download
        with open(predictions_csv_path, 'rb') as file:
            st.session_state.processed_predictions_csv = file.read()

        st.download_button(label="Download Predictions CSV", data=st.session_state.processed_predictions_csv, file_name="processed_predictions.csv", mime="text/csv")

        # Trim the video based on predictions
        trimmed_video_path = trim_video(predictions_csv_path, st.session_state['output_video_path'], os.path.join(st.session_state['stable_temp_path'], 'trimmed_video.mp4'))
        st.session_state.trimmed_video = trimmed_video_path

    else:
        st.error("CSV output path is not available. Please process a video first.")

# Button to compare videos
compare_videos_button = st.button("Compare Original and Trimmed Videos")

if compare_videos_button:
    if uploaded_file is not None and st.session_state.trimmed_video:
        # Display original video
        st.write("Original Video:")
        st.video(st.session_state['output_video_path'])

        # Display trimmed video
        st.write("Trimmed Video:")
        st.video(st.session_state.trimmed_video)
    else:
        st.error("Please upload a video, make predictions, and trim the video before comparing.")

# ... [existing code for processing and predictions] ...

# Button to calculate IoU
calculate_iou_button = st.button("Calculate IoU")

if calculate_iou_button:
    if 'csv_output_path' in st.session_state and 'ground_truth_path' in st.session_state:
        # Call the comparison function
        accuracy, comparison_df = compare_predictions(st.session_state['csv_output_path'], st.session_state['ground_truth_path'])
        
        # Display the accuracy as a percentage
        st.write(f"IoU Accuracy: {accuracy * 100:.2f}%")
    else:
        st.error("Please ensure both prediction CSV and ground truth CSV are available before calculating IoU.")


# ... [rest of your code] ...