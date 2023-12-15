# Auto Trimmer for golf videos(Bierdify)

Our objective with this project was to create a user friendly interface where users can upload a video(golf swing ofcourse) and the app will automatically trim that video 
to only have the part where the golf swing is taking place.

To achieve this we have the app.py

# App.py functionalities:

Session State Initialization:

Initializes session variables to keep track of processed videos, CSV outputs, and trimmed videos.
Model Loading:

Loads a YOLO NAS Pose Large model with COCO pretrained weights for pose estimation. It checks for GPU availability and uses it if possible.
Video Processing Functions:

process_single_frame: Processes a single frame of the video, returning the processed frame and pose data.
create_csv: Creates a CSV file from the pose data list generated from the video.
create_video_from_frames: Compiles processed frames into a video.
process_video: Captures frames from the uploaded video, processes each frame, and then compiles them back into a processed video. It also creates a CSV with pose data.
Streamlit User Interface:

Title and file uploader for video and ground truth CSV.
Processes the uploaded video when the "Process Video" button is clicked. It shows a progress bar during processing.
Displays the processed video and provides download buttons for the video and the generated CSV file.
Prediction and Video Trimming:

When "Make Predictions" button is clicked, it processes the CSV to make predictions using a loaded model.
Downloads the CSV with predictions.
Trims the video based on predictions.
Video Comparison:

Compares the original and trimmed videos when "Compare Original and Trimmed Videos" button is clicked.
IoU Calculation:

Calculates the Intersection over Union (IoU) between the model predictions and ground truth when "Calculate IoU" button is clicked.


