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

# Model Prediction.py functionalities:

This code includes functions for loading a machine learning model, making predictions, and refining these predictions based on specific rules and averages. Here's a summary of what each part does:

load_model function:

Loads a machine learning model from a specified file path (using joblib) and returns the loaded model.
make_predictions function:

Uses the loaded model to make predictions based on input features.
calculate_average_frame_counts function:

Groups a DataFrame by the 'predictions' column and calculates the average count of frames for each unique prediction label.
refine_predictions function:

Refines predictions by altering sequences of predicted labels. It ensures that:
If two consecutive frames are predicted as label '1', the frame before these is also set to '1'.
A certain number of frames before this sequence are set to '0' and a certain number of frames after are set to '2', based on average counts (avg_0 and avg_2).
refine_predictions_with_label_1_rule function:

Further refines predictions specifically focusing on the transition between labels '0' and '2'. It ensures that there are at least four '1's between the last '0' and the first '2' in any sequence.
combined_refinement function:

Combines the refinement steps by first applying the label '1' specific rule and then the general refinement based on averages.

# csv_processing.py functionalities:


This code processes a CSV file containing pose estimation data and calculates additional features and metrics. Here's a breakdown of its functionality:

Reading the CSV File:

The CSV file specified by csv_path is read into a pandas DataFrame.
Extracting Coordinates:

A function extract_coordinates is used to parse coordinate strings from the DataFrame, extracting numerical values for x, y, and z coordinates.
Processing Coordinate Columns:

The code loops through columns named bbox_0_coord_i (where i ranges from 0 to 16) and splits each coordinate into separate x, y, and z columns.
The original bbox_0_coord_i columns are then dropped from the DataFrame.
Additionally, z-coordinate columns are also dropped.
Calculating Relative Positions and Movements:

Various new features are calculated, including:
Distance between hands and head (hand_to_head_x, hand_to_head_y, hand_head_distance).
Distance between right and left hands (relative_distance).
Movement of right and left hands per frame (right_hand_movement, left_hand_movement).
Maximum hand movement over a rolling window of 100 frames (max_right_hand_movement, max_left_hand_movement).
Whether the current frame has the max hand movement (right_hand_max_movement_frame, left_hand_max_movement_frame).
Hand displacement and cumulative movement over 20 frames (right_hand_displacement, left_hand_displacement, right_hand_movement_20_frames, left_hand_movement_20_frames).
Frame Rate and Speed Calculations:

The code assumes a frame rate (30 fps in this case) and calculates the speed of hand movements over a 10-frame window.
Weighted Sum Calculation:

A weighted sum of various features is calculated based on predefined weights. This sum could be used for further analysis or as a feature in machine learning models.

# trimmer.py

Function Purpose:

trim_video trims a video file according to specified start and end frames derived from a predictions CSV file.
Parameters:

predictions_csv_path: The file path to the CSV containing frame-wise predictions.
video_path: The file path to the original video that needs to be trimmed.
output_video_path: The file path where the trimmed video will be saved.
buffer_frames (optional): A specified number of frames to add as a buffer at the beginning and end of the trimmed video segment. The default value is 5 frames.
Processing Steps:

The function loads the predictions from the CSV file into a DataFrame.
It identifies the first frame labeled as '0' and the last frame labeled as '2' within the predictions.
The start and end frames of the video to be trimmed are adjusted by adding/subtracting the buffer frames. It ensures the start frame does not go below zero.
The original video is loaded using the VideoFileClip class from the moviepy library.
The start and end times for trimming are calculated in seconds based on the video's frame rate (fps).
The video is then trimmed to these specified start and end times using the subclip method.
The trimmed video clip is saved to the specified output_video_path using the write_videofile method with 'libx264' codec.
Return Value:

The function returns the path to the saved trimmed video file.


