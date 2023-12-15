# trimmer.py
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

def trim_video(predictions_csv_path, video_path, output_video_path, buffer_frames=5):
    """
    Trims a video based on prediction data.

    :param predictions_csv_path: Path to the CSV file with predictions.
    :param video_path: Path to the original video file.
    :param output_video_path: Path where the trimmed video will be saved.
    :param buffer_frames: Number of frames to buffer at start and end (default is 5).
    :return: Path to the trimmed video file.
    """

    # Load the predictions CSV
    predictions_df = pd.read_csv(predictions_csv_path)

    # Find the first frame labeled as 0 and the last frame labeled as 2
    first_frame_0 = predictions_df[predictions_df['predictions'] == 0]['frame_index'].min()
    last_frame_2 = predictions_df[predictions_df['predictions'] == 2]['frame_index'].max()

    # Adjust the start and end frames
    start_frame = max(first_frame_0 - buffer_frames, 0)  # Ensure the start frame is not negative
    end_frame = last_frame_2 + buffer_frames  # Add buffer frames to the end frame

    # Load the original video
    video_clip = VideoFileClip(video_path)

    # Calculate the start and end times in seconds
    fps = video_clip.fps  # Frames per second in the video
    start_time = start_frame / fps
    end_time = end_frame / fps

    # Trim the video
    trimmed_clip = video_clip.subclip(start_time, end_time)

    # Save the trimmed video
    trimmed_clip.write_videofile(output_video_path, codec='libx264')

    return output_video_path
