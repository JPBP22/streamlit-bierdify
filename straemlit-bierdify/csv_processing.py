import pandas as pd
import numpy as np
import re

def process_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Function to extract coordinates
    def extract_coordinates(coord_string):
        coord_string = str(coord_string)
        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', coord_string)
        # Ensure that three elements are always returned
        return [float(numbers[i]) if i < len(numbers) else None for i in range(3)]

    # Process each coordinate column
    for i in range(17):  # Adjust this range based on the number of bbox_0_coord columns
        col_name = f'bbox_0_coord_{i}'
        df[[f'{col_name}_x', f'{col_name}_y', f'{col_name}_z']] = df[col_name].apply(lambda x: pd.Series(extract_coordinates(x)))

    # Optionally, drop the original coordinate columns
    df.drop([f'bbox_0_coord_{i}' for i in range(17)], axis=1, inplace=True)

    # Dropping the z coordinate columns
    for i in range(17):  # Adjust this range based on the number of bbox_0_coord columns
        z_col_name = f'bbox_0_coord_{i}_z'
        if z_col_name in df.columns:
            df.drop(z_col_name, axis=1, inplace=True)

    # Calculating relative positions and other features
    df['hand_to_head_x'] = df['bbox_0_coord_4_x'] - df['bbox_0_coord_0_x']
    df['hand_to_head_y'] = df['bbox_0_coord_4_y'] - df['bbox_0_coord_0_y']
    df['hand_head_distance'] = ((df['hand_to_head_x']**2 + df['hand_to_head_y']**2)**0.5)
    df['relative_distance'] = np.sqrt((df['bbox_0_coord_4_x'] - df['bbox_0_coord_10_x']) ** 2 + (df['bbox_0_coord_4_y'] - df['bbox_0_coord_10_y']) ** 2)
    df['right_hand_movement'] = np.sqrt(df['bbox_0_coord_4_x'].diff() ** 2 + df['bbox_0_coord_4_y'].diff() ** 2)
    df['left_hand_movement'] = np.sqrt(df['bbox_0_coord_7_x'].diff() ** 2 + df['bbox_0_coord_7_y'].diff() ** 2)
    df['right_hand_movement'].fillna(0, inplace=True)
    df['left_hand_movement'].fillna(0, inplace=True)
    df['max_right_hand_movement'] = df['right_hand_movement'].rolling(window=100, min_periods=1).max()
    df['max_left_hand_movement'] = df['left_hand_movement'].rolling(window=100, min_periods=1).max()
    df['right_hand_max_movement_frame'] = df['right_hand_movement'] == df['max_right_hand_movement']
    df['left_hand_max_movement_frame'] = df['left_hand_movement'] == df['max_left_hand_movement']
    df['right_hand_displacement'] = np.sqrt((df['bbox_0_coord_4_x'].diff() ** 2) + (df['bbox_0_coord_4_y'].diff() ** 2))
    df['left_hand_displacement'] = np.sqrt((df['bbox_0_coord_7_x'].diff() ** 2) + (df['bbox_0_coord_7_y'].diff() ** 2))
    df['right_hand_movement_20_frames'] = df['right_hand_displacement'].rolling(window=20, min_periods=1).sum()
    df['left_hand_movement_20_frames'] = df['left_hand_displacement'].rolling(window=20, min_periods=1).sum()

    # Frame rate and other calculations
    frame_rate = 30  # Adjust based on your actual frame rate
    time_interval = 1 / frame_rate
    df['right_hand_movement_10_frames'] = df['right_hand_displacement'].rolling(window=10, min_periods=1).sum()
    df['left_hand_movement_10_frames'] = df['left_hand_displacement'].rolling(window=10, min_periods=1).sum()
    df['right_hand_speed_10_frames'] = df['right_hand_movement_10_frames'] / (10 * time_interval)
    df['left_hand_speed_10_frames'] = df['left_hand_movement_10_frames'] / (10 * time_interval)

    # Weighted sum calculation
    weights = {
        'hand_to_head_x': 1.0,
        'hand_to_head_y': 1.0,
        'hand_head_distance': 1.0,
        'relative_distance': 2.0,
        'right_hand_displacement': 3.0,
        'left_hand_displacement': 3.0,
        'right_hand_movement_20_frames': 6.0,
        'left_hand_movement_20_frames': 6.0,
        'max_right_hand_movement':3.0,
        'max_left_hand_movement':	3.0,
        'right_hand_max_movement_frame':5.0,
        'left_hand_max_movement_frame':5.0,
        'right_hand_movement_10_frames': 3.0,
        'left_hand_movement_10_frames':	3.0,
        'right_hand_speed_10_frames	': 8.0,
        'left_hand_speed_10_frames': 8.0,
    }
    df['weighted_sum'] = np.sum([df[col] * weights.get(col, 1.0) for col in df.columns], axis=0)

    return df
