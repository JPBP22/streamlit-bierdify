import joblib
import numpy as np
import pandas as pd

model_path = r"C:\Users\struc\Documents\GitHub\streamlit-bierdify\straemlit-bierdify\modelv7.joblib"

def load_model(model_path):
    # Load and return the model
    model = joblib.load(model_path)
    return model

def make_predictions(model, features):
    # Make predictions using the model
    predictions = model.predict(features)
    return predictions

def calculate_average_frame_counts(df):
    # Group by predictions and count frames
    frame_counts = df.groupby(['predictions']).size().reset_index(name='counts')
    # Calculate average frame count for each prediction label across all videos
    average_frame_counts = frame_counts.groupby('predictions')['counts'].mean().to_dict()
    return average_frame_counts

def refine_predictions(predictions, averages):
    refined = predictions.copy()
    avg_0 = int(averages.get(0, 20))  # Default to 20 if average not found
    avg_2 = int(averages.get(2, 30))  # Default to 30 if average not found

    for i in range(1, len(predictions) - 1):
        if predictions[i] == 1 and predictions[i + 1] == 1:
            # Convert the frame before to 1
            refined[i - 1] = 1

            # Convert the previous frames to 0
            start_index = max(i - 1 - avg_0, 0)
            refined[start_index:i - 1] = [0] * (i - 1 - start_index)

            # Convert the next frames to 2
            end_index = min(i + 2 + avg_2, len(predictions))
            refined[i + 2:end_index] = [2] * (end_index - i - 2)

    return refined

def refine_predictions_with_label_1_rule(predictions):
    refined = predictions.copy()

    # Iterate over the predictions
    i = 0
    while i < len(refined) - 4:
        if refined[i] == 0:
            # Convert the relevant part of the array to a list to use the index method
            refined_list = list(refined[i:])

            try:
                # Find the index of the next 2 after the last 0
                next_2_index = refined_list.index(2)
            except ValueError:
                break  # No more 2's found, exit loop

            last_0_index = i + refined_list[:next_2_index].count(0) - 1
            first_2_index = last_0_index + 1 + refined_list[last_0_index + 1:next_2_index].count(1)

            # Ensure there are at least four 1's between the last 0 and the first 2
            num_ones_needed = 4 - (first_2_index - last_0_index - 1)
            if num_ones_needed > 0:
                refined[last_0_index + 1:last_0_index + 1 + num_ones_needed] = [1] * num_ones_needed

            i = first_2_index + i  # Continue from the first 2 index
        else:
            i += 1

    return refined

def combined_refinement(predictions, averages):
    # First, apply the label 1 specific rule
    refined = refine_predictions_with_label_1_rule(predictions)

    # Then, apply the refinement based on averages
    refined = refine_predictions(refined, averages)

    return refined
