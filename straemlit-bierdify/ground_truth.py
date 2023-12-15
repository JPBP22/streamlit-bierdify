import pandas as pd

def read_csv(file_path):
    return pd.read_csv(file_path)

def compare_predictions(predictions_csv_path, ground_truth_csv_path):
    # Read CSV files
    predictions_df = read_csv(predictions_csv_path)
    ground_truth_df = read_csv(ground_truth_csv_path)

    # Column names for frame index in each CSV
    predictions_frame_column = 'frame_index'
    ground_truth_frame_column = 'frame'

    # Ensure both DataFrames are sorted by their respective frame index
    predictions_df.sort_values(predictions_frame_column, inplace=True)
    ground_truth_df.sort_values(ground_truth_frame_column, inplace=True)

    # Merge the DataFrames on their respective frame index columns
    merged_df = pd.merge(predictions_df, ground_truth_df, left_on=predictions_frame_column, right_on=ground_truth_frame_column, suffixes=('_pred', '_gt'))

    # Compare predictions with ground truth and calculate accuracy
    merged_df['correct_prediction'] = merged_df['prediction'] == merged_df['label']
    accuracy = merged_df['correct_prediction'].mean()

    return accuracy, merged_df

def main(predictions_csv_path, ground_truth_csv_path):
    accuracy, comparison_df = compare_predictions(predictions_csv_path, ground_truth_csv_path)
    print(f"Accuracy of predictions: {accuracy * 100:.2f}%")
    # Optionally, save or return the comparison DataFrame
    # comparison_df.to_csv('comparison_results.csv', index=False)
    return comparison_df

if __name__ == "__main__":
    # Example usage
    predictions_path = 'path/to/predictions.csv'
    ground_truth_path = 'path/to/ground_truth.csv'
    comparison_df = main(predictions_path, ground_truth_path)
