import pandas as pd

def read_csv(file_path):
    return pd.read_csv(file_path)

def compare_predictions(predictions_csv_path, ground_truth_csv_path):
    # Read CSV files
    predictions_df = read_csv(predictions_csv_path)
    ground_truth_df = read_csv(ground_truth_csv_path)

    # Ensure both DataFrames are sorted by frame index (assuming a column 'frame_index' exists)
    predictions_df.sort_values('frame_index', inplace=True)
    ground_truth_df.sort_values('frame_index', inplace=True)

    # Merge the DataFrames on frame index
    merged_df = pd.merge(predictions_df, ground_truth_df, on='frame_index', suffixes=('_pred', '_gt'))

    # Compare predictions with ground truth and calculate accuracy
    merged_df['correct_prediction'] = merged_df['predictions'] == merged_df['label']
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
