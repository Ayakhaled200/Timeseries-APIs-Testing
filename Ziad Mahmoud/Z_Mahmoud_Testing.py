import httpx
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define the API endpoint
API_URL = "http://127.0.0.1:5000/predict"

# Folder where test datasets are stored
test_data_folder = r"E:\Giza Systems\Zaid_Mahmoud\Z_Mahmoud\test_splits"

# Load the CSV that has dataset_id and num_of_values
num_of_values_df = pd.read_csv(
    r"E:\Giza Systems\Zaid_Mahmoud\Z_Mahmoud\updated_dataset_ids.csv")
# Initialize lists to store results
dataset_results = []
results = []
predictions = []
# Initialize HTTP client
with httpx.Client() as client:
    # Iterate through each dataset file in the test folder
    for index, row in num_of_values_df.iterrows():
        dataset_id = int(row['dataset'])

        # Construct file path for each dataset
        dataset_path = os.path.join(test_data_folder, f"test_{dataset_id}.csv")

        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found.")
            continue

        # Read the dataset (assuming it's in CSV format)
        df = pd.read_csv(dataset_path)

        # Limit the dataset to the first 1000 rows, if it has more
        df = df.head(1000)

        # Replace NaNs with None (null) which represents null in databases or JSON
        df.replace(np.nan, None, inplace=True)

        # Prepare the actual values for comparison
        actual_values = df['value'].tolist()

        # Check if the "anomaly" column exists in the dataset
        if "anomaly" in df.columns:
            # If "anomaly" column exists, include it in the payload
            payload = {
                "dataset_id": int(dataset_id),
                "values": [
                    {
                        "timestamp": str(row["timestamp"]),
                        "value": row["value"],
                        "anomaly": int(row["anomaly"])  # Include "anomaly" only if it exists
                    } for _, row in df.iterrows()
                ]
            }
        else:
            # If "anomaly" column does not exist, exclude it from the payload
            payload = {
                "dataset_id": int(dataset_id),
                "values": [
                    {
                        "timestamp": str(row["timestamp"]),
                        "value": row["value"]
                    } for _, row in df.iterrows()
                ]
            }
        predictions = []
        # Send the POST request to the API
        start_time = time.time()
        with open(dataset_path, 'rb') as f:
            response = client.post(API_URL, files={'dataset_file': f})
            # response = client.post(API_URL, json=payload)
            latency = time.time() - start_time
        # Check response
        if response.status_code == 200:
            # Get the predicted values from the API response
            predictions = response.json().get('prediction')

            # Append latency for tracking
            dataset_results.append({
                "dataset_id": dataset_id,
                "latency": latency
            })
        else:
            print(f"Request failed with status code {response.status_code} for dataset {dataset_id}")

        print(len(predictions), " == ", len(actual_values))

       # Ensure predictions and actual_values are aligned and filter out None values
        aligned_actual_values = [val for val, pred in zip(actual_values, predictions)
                                 if val is not None and not np.isnan(val)]
        aligned_predictions = [pred for val, pred in zip(actual_values, predictions)
                               if val is not None and not np.isnan(val)]

         # Check if aligned lists are empty before calculating MSE
        if len(aligned_actual_values) == 0 or len(aligned_predictions) == 0:
            print(f"Dataset {dataset_id} is empty or has no valid samples. Skipping MSE calculation.")
        else:
            # Calculate MSE if lists are not empty
            mse = mean_squared_error(aligned_actual_values, aligned_predictions, squared=True)
            average_latency = pd.DataFrame(dataset_results)['latency'].mean()

            # Store the results for each dataset
            results.append({
                "dataset_id": dataset_id,
                "mse": mse,
                "average_latency": average_latency
            })
            print(f"MSE for dataset {dataset_id}: {mse}")

# Convert results to a DataFrame for all datasets
all_results_df = pd.DataFrame(results)

# Save the results to an Excel file
output_path = r"E:\Giza Systems\Zaid_Mahmoud\Z_Mahmoud\Z_Mahmoud_model_performance_report_1000row.xlsx"
all_results_df.to_excel(output_path, index=False)

print(f"Model performance report generated: {output_path}")




'''
adjust the names of the model files
'''
# import os
#
# # Specify the directory containing the files
# directory = r"E:\Giza Systems\Zaid_Mahmoud\Z_Mahmoud\models"  # Update this path to your actual directory
#
# # Loop through all files in the directory
# for filename in os.listdir(directory):
#     # Check if the filename starts with 'train_'
#     if filename.startswith("train_"):
#         # Construct the new filename by replacing 'train_' with 'test_'
#         new_filename = filename.replace("train_", "test_")
#
#         # Full file paths
#         old_file_path = os.path.join(directory, filename)
#         new_file_path = os.path.join(directory, new_filename)
#
#         # Rename the file
#         os.rename(old_file_path, new_file_path)
#         print(f"Renamed: {filename} -> {new_filename}")
#
