import pandas as pd
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import re
import django
import sys
import httpx  # Import httpx instead of requests

sys.path.append(r'H:\Giza Systems\Time Series\Ali Badawy\src\my_time_project')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_time_project.settings')
django.setup()

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/api/predict/"

# Folder where test datasets are stored
test_data_folder = r"H:\Giza Systems\Time Series\Ali Badawy\test_splits"

# Load the CSV that has dataset_id and num_of_values
num_of_values_df = pd.read_csv(r"H:\Giza Systems\Time Series\Ali Badawy\Dataset_inputs.csv")

# Initialize lists to store results
dataset_results = []
results = []

# Initialize the httpx client for persistent connections
with httpx.Client() as client:
    # Iterate through each dataset file in the test folder
    for index, row in num_of_values_df.iterrows():
        ds_id = row['Dataset_ID']
        lag = row['inference_ Input_Values']

        # Extract numeric dataset ID
        dataset_id = re.search(r'\d+', ds_id).group()
        dataset_path = os.path.join(test_data_folder, f"test_{dataset_id}.csv")

        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found.")
            continue

        # Read the dataset (assuming it's in CSV format)
        df = pd.read_csv(dataset_path)
        df = df.head(1000)

        # Replace NaNs with None (null) which represents null in databases or JSON
        df.replace(np.nan, None, inplace=True)

        predictions = []
        actual_values = []
        mse_list = []
        for i in range(lag, len(df)):
            # Construct payload for each prediction based on lagged values
            chunk = df.iloc[i - lag:i]
            actual_value = df.iloc[i]['value']
            actual_values.append(actual_value)

            # Build the payload based on whether "anomaly" column exists
            if "anomaly" in chunk.columns:
                payload = {
                    "dataset_id": int(dataset_id),
                    "values": [
                        {
                            "timestamp": str(row["timestamp"]),
                            "value": row["value"],
                            "anomaly": int(row["anomaly"])  # Include "anomaly" only if it exists
                        } for _, row in chunk.iterrows()
                    ]
                }
            else:
                payload = {
                    "dataset_id": int(dataset_id),
                    "values": [
                        {
                            "timestamp": str(row["timestamp"]),
                            "value": row["value"]
                        } for _, row in chunk.iterrows()
                    ]
                }

            # Send the POST request to the API using httpx
            start_time = time.time()
            response = client.post(API_URL, json=payload)
            latency = time.time() - start_time

            if response.status_code == 200:
                # Get the predicted value from the API response
                prediction = response.json().get('Prediction')
                predictions.append(prediction)

                # Append latency for tracking
                dataset_results.append({
                    "dataset_id": dataset_id,
                    "latency": latency
                })

        print(len(predictions), " -- ", len(actual_values))

        # Filter out None values from actual_values and predictions for alignment
        # This approach assumes that predictions has no None values, so it only filters out pairs
        # where actual_values is None or NaN, without checking for None in predictions.
        aligned_actual_values = [val for val, pred in zip(actual_values, predictions) if
                                 val is not None and not np.isnan(val)]
        aligned_predictions = [pred for val, pred in zip(actual_values, predictions) if
                               val is not None and not np.isnan(val)]

        print(len(aligned_predictions), " == ", len(aligned_actual_values))

        # Calculate MSE (Mean Squared Error) for the entire dataset
        if len(aligned_predictions) == len(aligned_actual_values):
            mse = mean_squared_error(aligned_actual_values, aligned_predictions, squared=True)
            average_latency = pd.DataFrame(dataset_results)['latency'].mean()

            # Store the results for each dataset
            results.append({
                "dataset_id": dataset_id,
                "mse": mse,
                "average_latency": average_latency
            })

            print(f"MSE for dataset {dataset_id}: {mse}")
        else:
            print(f"Mismatch in prediction and actual value lengths for dataset {dataset_id}")

# Convert results to a DataFrame for all datasets
all_results_df = pd.DataFrame(results)

# Save the results to an Excel file
output_path = r"H:\Giza Systems\Time Series\Ali Badawy\badawy_model_performance_report_1000row3.xlsx"
all_results_df.to_excel(output_path, index=False)

print(f"Model performance report generated: {output_path}")