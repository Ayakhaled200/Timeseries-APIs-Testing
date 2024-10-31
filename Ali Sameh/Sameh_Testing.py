import httpx
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
import os

# Define the API endpoint
API_URL = "http://127.0.0.1:5000/predict"

test_data_folder = r"H:\Giza Systems\Time Series\Ali Sameh\test_splits"
data_ids_lags = pd.read_csv(
    r"H:\Giza Systems\Time Series\Ali Sameh\input_lengths (1).csv")

dataset_results = []
results = []

# Initialize HTTP client
with httpx.Client() as client:
    # Iterate through each dataset file in the test folder
    for index, row in data_ids_lags.iterrows():
        dataset_id = int(row['dataset_id'].replace('train_', ''))  # Taking only the number part
        lag = row['input_length']

        dataset_path = os.path.join(test_data_folder, f"test_{dataset_id}.csv")

        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found.")
            continue

        df = pd.read_csv(dataset_path)

        # Limit the dataset to the first 1000 rows, if it has more
        df = df.head(1000)

        # Replace NaNs with None (null) which represents null in databases or JSON
        df.replace(np.nan, None, inplace=True)

        predictions = []
        actual_values = []

        # Iterate in blocks defined by the lag value
        for i in range(lag, len(df)):
            # Construct payload for each prediction based on lagged values
            chunk = df.iloc[i - lag:i]
            actual_value = df.iloc[i]['value']
            actual_values.append(actual_value)

            # Check if the "anomaly" column exists in the dataset
            if "anomaly" in chunk.columns:
                # If "anomaly" column exists, include it in the payload
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
                # If "anomaly" column does not exist, exclude it from the payload
                payload = {
                    "dataset_id": int(dataset_id),
                    "values": [
                        {
                            "timestamp": str(row["timestamp"]),
                            "value": row["value"]
                        } for _, row in chunk.iterrows()
                    ]
                }

            # Send the POST request to the API
            start_time = time.time()
            response = client.post(API_URL, json=payload)
            latency = time.time() - start_time

            if response.status_code == 200:
                # Get the predicted value from the API response
                prediction = response.json().get('prediction')
                predictions.append(prediction)
                # Append latency for tracking
                dataset_results.append({
                    "dataset_id": dataset_id,
                    "latency": latency
                })
            else:
                print(f"Request failed with status code {response.status_code} for dataset {dataset_id}")
                predictions.append(None)

        print(len(predictions), " == ", len(actual_values))

        # not including null in both predicition and actual
        aligned_actual_values = [val for val, pred in zip(actual_values, predictions) if
                                 val is not None and pred is not None and not np.isnan(val)]
        aligned_predictions = [pred for val, pred in zip(actual_values, predictions) if
                               val is not None and pred is not None and not np.isnan(val)]

        # Calculate MSE (Mean Squared Error) for the entire dataset
        if len(aligned_predictions) == len(aligned_actual_values) and len(predictions) > 0:
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
output_path = r"H:\Giza Systems\Time Series\Ali Sameh\Sameh_model_performance_report_1000row2.xlsx"
all_results_df.to_excel(output_path, index=False)

print(f"Model performance report generated: {output_path}")
