import httpx
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
import os

# Define the API endpoint
API_URL = "http://127.0.0.1:5000/predict"

test_data_folder = r"H:\Giza Systems\Time Series\Mohamed Ezzet\Time_Series_Task-main\test_splits"
data_ids = pd.read_csv(
    r"H:\Giza Systems\Time Series\Mohamed Ezzet\Time_Series_Task-main\data_ids.csv")

dataset_results = []
results = []

# Initialize HTTP client
with httpx.Client() as client:
    # Iterate through each dataset file in the test folder
    for index, row in data_ids.iterrows():
        dataset_id = row['dataset_id']
        lag = 4

        dataset_path = os.path.join(test_data_folder, f"test_{dataset_id}.csv")

        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found.")
            continue

        df = pd.read_csv(dataset_path)

        # Limit the dataset to the first 1000 rows, if it has more
        df = df.head(1000)

        # Replace NaNs with None (null) which represents null in databases or JSON
        df.replace(np.nan, None, inplace=True)
        # df.dropna(inplace=True)

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

        print(len(predictions), "==", len(actual_values))

        # Filter out None values from actual_values and predictions for alignment
        aligned_actual_values = [val for val, pred in zip(actual_values, predictions) if
                                 val is not None and pred is not None and not np.isnan(val)]
        aligned_predictions = [pred for val, pred in zip(actual_values, predictions) if
                               val is not None and pred is not None and not np.isnan(val)]
        print(len(aligned_predictions), " == ", len(aligned_actual_values))

        # Calculate MSE (Mean Squared Error) for the entire dataset
        # Check if aligned_actual_values and aligned_predictions are empty
        if len(aligned_actual_values) == 0 or len(aligned_predictions) == 0:
            print(f"Dataset {dataset_id} is empty or has no valid samples. Skipping MSE calculation.")
        else:
            # Calculate MSE if lists are not empty
            mse = mean_squared_error(aligned_actual_values, aligned_predictions, squared=True)
            average_latency = pd.DataFrame(dataset_results)['latency'].mean()

        results.append({
            "dataset_id": dataset_id,
            "mse": mse,
            "average_latency": average_latency
        })

        print(f"MSE for dataset {dataset_id}: {mse}")


# Convert results to a DataFrame for all datasets
all_results_df = pd.DataFrame(results)

# Save the results to an Excel file
output_path = r"H:\Giza Systems\Time Series\Mohamed Ezzet\Time_Series_Task-main\Ezzet_model_performance_report_1000row4.xlsx"
all_results_df.to_excel(output_path, index=False)

print(f"Model performance report generated: {output_path}")


"""
#code to make ids file for the datasets
"""
# import os
# import re
# import pandas as pd
#
# # Define the folder path where your files are stored
# folder_path = r"H:\Giza Systems\Time Series\Mohamed Ezzet\Time_Series_Task-main\test_splits"
#
# # List to store extracted numbers
# numbers = []
#
# # Iterate over each file in the folder
# for filename in os.listdir(folder_path):
#     # Use regular expression to find the number in the filename
#     match = re.search(r'test_(\d+)', filename)
#     if match:
#         # Extract the number and convert it to an integer
#         number = int(match.group(1))
#         numbers.append({"dataset_id": number})
#
# # Convert the list to a DataFrame and save to a CSV
# df = pd.DataFrame(numbers)
# output_csv = os.path.join(r"H:\Giza Systems\Time Series\Mohamed Ezzet\Time_Series_Task-main", "data_ids.csv")
# df.to_csv(output_csv, index=False)
#
# print(f"Extracted numbers saved to {output_csv}")
