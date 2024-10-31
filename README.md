# Testing 10 APIs

This project benchmarks Testing 10 APIs made using Flask and Django frameworks, Each API is evaluated based on Mean Squared Error (MSE), latencyn across all datasets using various machine learning models.

### Project Overview

The main objective of this project is to test, evaluate, and compare 10 APIs created with Flask and Django. Each API handles different datasets and uses distinct models, providing detailed calculations of **MSE**, **latency**, and other relevant metrics for a comprehensive comparison of each framework’s performance with varying data and model complexity.

### Datasets

The project uses 96 datasets time series data that might take 2 formats, with different sizes, to test the performance of 2 machine/timeseries models on each API endpoint.
 
### API Structure

Each API represents an endpoint to test different models and datasets using Flask and Django:

1. **Endpoint format**: nearly 10 different formats
2. **Methods**: `GET` and `POST` supported, with `POST` for uploading datasets or configurations.
3. **Response format**: JSON with calculated MSE, latency, and other metrics.

### Metrics Calculated

The following metrics are calculated for each API request:

- **Mean Squared Error (MSE)**: Measures prediction accuracy.
- **Latency**: Measures response time.
