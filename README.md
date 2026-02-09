Customer Churn Prediction: End-to-End Production Pipeline

Project Overview
This project demonstrates a production-grade machine learning system designed to predict customer churn. Moving beyond simple model training, this repository implements a complete MLOps lifecycle:
Data Integrity: Automated validation of raw inputs.
Feature Engineering: Domain-driven transformation of behavioral signals.
Experiment Tracking: Systematic model comparison and logging via MLflow.
Scalable Deployment: Containerized inference service using FastAPI and Docker.

Project Structure
The project follows the src/ layout, a standard for production Python applications to ensure clean imports and modularity.
customer_prediction/
├── data/               # Raw and processed datasets (Git ignored)
├── model/              # Local storage for winning .pkl artifacts
├── src/                # Core Logic
│   ├── api.py          # FastAPI application
│   ├── feature_eng.py  # Engineering logic
│   ├── validation.py   # Data quality checks
│   └── train.py        # MLflow-tracked training & selection
├── mlruns/             # MLflow local database for experiments
├── Dockerfile          # Container recipe
├── requirements.txt    # Managed dependencies
└── README.md           # Documentation


Step 1: Experiment Tracking with MLflow
Instead of manual logging, we use MLflow to manage the "winner" between models. Every training run is recorded as a scientific entry.
Implementation
Nested Runs: We wrap our model competition in a "Parent" run (e.g., comparison) containing "Child" runs for individual models (Logistic Regression vs. Decision Tree).
Metadata Tracked:
Parameters: max_depth, max_iter, feature_version.
Metrics: val_auc, test_auc.
Artifacts: The entire Pipeline object (Scaler + Encoder + Model) is logged to ensure consistency.
Winning Model: Decision Tree reached a 0.9370 AUC on the test set.

Step 2: Containerization with Docker
To ensure the model runs identically in any environment (Local, Cloud, or CI/CD), the entire API is encapsulated in a Docker Image.
The Docker Workflow
Build: Freezes the environment, libraries, and model into a read-only Image.
docker build -t churn-api .
Run: Launches an isolated Container that handles the prediction service.
docker run -p 8000:8000 churn-api

Setup & Usage 
Local Environment
# Install dependencies
pip install -r requirements.txt
# Start MLflow UI to view experiment history
mlflow ui
View experiments at http://localhost:5000

API Inference
With the Docker container running, the API is available for real-time predictions.
URL: http://localhost:8000/docs
Endpoint: POST /predict
Example Request:

JSON
{
  "tenure": 12,
  "support_calls": 2,
  "payment_delay": 0,
  "total_spend": 500,
  "usage_per_tenure": 5.5,
  "is_unhappy": 0,
  "is_high_value": 1,
  "is_inactive": 0,
  "gender": "male",
  "contract_length": "monthly",
  "age_bucket": "31-40"
}

Key Insights & Results
Behavioral Dominance: support_calls and payment_delay were the strongest churn indicators.
Interpretation over Complexity: The Decision Tree provided competitive AUC (~0.94) while allowing business stakeholders to visualize the decision rules.
Cost Optimization: By shifting from class labels to probabilities, we optimized the threshold to minimize the business cost of False Negatives (lost customers).