import shap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


model_path = "model/churn_pipeline.pkl"
data_path = "data/processed/dataset_features.csv"

def run_shap_explanation():
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("missing model or data files")
        return

    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    FEATURES = ['tenure', 'support_calls', 'payment_delay', 'total_spend',
                'usage_per_tenure', 'is_unhappy', 'is_high_value',
                'is_inactive', 'gender', 'contract_length', 'age_bucket']

    X = df[FEATURES]
    X_processed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    print("calculating SHAP values")
    background = shap.sample(X_processed, 100, random_state=42)  # Smaller sample for speed
    explainer = shap.Explainer(model, background)

    X_explain = shap.sample(X_processed, 500, random_state=42)
    shap_values = explainer(X_explain)

    shap_churn = shap_values.values[:, :, 1]
    mean_abs_shap = np.abs(shap_churn).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][:15][::-1], importance_df["mean_abs_shap"][:15][::-1])
    plt.title("Global Feature Importance (SHAP)")
    plt.xlabel("Impact on Churn Prediction")

    plt.savefig("model/shap_importance.png")
    print("Saved SHAP global importance to model/shap_importance.png")
    plt.show()

def main():
    run_shap_explanation()

if __name__ == "__main__":
    main()