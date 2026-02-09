import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score


input_path= "data/processed/dataset_features.csv"
model = "model"
model_path = os.path.join(model, "churn_pipeline.pkl")

TARGET = 'churn'
numeric_features = [
    'tenure',
    'support_calls',
    'payment_delay',
    'total_spend',
    'usage_per_tenure']

binary_features = [
    'is_unhappy',
    'is_high_value',
    'is_inactive']

categorical_features = [
    'gender',
    'contract_length',
    'age_bucket']

FEATURES = numeric_features + binary_features + categorical_features

def pre_process():
    return ColumnTransformer(
        transformers=[
        ('num',StandardScaler(),numeric_features),
        ('bin','passthrough', binary_features),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features)])

def train_eval(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
X, y, test_size=0.30, stratify=y, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    preprocessor= pre_process()

    mlflow.set_experiment("comparison")

    with mlflow.start_run(run_name="winner"):

        mlflow.log_param("feature_set", "basic_eng")
        mlflow.log_param("total_features", len(FEATURES))

        with mlflow.start_run(run_name="logistic_regression", nested=True):
            lr_pipeline = Pipeline(
                steps=[
                    ('preprocess', preprocessor),
                    ('model', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1))])

            lr_pipeline.fit(X_train, y_train)
            val_preds_lr = lr_pipeline.predict_proba(X_val)[:, 1]
            val_auc_lr = roc_auc_score(y_val, val_preds_lr)

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_metric("val_auc", val_auc_lr)
            mlflow.sklearn.log_model(lr_pipeline, "lr_model")
            print(f"LR validation AUC: {val_auc_lr:.4f}")



        with mlflow.start_run(run_name='decisiom_tree', nested=True):
            tree_pipeline = Pipeline(
                steps=[
                    ('preprocess', preprocessor),
                    ('model', DecisionTreeClassifier( max_depth=6, min_samples_leaf=100, class_weight='balanced', random_state=42))])

            tree_pipeline.fit(X_train, y_train)
            val_preds_tree = tree_pipeline.predict_proba(X_val)[:, 1]
            val_auc_tree = roc_auc_score(y_val, val_preds_tree)

            mlflow.log_param("model_type", "DecisionTree")
            mlflow.log_metric("val_auc", val_auc_tree)
            mlflow.sklearn.log_model(tree_pipeline, "tree_model")
            print(f"Tree Validation AUC: {val_auc_tree:.4f}")


        if val_auc_lr > val_auc_tree:
            best_model = lr_pipeline
            model_name = "Logistic Regression"
            best_val_auc = val_auc_lr

        else:
            best_model = tree_pipeline
            model_name = "Decision Tree"
            best_val_auc = val_auc_tree

        test_preds = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_preds)

        mlflow.log_param("chosen_model", model_name)
        mlflow.log_metric("val_auc", best_val_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.sklearn.log_model(best_model, model_name)
        print(f"{model_name} and validation AUC: {test_auc:.4f}")


    return best_model


def main():
    if not os.path.exists(input_path):
        print(f'{input_path} not found')
        return

    df= pd.read_csv(input_path)
    X = df[FEATURES]
    y = df[TARGET]

    best_pipeline= train_eval(X,y)

    if not os.path.exists(model):
        os.makedirs(model)

    joblib.dump(best_pipeline, model_path)
    print(f"model saved successfully in {model_path}")

if __name__ == "__main__":
    main()