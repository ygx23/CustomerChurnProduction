import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

file_path = "data/processed/dataset_cleaned.csv"

def run_eda(df: pd.DataFrame):

    print(f"records: {len(df):,}")
    print(f"total features: {df.shape[1]}")
    churn_num = pd.to_numeric(df['churn'], errors='coerce')
    print(f"churn rate: {churn_num.mean():.2%}")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='churn', y='last_interaction')
    plt.title('Days Since Last Interaction by Churn')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='last_interaction', hue='churn', fill=True, common_norm=False)
    plt.title('Last Interaction Density Pattern')
    plt.xlabel('Days Since Last Interaction')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='churn', y='total_spend')
    plt.title('Total Spend Distribution by Churn')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='total_spend', hue='churn', fill=True, common_norm=False)
    plt.title('Total Spend Density: Churn vs. No Churn')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(ax=ax[0], data=df, x='gender', y='payment_delay', hue='churn')
    ax[0].set_title('Payment Delay by Gender & Churn Status')

    sns.boxplot(ax=ax[1], data=df, x='gender', y='age', hue='churn')
    ax[1].set_title('Age Distribution by Gender & Churn Status')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='age', hue='churn', fill=True, common_norm=False)
    plt.title('Age Distribution Density: Churn vs. No Churn')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.show()

    df["churn"] = df["churn"].astype(str)
    numerical_features = ["usage_frequency", "total_spend"]

    for col in numerical_features:
        df.boxplot(column=col, by="churn")
        plt.title(f"{col} by churn")
        plt.suptitle("")
        plt.xlabel("churn")
        plt.ylabel(col)
        plt.show()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='tenure', hue='churn', fill=True, common_norm=False, palette='viridis')
    plt.title('Tenure Distribution: Churn vs. No Churn')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Density')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='tenure', hue='churn', fill=True, common_norm=False, palette='viridis')
    plt.title('Tenure Density: When do customers leave?')
    plt.xlabel('Tenure (Months)')
    plt.show()

    print(df.groupby('churn')['tenure'].describe())
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="churn", y="support_calls", data=df)
    plt.title("Support Calls Distribution by Churn")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x="churn", y="payment_delay", data=df)
    plt.title("Payment Delay by Churn")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.countplot(data=df, y='payment_delay', hue='churn', palette='viridis')
    plt.xlabel('Payment Delay')
    plt.ylabel('Churn Rate')
    plt.title('Churn Pattern by Payment Delay')
    plt.xticks(rotation=90)
    plt.show()

    filtered = df.groupby(['subscription_type', 'churn']).size().unstack()
    X = list(filtered.index)
    churn_0 = list(filtered.iloc[:, 0])
    churn_1 = list(filtered.iloc[:, 1])
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, churn_1, 0.4, label='Churn')
    plt.bar(X_axis + 0.2, churn_0, 0.4, label='Not Churn')
    plt.xticks(X_axis, X, rotation=45)
    plt.xlabel('subscription_type')
    plt.ylabel('Count')
    plt.title("Churn rate based on subscription type")
    plt.legend(loc='center right')
    plt.grid(axis='y')
    plt.show()

    tenure_churn = df.groupby(['gender', 'churn']).size().unstack()
    X = list(tenure_churn.index)
    churn_0 = list(tenure_churn.iloc[:, 0])
    churn_1 = list(tenure_churn.iloc[:, 1])
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, churn_1, 0.4, label='churn')
    plt.bar(X_axis + 0.2, churn_0, 0.4, label='Not Churn')
    plt.xticks(X_axis, X)
    plt.xlabel('gender')
    plt.ylabel('Count')
    plt.title("Gender wise churn rate")
    plt.legend(loc='center right')
    plt.grid(axis='y')
    plt.show()

    filtered = df.groupby(['contract_length', 'churn']).size().unstack()
    X = list(filtered.index)
    churn_0 = list(filtered.iloc[:, 0])
    churn_1 = list(filtered.iloc[:, 1])
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, churn_1, 0.4, label='Churn')
    plt.bar(X_axis + 0.2, churn_0, 0.4, label='Not Churn')
    plt.xticks(X_axis, X, rotation=45)
    plt.xlabel('contract_length')
    plt.ylabel('Count')
    plt.title("Churn rate based on contract length")
    plt.legend(loc='center right')
    plt.grid(axis='y')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='support_calls', hue='churn', palette='magma')
    plt.title('Churn by Number of Support Calls')
    plt.xlabel('Number of Support Calls')
    plt.ylabel('Customer Count')
    plt.show()

    contract_dist = df.groupby('contract_length')['churn'].value_counts(normalize=True).unstack()
    contract_dist.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'], figsize=(8, 5))
    plt.title('Churn Proportion by Contract Length')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    usage_counts = df['usage_frequency'].value_counts(normalize=True) * 100
    sns.barplot(
        x=usage_counts.index,
        y=usage_counts.values,
        hue=usage_counts.index,
        palette='Blues',
        legend=False)
    plt.ylabel('Percentage of Customers (%)')
    plt.title('Distribution of Usage Frequency')
    plt.xticks(rotation=90)
    plt.show()

    df['churn'] = pd.to_numeric(df['churn'], errors='coerce')
    age_bins = [0, 20, 30, 40, 50, 60, float('inf')]
    age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61+']
    df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    age_churn_rate = df.groupby('AgeGroup', observed=False)['churn'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=age_churn_rate.index, y=age_churn_rate.values, hue=age_churn_rate.index, palette='pastel', legend=False)
    plt.xlabel('Age Group')
    plt.ylabel('Churn Rate')
    plt.title('Churn Rate by Age Group')
    plt.xticks(rotation=45)
    plt.show()

    sns.countplot(data=df, x='churn')
    plt.title('Churn Distribution')
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(ax=axes[0], data=df, x='churn', y='tenure')
    sns.boxplot(ax=axes[1], data=df, x='churn', y='support_calls')
    sns.boxplot(ax=axes[2], data=df, x='churn', y='payment_delay')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='contract_length', hue='churn')
    plt.title('Churn by Contract Length')
    plt.show()

    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix[['churn']], annot=True, cmap='RdBu_r')
    plt.title('Correlations between Features and Churn')
    plt.show()

    df['support_call_density'] = df['support_calls'] / (df['tenure'] + 1)
    df['spend_velocity'] = df['total_spend'] / (df['tenure'] + 1)
    df['usage_per_tenure'] = df['usage_frequency'] / (df['tenure'] + 1)

    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 18)

    fig, axes = plt.subplots(4, 1, figsize=(10, 25))

    sns.kdeplot(data=df, x='support_call_density', hue='churn', fill=True, ax=axes[0])
    axes[0].set_title('Distribution of Support Call Density by Churn Status', fontsize=14)
    axes[0].set_xlabel('Calls per Month of Tenure', fontsize=12)

    sns.kdeplot(data=df, x='spend_velocity', hue='churn', fill=True, ax=axes[1])
    axes[1].set_title('Distribution of Spend Velocity by Churn Status', fontsize=14)
    axes[1].set_xlabel('Dollars Spent per Month of Tenure', fontsize=12)

    sns.kdeplot(data=df, x='usage_per_tenure', hue='churn', fill=True, ax=axes[2])
    axes[2].set_title('Distribution of Usage Frequency per Tenure by Churn Status', fontsize=14)
    axes[2].set_xlabel('Usage Frequency per Month of Tenure', fontsize=12)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask for the upper triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0, ax=axes[3])
    axes[3].set_title('Feature Correlation Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig('engineered_features_eda.png')


def main():
    if not os.path.exists(file_path):
        print(f"could not find {file_path}.")
        return

    df = pd.read_csv(file_path)
    run_eda(df)
    print("EDA script finished successfully.")

if __name__ == "__main__":
    main()