import pandas as pd
import logging
import os

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def validate_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ['gender', 'subscription_type', 'contract_length']
    for col in categorical_cols:
        if col in data.columns:
            data.loc[:, col] = data[col].str.strip().str.lower()

    data = data.dropna(subset=['customerid']).copy()

    data['customerid'] = data['customerid'].astype(int)
    data['churn'] = data['churn'].astype(int)

    data = data[data['churn'].isin([0, 1])]

    non_neg = ['tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'age', 'total_spend']
    for col in non_neg:
        if col in data.columns:
            data.loc[data[col] < 0, col] = 0
    return data


def main():
    logging.basicConfig(filename='validation_report.txt', level=logging.INFO, force=True)

    try:
        df1 = pd.read_csv(f"{RAW_DATA_PATH}/part1.csv")
        df2 = pd.read_csv(f"{RAW_DATA_PATH}/part2.csv")
    except FileNotFoundError:
        print(f"Error: ensure part1.csv and part2.csv are in {RAW_DATA_PATH}")
        return

    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    print(f"combined dataset size: {df.shape[0]} rows")

    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df_clean = validate_clean_data(df)

    output_file = f"{PROCESSED_DATA_PATH}/dataset_cleaned.csv"
    df_clean.to_csv(output_file, index=False)

    print(f"complete: Saved to {output_file}")


if __name__ == "__main__":
    main()
