import pandas as pd
import numpy as np
import os

input_path = "data/processed/dataset_cleaned.csv"
output_path = "data/processed/dataset_features.csv"
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df['is_unhappy'] = (df['support_calls'] >= 5).astype(int)
    df['is_high_value'] = (df['total_spend'] > 500).astype(int)
    # Severe payment issues
    df['is_delinquent'] = (df['payment_delay'] > 20).astype(int)

    def get_tenure_stage(months):
        if months <= 6:
            return 'danger_zone'
        elif months <= 18:
            return 'established'
        else:
            return 'veteran'

    df['tenure_stage'] = df['tenure'].apply(get_tenure_stage)

    df['is_inactive'] = (df['last_interaction'] > 15).astype(int)

    df['support_call_density'] = df['support_calls'] / (df['tenure'] + 1)

    df['spend_velocity'] = df['total_spend'] / (df['tenure'] + 1)

    df['usage_per_tenure'] = df['usage_frequency'] / (df['tenure'] + 1)

    contract_map = {
        'monthly': 3,
        'quarterly': 2,
        'annual': 1 }

    df['contract_stability'] = (
        df['contract_length']
        .str.lower()
        .map(contract_map)
        .fillna(2))

    df['age_bucket'] = pd.cut(
        df['age'],
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=['0-20', '21-30', '31-40', '41-50', '51-60', '61+'],
        right=False).astype(str)

    return df

def main():
    if not os.path.exists(input_path):
        (f'{input_path} not found')
        return

    df_clean = pd.read_csv(input_path)
    df_features = engineer_features(df_clean)
    df_features.to_csv(output_path, index=False)

    print([col for col in df_features.columns if col not in df_clean.columns])

if __name__== '__main__':
    main()



