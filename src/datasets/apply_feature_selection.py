import warnings

import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class FeatureSelector:
    def __init__(self, dataset_path='data/data_with_features.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()

    def load_data(self):
        print('Loading dataset...')
        self.df = pd.read_csv(self.dataset_path)

        exclude_cols = ['ticker', 'end_of_period', 'target_log']

        potential_feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        numeric_cols = []
        for col in potential_feature_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_cols.append(col)

        self.features = self.df[numeric_cols].copy()
        self.target = self.df['target_log'].copy()

        self.features = self.features.fillna(self.features.median())

        print(f'Dataset shape: {self.df.shape}')
        print(f'Total columns: {len(self.df.columns)}')
        print(f'Numeric feature columns: {len(numeric_cols)}')
        print(f'Excluded columns: {exclude_cols}')

        return self.features, self.target

    def calculate_feature_importance(self):
        print('\nCalculating feature importance...')

        print('- Random Forest...')
        features_scaled = self.scaler.fit_transform(self.features)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(features_scaled, self.target)
        rf_importance = rf.feature_importances_

        print('- Mutual Information...')
        mi_scores = mutual_info_regression(features_scaled, self.target, random_state=42)

        print('- Correlation Analysis...')
        correlations = []
        for col in self.features.columns:
            corr, _ = pearsonr(self.features[col], self.target)
            correlations.append(abs(corr))

        results_df = pd.DataFrame(
            {
                'feature': self.features.columns,
                'rf_importance': rf_importance,
                'mutual_info': mi_scores,
                'abs_correlation': correlations,
            }
        )

        results_df['rf_importance_norm'] = (results_df['rf_importance'] - results_df['rf_importance'].min()) / (
            results_df['rf_importance'].max() - results_df['rf_importance'].min()
        )
        results_df['mutual_info_norm'] = (results_df['mutual_info'] - results_df['mutual_info'].min()) / (
            results_df['mutual_info'].max() - results_df['mutual_info'].min()
        )
        results_df['abs_correlation_norm'] = (results_df['abs_correlation'] - results_df['abs_correlation'].min()) / (
            results_df['abs_correlation'].max() - results_df['abs_correlation'].min()
        )

        results_df['avg_importance'] = (
            results_df['rf_importance_norm'] + results_df['mutual_info_norm'] + results_df['abs_correlation_norm']
        ) / 3

        results_df = results_df.sort_values('avg_importance', ascending=False)

        return results_df

    def select_top_features(self, importance_df, n_features=15):
        top_features = importance_df.head(n_features)['feature'].tolist()

        print(f'\nTop {n_features} selected features:')
        print('-' * 50)
        for i, (_, row) in enumerate(importance_df.head(n_features).iterrows(), 1):
            print(f'{i:2d}. {row["feature"]:35s} (score: {row["avg_importance"]:.3f})')

        return top_features

    def create_reduced_dataset(self, selected_features, output_path='data/datasets/dataset_reduced_features.csv'):
        print(f'\nCreating reduced dataset with {len(selected_features)} features...')

        identifier_cols = ['ticker', 'end_of_period', 'target_log']
        final_columns = identifier_cols + selected_features

        reduced_df = self.df[final_columns].copy()

        reduced_df.to_csv(output_path, index=False)

        print(f'Reduced dataset saved to: {output_path}')
        print(f'Original features: {self.features.shape[1]}')
        print(f'Selected features: {len(selected_features)}')
        print(f'Reduction: {(1 - len(selected_features) / self.features.shape[1]) * 100:.1f}%')

        return reduced_df

    def run_feature_selection(self, n_features=15, output_path='data/datasets/dataset_reduced_features.csv'):
        print('=' * 60)
        print('SIMPLE FEATURE SELECTION')
        print('=' * 60)

        self.load_data()

        importance_df = self.calculate_feature_importance()

        selected_features = self.select_top_features(importance_df, n_features)

        reduced_df = self.create_reduced_dataset(selected_features, output_path)

        importance_path = output_path.replace('.csv', '_importance_ranking.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f'Feature importance ranking saved to: {importance_path}')

        return reduced_df, importance_df, selected_features


def main():
    selector = FeatureSelector()

    feature_counts = [10, 15, 20]

    for n_features in feature_counts:
        print(f'\n{"=" * 60}')
        print(f'CREATING DATASET WITH {n_features} FEATURES')
        print(f'{"=" * 60}')

        output_path = f'data/datasets/dataset_{n_features}_features.csv'
        reduced_df, importance_df, selected_features = selector.run_feature_selection(
            n_features=n_features, output_path=output_path
        )

    print(f'\n{"=" * 60}')
    print('FEATURE SELECTION COMPLETE')
    print('=' * 60)
    print('Created datasets:')
    for n in feature_counts:
        print(f'- data/datasets/dataset_{n}_features.csv')


if __name__ == '__main__':
    main()
