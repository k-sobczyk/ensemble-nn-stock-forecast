import matplotlib.pyplot as plt
import pandas as pd


def analyze_data_availability():
    df = pd.read_csv('src/model/individual/dataset_1_full_features.csv')
    df['end_of_period'] = pd.to_datetime(df['end_of_period'])
    df = df.sort_values(['ticker', 'end_of_period']).reset_index(drop=True)

    print(f'Dataset shape: {df.shape}')
    print(f'Date range: {df["end_of_period"].min()} to {df["end_of_period"].max()}')
    print(f'Unique companies: {df["ticker"].nunique()}')

    # Analyze data availability per ticker
    ticker_stats = []

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('end_of_period')

        # Count total periods
        total_periods = len(ticker_data)

        # Check for consecutive periods (quarterly data)
        dates = ticker_data['end_of_period'].dt.to_period('Q')
        consecutive_periods = []
        current_streak = 1

        for i in range(1, len(dates)):
            if dates.iloc[i] == dates.iloc[i - 1] + 1:  # Next quarter
                current_streak += 1
            else:
                consecutive_periods.append(current_streak)
                current_streak = 1
        consecutive_periods.append(current_streak)

        max_consecutive = max(consecutive_periods) if consecutive_periods else 0

        # Date range for this ticker
        start_date = ticker_data['end_of_period'].min()
        end_date = ticker_data['end_of_period'].max()

        # Check data before and after 2021
        pre_2021 = len(ticker_data[ticker_data['end_of_period'].dt.year < 2021])
        post_2021 = len(ticker_data[ticker_data['end_of_period'].dt.year >= 2021])

        ticker_stats.append(
            {
                'ticker': ticker,
                'total_periods': total_periods,
                'max_consecutive': max_consecutive,
                'start_date': start_date,
                'end_date': end_date,
                'years_span': (end_date - start_date).days / 365.25,
                'pre_2021_periods': pre_2021,
                'post_2021_periods': post_2021,
            }
        )

    stats_df = pd.DataFrame(ticker_stats)

    # Print summary statistics
    print('\n' + '=' * 50)
    print('DATA AVAILABILITY ANALYSIS')
    print('=' * 50)

    print('\nPeriods per company:')
    print(f'  Mean: {stats_df["total_periods"].mean():.1f}')
    print(f'  Median: {stats_df["total_periods"].median():.1f}')
    print(f'  Min: {stats_df["total_periods"].min()}')
    print(f'  Max: {stats_df["total_periods"].max()}')
    print(f'  Std: {stats_df["total_periods"].std():.1f}')

    print('\nMaximum consecutive periods per company:')
    print(f'  Mean: {stats_df["max_consecutive"].mean():.1f}')
    print(f'  Median: {stats_df["max_consecutive"].median():.1f}')
    print(f'  Min: {stats_df["max_consecutive"].min()}')
    print(f'  Max: {stats_df["max_consecutive"].max()}')
    print(f'  Std: {stats_df["max_consecutive"].std():.1f}')

    print('\nData span in years:')
    print(f'  Mean: {stats_df["years_span"].mean():.1f}')
    print(f'  Median: {stats_df["years_span"].median():.1f}')
    print(f'  Min: {stats_df["years_span"].min():.1f}')
    print(f'  Max: {stats_df["years_span"].max():.1f}')

    # Analyze sequence length options
    sequence_lengths = [3, 4, 5, 6, 8, 10, 12]
    print('\n' + '=' * 50)
    print('SEQUENCE LENGTH ANALYSIS')
    print('=' * 50)

    for seq_len in sequence_lengths:
        companies_with_enough_data = len(stats_df[stats_df['max_consecutive'] >= seq_len])
        percentage = (companies_with_enough_data / len(stats_df)) * 100
        print(
            f'Sequence length {seq_len:2d}: {companies_with_enough_data:3d} companies ({percentage:5.1f}%) have enough consecutive data'
        )

    # Recommend sequence length
    # Find sequence length where at least 80% of companies have enough data
    for seq_len in sequence_lengths:
        companies_with_enough_data = len(stats_df[stats_df['max_consecutive'] >= seq_len])
        percentage = (companies_with_enough_data / len(stats_df)) * 100
        if percentage >= 80:
            recommended_seq_len = seq_len
        else:
            break

    print('\n' + '=' * 50)
    print(f'RECOMMENDATION: Use sequence length {recommended_seq_len}')
    print(
        f'This allows {len(stats_df[stats_df["max_consecutive"] >= recommended_seq_len])} companies ({(len(stats_df[stats_df["max_consecutive"] >= recommended_seq_len]) / len(stats_df) * 100):.1f}%) to contribute to training'
    )
    print('=' * 50)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Distribution of total periods per company
    axes[0, 0].hist(stats_df['total_periods'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(
        stats_df['total_periods'].mean(),
        color='red',
        linestyle='--',
        label=f'Mean: {stats_df["total_periods"].mean():.1f}',
    )
    axes[0, 0].axvline(
        stats_df['total_periods'].median(),
        color='orange',
        linestyle='--',
        label=f'Median: {stats_df["total_periods"].median():.1f}',
    )
    axes[0, 0].set_xlabel('Total Periods per Company')
    axes[0, 0].set_ylabel('Number of Companies')
    axes[0, 0].set_title('Distribution of Total Periods per Company')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distribution of max consecutive periods
    axes[0, 1].hist(stats_df['max_consecutive'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(
        stats_df['max_consecutive'].mean(),
        color='red',
        linestyle='--',
        label=f'Mean: {stats_df["max_consecutive"].mean():.1f}',
    )
    axes[0, 1].axvline(
        stats_df['max_consecutive'].median(),
        color='orange',
        linestyle='--',
        label=f'Median: {stats_df["max_consecutive"].median():.1f}',
    )
    axes[0, 1].axvline(
        recommended_seq_len, color='purple', linestyle='-', linewidth=2, label=f'Recommended: {recommended_seq_len}'
    )
    axes[0, 1].set_xlabel('Max Consecutive Periods per Company')
    axes[0, 1].set_ylabel('Number of Companies')
    axes[0, 1].set_title('Distribution of Max Consecutive Periods')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Sequence length vs companies coverage
    coverage_data = []
    for seq_len in range(1, 16):
        companies_with_enough_data = len(stats_df[stats_df['max_consecutive'] >= seq_len])
        percentage = (companies_with_enough_data / len(stats_df)) * 100
        coverage_data.append({'seq_len': seq_len, 'coverage': percentage})

    coverage_df = pd.DataFrame(coverage_data)
    axes[1, 0].plot(coverage_df['seq_len'], coverage_df['coverage'], marker='o', linewidth=2, markersize=6)
    axes[1, 0].axhline(80, color='red', linestyle='--', label='80% threshold')
    axes[1, 0].axvline(
        recommended_seq_len, color='purple', linestyle='-', linewidth=2, label=f'Recommended: {recommended_seq_len}'
    )
    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Company Coverage (%)')
    axes[1, 0].set_title('Company Coverage vs Sequence Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 105)

    # Plot 4: Data availability over time
    # Count companies with data in each year
    years = range(df['end_of_period'].dt.year.min(), df['end_of_period'].dt.year.max() + 1)
    companies_per_year = []
    for year in years:
        companies_in_year = df[df['end_of_period'].dt.year == year]['ticker'].nunique()
        companies_per_year.append(companies_in_year)

    axes[1, 1].plot(years, companies_per_year, marker='o', linewidth=2, markersize=4)
    axes[1, 1].axvline(2021, color='red', linestyle='--', label='Test data start (2021)')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Number of Companies with Data')
    axes[1, 1].set_title('Data Availability Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_availability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return recommended_seq_len, stats_df


if __name__ == '__main__':
    recommended_seq_len, stats_df = analyze_data_availability()
    print(f'\nRecommended sequence length for LSTM: {recommended_seq_len}')
