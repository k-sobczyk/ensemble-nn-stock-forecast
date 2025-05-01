import warnings
from lstm.execute import run_lstm_model

warnings.filterwarnings("ignore")


def main():
    DATA_FILE = r'C:\\Users\\ksobc\\PycharmProjects\\ensemble-nn-stock-forecast\\src\\models\\model_with_features.csv'
    DATE_COLUMN = 'end_of_period'
    TARGET_COLUMN = 'target'
    TICKER_COLUMN = 'ticker'
    TEST_YEARS = [2021, 2022]
    MIN_SAMPLES_PER_COMPANY = 4
    N_TRIALS = 50
    OPTUNA_TIMEOUT = 3600

    run_lstm_model(
        data_file=DATA_FILE,
        date_column=DATE_COLUMN,
        target_column=TARGET_COLUMN,
        ticker_column=TICKER_COLUMN,
        test_years=TEST_YEARS,
        min_samples_per_company=MIN_SAMPLES_PER_COMPANY,
        n_trials=N_TRIALS,
        optuna_timeout=OPTUNA_TIMEOUT
    )


if __name__ == "__main__":
    main()
