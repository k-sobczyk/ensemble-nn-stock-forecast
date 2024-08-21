# Introduction


# Data
## Selected Features for Stock Price Prediction
This project uses the following features to predict stock prices based on changes in capital and asset structure:
Original Columns

1. date
2. company_id
3. sector

Asset Structure

4. Total Assets
5. Non-current assets
6. Current assets
7. Property, plant and equipment
8. Intangible assets
9. Inventories
10. Trade receivables
11. Cash and cash equivalents

Capital Structure

12. Equity shareholders of the parent
13. Share capital
14. Retained earnings / accumulated losses
15. Non-current liabilities
16. Current liabilities
17. Non-current loans and borrowings
18. Financial liabilities (loans and borrowings)

Market Data

19. Total shares
20. stock_price

Feature-Engineered Columns
For each of the asset and capital structure variables (columns 4-18), we've added four new columns representing the percentage change over different time periods:

Total number of columns after feature engineering: 79
(19 original columns + 15 variables * 4 change periods = 79)
[column_name]_change_1Q: Change over the last quarter
[column_name]_change_2Q: Change over the last two quarters (half-year)
[column_name]_change_4Q: Change over the last four quarters (year)
[column_name]_change_8Q: Change over the last eight quarters (two years)

For example, for 'Total Assets':
20. Total_Assets_change_1Q
21. Total_Assets_change_2Q
22. Total_Assets_change_4Q
23. Total_Assets_change_8Q

This pattern is repeated for all columns 4-18.
Total Feature Count
The final dataset contains 79 columns:

19 original columns
60 feature-engineered columns (15 variables * 4 change periods)

These features capture both the current financial state of companies and the dynamics of their asset and capital structures over different time horizons, providing a comprehensive basis for stock price prediction.

---


# LSTM

This document outlines three Long Short-Term Memory (LSTM) neural network architectures designed for predicting stock prices based on changes in company capital and asset structure.
The proposed models are:
1. Basic LSTM
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
```
* The basic LSTM model serves as a foundation for time series prediction tasks.
* It can capture long-term dependencies in sequential data, making it suitable for analyzing financial time series.
* The addition of dropout helps prevent overfitting, which is crucial when dealing with potentially noisy financial data.


2. Bidirectional LSTM (BiLSTM)
```python
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
```
* BiLSTM processes the input sequence in both forward and backward directions.
* This bidirectional approach allows the model to capture both past and future context at each time step.
* In financial markets, future stock prices can be influenced by both historical trends and future expectations, making BiLSTM particularly relevant.

3. Stacked LSTM
```python
class StackedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.0):
        super(StackedLSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_sizes[i-1], 
                    hidden_sizes[i], 
                    1, 
                    batch_first=True, 
                    dropout=dropout if i < len(hidden_sizes) - 1 else 0) 
            for i in range(len(hidden_sizes))
        ])
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
```
* Stacked LSTM allows for creating deeper networks with multiple LSTM layers.
* Each layer can learn different levels of abstraction from the data.
* In financial prediction, this can be beneficial for capturing hierarchical patterns in market behavior.

# CNN


# Transformer
