import yfinance as yf
import numpy as np


def get_ticker(company_name):
    """
    Retrieves the ticker for a given company name assuming the company is listed on GPW.
    """
    ticker = company_name.replace(' ', '').replace('SA', '').upper() + '.WA'
    return ticker


def get_market_value(ticker, date):
    """
    Retrieves the average value from the opening and closing prices for the given ticker and date.
    """
    stock = yf.Ticker(ticker)
    try:
        data = stock.history(start=date, end=date)
        if not data.empty:
            open_price = data['Open'].iloc[0]
            close_price = data['Close'].iloc[0]
            return (open_price + close_price) / 2
        else:
            return None
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date}: {e}")
        return None


def get_possible_tickers(company_name):
    """
    Generates possible tickers for a company listed on GPW.
    """
    ticker_1 = company_name[:3].upper().replace(' ', '') + '.WA'
    ticker_2 = ''.join([word[0] for word in company_name.split()]).upper() + '.WA'

    return ticker_1, ticker_2
