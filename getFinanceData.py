import yfinance as yf
import pandas as pd

def get_finance_data(one_sym, two_sym, three_sym, period='1d', interval='5m'):
    symbols = [one_sym, two_sym, three_sym]
    result = []

    for sym in symbols:
        df = yf.download(sym, period=period, interval=interval)
        df.reset_index(inplace=True)
        df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df.to_csv(f"{sym}_data.csv", index=False)
        result.append(df)
    return result[0], result[1], result[2]
