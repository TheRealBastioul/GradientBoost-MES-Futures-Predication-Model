import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import os

def showCharts(futuredata):
    clean_datetime = futuredata['datetime'].str.replace(' EDT', '')
    futuredata['datetime'] = pd.to_datetime(clean_datetime, format='%Y-%m-%d %I:%M %p')
    futuredata = futuredata.sort_values('datetime')

    # Candlestick trace
    candle = go.Candlestick(
        x=futuredata['datetime'],
        open=futuredata['Open'],
        high=futuredata['High'],
        low=futuredata['Low'],
        close=futuredata['Close'],
        name='Candlesticks'
    )

    # Close price line trace
    close_line = go.Scatter(
        x=futuredata['datetime'],
        y=futuredata['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=1)
    )

    # Build the figure
    fig = go.Figure(data=[candle, close_line])

    # Update layout for a modern look
    utc_now = datetime.now(pytz.utc)
    est = pytz.timezone('US/Eastern')
    est_now = utc_now.astimezone(est)

    formatted_time = est_now.strftime('%Y-%m-%d-%I%M-%p-%Z')
    current_month_day = est_now.strftime('%m-%d')
    os.makedirs(current_month_day, exist_ok=True)
    print(formatted_time)
    fig.update_layout(
        title=f"Generated on {current_month_day}/{formatted_time}",
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,  # hide zoom slider
        template='plotly_white',
        height=600,
        margin=dict(l=50, r=25, t=50, b=40),
        xaxis=dict(
            tickformat='%Y-%m-%d %I:%M %p',  # Changed from %H:%M to %I:%M %p for 12-hour format
            tickmode='auto'
        )
    )

    fig.write_html(f"{current_month_day}/{formatted_time}_Prediction.html")
    fig.show()
