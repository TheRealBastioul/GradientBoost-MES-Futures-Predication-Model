# Financial Market Predictor

A sophisticated machine learning system for predicting financial market movements using XGBoost with advanced feature engineering and multi-step direct forecasting.

## Project Structure

```
├── run_it.py              # Main execution script
├── getFinanceData.py      # Data fetching from Yahoo Finance
├── engineerFeatures.py    # Feature engineering and transformations
├── getPrediction.py       # Machine learning prediction pipeline
├── showCharts.py          # Visualization and chart generation
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Required Dependencies

```bash
pip install pandas numpy yfinance xgboost scikit-learn plotly pytz
```

### Alternative Installation

Create a `requirements.txt` file:
```
pandas>=1.5.0
numpy>=1.21.0
yfinance>=0.2.0
xgboost>=1.6.0
scikit-learn>=1.1.0
plotly>=5.10.0
pytz>=2022.1
```

Then install:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete prediction pipeline:

```bash
python run_it.py
```

This will:
1. Fetch real-time data for MES=F, NQ=F, and JPY=X
2. Engineer 100+ features
3. Train XGBoost models
4. Generate predictions
5. Create interactive charts
6. Save results to CSV and HTML files

## Custom Prediction

```python
from getPrediction import get_predictions

# Generate 480 predictions (1 day of 5-minute bars)
predictions = get_predictions(
    sym_one='MES=F',
    sym_two='NQ=F', 
    sym_three='JPY=X',
    steps_ahead=480,
    forecast_horizon=8
)
```

**Parameters:**
- `sym_one`: Primary symbol to predict
- `sym_two`: Secondary correlated symbol  
- `sym_three`: Third correlated symbol
- `steps_ahead`: Number of future predictions to generate
- `forecast_horizon`: Model prediction depth (default: 8)

## Custom Data Fetching

```python
from getFinanceData import get_finance_data

# Fetch different time periods/intervals
mes_data, nq_data, jpy_data = get_finance_data(
    'MES=F', 'NQ=F', 'JPY=X',
    period='5d',    # 5 days of data
    interval='1m'   # 1-minute intervals
)
```

**Supported Periods:** `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

**Supported Intervals:** `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

## Output Files

- `{symbol}_data.csv` - Raw market data
- `engineered_enhanced.csv` - Feature-engineered dataset
- `{symbol}_future_predictions.csv` - Prediction results
- `{MM-DD}/{timestamp}_Prediction.html` - Interactive charts

## Disclaimer

⚠️ **Important**: This software is for educational and research purposes only. It is not financial advice. Trading involves substantial risk.

---

**Happy Trading! 📈**
