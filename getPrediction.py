import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from getFinanceData import get_finance_data
from engineerFeatures import engineer_features
import numpy as np
from datetime import datetime, timedelta


def get_predictions(sym_one, sym_two, sym_three, steps_ahead=None, forecast_horizon=8):
    if steps_ahead is None:
        steps_ahead = 24 * 60 // 5  # Default: one day of 5-minute bars

    print(f"Generating {steps_ahead} total predictions in {forecast_horizon}-step direct forecast chunks...")
    oneish, twoish, threeish = get_finance_data(sym_one, sym_two, sym_three)
    data = engineer_features(oneish, twoish, threeish)
    data['timestamp'] = data['Datetime']
    data = data.sort_values('timestamp').reset_index(drop=True)

    features = [
        'PER', 'ISO', 'VII', 'VPR', 'p_up', 'p_down', 'cei', 'vaa', 'fps',
        'SMA', 'EMA', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower',
        'MACD', 'MACD_signal', 'VWAP',
        'Close_lag1', 'Volume_lag1', 'Close_lag2', 'Volume_lag2', 'Close_lag3', 'Volume_lag3',
        'Close_momentum_3', 'Volume_change_5', 'cum_return_5', 'is_up_trend', 'price_vol_corr',
        'return_3', 'volatility_5', 'range_spike', 'range_spike_flag',
        'near_upper_band', 'near_lower_band', 'breakout_pressure', 'MACD_slope',
        'bullish_divergence', 'price_acceleration', 'z_score_close', 'z_score_spike',
        'relative_volume', 'volume_spike', 'price_jump', 'jump_with_volume', 'chaikin_vol',
        'n_close', 'n_open', 'n_high', 'n_low', 'n_volume', 'j_close', 'j_open', 'j_high', 'j_low',
        'j_volume', 'n_momentum_diff', 'j_momentum_diff', 'n_lead_return', 'j_lead_return', 'mes_catchup',
        'n_spread', 'j_spread', 'spread_deviation', 'volume_momentum_ratio', 'agreement', 'momentum_entanglement',
        'cross_vol_weighted_strength', 'volatility_sync', 'directional_agreement_volume', 'momentum_ratio_oscillator'
    ]


    for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
        for lag in [1, 2, 3]:
            data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
            features.append(f'{feature}_lag_{lag}')

    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    features.extend(['hour', 'minute', 'day_of_week'])

    data = data.dropna().reset_index(drop=True)
    targets = ['Open', 'High', 'Low', 'Close', 'Volume']

    for step in range(1, forecast_horizon + 1):
        for t in targets:
            data[f'{t}_t+{step}'] = data[t].shift(-step)

    data = data.dropna().reset_index(drop=True)

    X = data[features]
    y_steps = {
        step: data[[f'{t}_t+{step}' for t in targets]]
        for step in range(1, forecast_horizon + 1)
    }

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 200,
        'tree_method': 'hist',
        'predictor': 'cpu_predictor',
        'verbosity': 0,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    }

    print("Training multi-step direct forecast models...")
    models = {}
    for step in range(1, forecast_horizon + 1):
        models[step] = {}
        for i, target in enumerate(targets):
            print(f"Training model for {target} t+{step}...")
            model = xgb.XGBRegressor(**params)
            model.fit(X, y_steps[step].iloc[:, i])
            models[step][target] = model

    current_timestamp = data['timestamp'].iloc[-1]
    lookback_window = max(64, forecast_horizon)
    prediction_data = data[data['timestamp'] <= current_timestamp].iloc[-lookback_window:].copy().reset_index(drop=True)

    future_predictions = []
    steps_made = 0

    while steps_made < steps_ahead:
        for step in range(1, forecast_horizon + 1):
            if steps_made >= steps_ahead:
                break

            input_row = prediction_data[features].iloc[[-1]].copy()
            pred_row = {'timestamp': current_timestamp + pd.Timedelta(minutes=step * 5)}

            for target in targets:
                pred_val = float(models[step][target].predict(input_row)[0])
                pred_row[target] = pred_val

            future_predictions.append(pred_row)
            steps_made += 1

        last_pred = future_predictions[-forecast_horizon:]
        new_rows = pd.DataFrame(last_pred)
        new_rows['month_day'] = new_rows['timestamp'].dt.strftime('%m-%d')
        new_rows['time'] = new_rows['timestamp'].dt.strftime('%I:%M %p').str.lstrip('0')
        new_rows['hour'] = new_rows['timestamp'].dt.hour
        new_rows['minute'] = new_rows['timestamp'].dt.minute
        new_rows['day_of_week'] = new_rows['timestamp'].dt.dayofweek

        extended_data = pd.concat([prediction_data, new_rows], ignore_index=True).reset_index(drop=True)
        for target in targets:
            for lag in [1, 2, 3]:
                col = f'{target}_lag_{lag}'
                extended_data[col] = extended_data[target].shift(lag)
                new_rows[col] = extended_data[col].iloc[len(prediction_data):].values

        close_returns = extended_data['Close'].pct_change().rolling(window=8).std().fillna(0)
        volume_returns = extended_data['Volume'].pct_change().rolling(window=8).std().fillna(0)

        # Add calculation of missing features for all new rows
        for idx in new_rows.index:
            open_price = new_rows.loc[idx, 'Open']
            close = new_rows.loc[idx, 'Close']
            high = new_rows.loc[idx, 'High']
            low = new_rows.loc[idx, 'Low']
            volume = new_rows.loc[idx, 'Volume']

            new_rows.loc[idx, 'PER'] = 0.5 * volume * (close - open_price) ** 2
            new_rows.loc[idx, 'ISO'] = ((close - low) - (high - close)) / (high - low + 1e-9)
            new_rows.loc[idx, 'VII'] = ((high - low) / (abs(close - open_price) + 1e-9)) * (1 / np.log10(volume + 1))
            new_rows.loc[idx, 'VPR'] = ((close - low) / (high - low + 1e-9)) * (volume / (volume + 1e-9))
            new_rows.loc[idx, 'p_up'] = (close - open_price) / (high - low + 1e-9)
            new_rows.loc[idx, 'p_down'] = 1 - new_rows.loc[idx, 'p_up']
            new_rows.loc[idx, 'cei'] = -(new_rows.loc[idx, 'p_up'] * np.log(np.clip(new_rows.loc[idx, 'p_up'], 1e-9, 1)) +
                                         new_rows.loc[idx, 'p_down'] * np.log(np.clip(new_rows.loc[idx, 'p_down'], 1e-9, 1)))
            new_rows.loc[idx, 'fps'] = np.abs((high - close) / (open_price - low + 1e-9) - volume / (close + 1e-9))

            recent_close = extended_data['Close'].iloc[max(0, idx + len(prediction_data) - 32):idx + len(prediction_data) + 1]
            new_rows.loc[idx, 'SMA'] = recent_close.rolling(window=8).mean().iloc[-1]
            new_rows.loc[idx, 'EMA'] = recent_close.ewm(span=8, adjust=False).mean().iloc[-1]

            delta = recent_close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=8).mean()
            avg_loss = loss.rolling(window=8).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            new_rows.loc[idx, 'RSI'] = 100 - (100 / (1 + rs.iloc[-1]))

            mean = recent_close.rolling(window=8).mean().iloc[-1]
            std = recent_close.rolling(window=8).std().iloc[-1]
            new_rows.loc[idx, 'BB_middle'] = mean
            new_rows.loc[idx, 'BB_upper'] = mean + 2 * std
            new_rows.loc[idx, 'BB_lower'] = mean - 2 * std

            # Add vaa calculation
            recent_volume = extended_data['Volume'].iloc[max(0, idx + len(prediction_data) - 8):idx + len(prediction_data) + 1]
            new_rows.loc[idx, 'vaa'] = volume / (recent_volume.mean() + 1e-9)

            # Add MACD and MACD_signal calculations
            recent_close_26 = extended_data['Close'].iloc[max(0, idx + len(prediction_data) - 25):idx + len(prediction_data) + 1]
            recent_close_12 = extended_data['Close'].iloc[max(0, idx + len(prediction_data) - 11):idx + len(prediction_data) + 1]
            ema_12 = recent_close_12.ewm(span=12, adjust=False).mean().iloc[-1]
            ema_26 = recent_close_26.ewm(span=26, adjust=False).mean().iloc[-1]
            new_rows.loc[idx, 'MACD'] = ema_12 - ema_26
            
            # Calculate MACD signal line (9-period EMA of MACD)
            if idx >= 8:
                macd_series = new_rows.loc[max(0, idx-8):idx, 'MACD']
                new_rows.loc[idx, 'MACD_signal'] = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
            else:
                new_rows.loc[idx, 'MACD_signal'] = new_rows.loc[idx, 'MACD']

            # Add VWAP calculation
            recent_data = extended_data.iloc[max(0, idx + len(prediction_data) - 8):idx + len(prediction_data) + 1]
            typical_price = (recent_data['High'] + recent_data['Low'] + recent_data['Close']) / 3
            cum_volume_price = (typical_price * recent_data['Volume']).sum()
            cum_volume = recent_data['Volume'].sum()
            new_rows.loc[idx, 'VWAP'] = cum_volume_price / (cum_volume + 1e-9)

            # Add Close_lag1, Volume_lag1 calculations
            if idx > 0:
                new_rows.loc[idx, 'Close_lag1'] = new_rows.loc[idx-1, 'Close']
                new_rows.loc[idx, 'Volume_lag1'] = new_rows.loc[idx-1, 'Volume']
            else:
                new_rows.loc[idx, 'Close_lag1'] = extended_data['Close'].iloc[-1]
                new_rows.loc[idx, 'Volume_lag1'] = extended_data['Volume'].iloc[-1]

            # Add Close_lag2, Volume_lag2 calculations
            if idx > 1:
                new_rows.loc[idx, 'Close_lag2'] = new_rows.loc[idx-2, 'Close']
                new_rows.loc[idx, 'Volume_lag2'] = new_rows.loc[idx-2, 'Volume']
            elif idx == 1:
                new_rows.loc[idx, 'Close_lag2'] = extended_data['Close'].iloc[-1]
                new_rows.loc[idx, 'Volume_lag2'] = extended_data['Volume'].iloc[-1]
            else:
                new_rows.loc[idx, 'Close_lag2'] = extended_data['Close'].iloc[-2] if len(extended_data) > 1 else extended_data['Close'].iloc[-1]
                new_rows.loc[idx, 'Volume_lag2'] = extended_data['Volume'].iloc[-2] if len(extended_data) > 1 else extended_data['Volume'].iloc[-1]

            # Add Close_lag3, Volume_lag3 calculations
            if idx > 2:
                new_rows.loc[idx, 'Close_lag3'] = new_rows.loc[idx-3, 'Close']
                new_rows.loc[idx, 'Volume_lag3'] = new_rows.loc[idx-3, 'Volume']
            else:
                lag3_idx = max(0, len(extended_data) - 3 + idx)
                new_rows.loc[idx, 'Close_lag3'] = extended_data['Close'].iloc[lag3_idx]
                new_rows.loc[idx, 'Volume_lag3'] = extended_data['Volume'].iloc[lag3_idx]

            # Add Close_momentum_3 calculation
            if idx >= 2:
                close_3_ago = new_rows.loc[idx-2, 'Close'] if idx >= 2 else extended_data['Close'].iloc[-3]
                new_rows.loc[idx, 'Close_momentum_3'] = (close - close_3_ago) / (close_3_ago + 1e-9)
            else:
                close_3_ago = extended_data['Close'].iloc[max(0, len(extended_data) - 3)]
                new_rows.loc[idx, 'Close_momentum_3'] = (close - close_3_ago) / (close_3_ago + 1e-9)

            # Add Volume_change_5 calculation
            recent_volume_5 = extended_data['Volume'].iloc[max(0, idx + len(prediction_data) - 4):idx + len(prediction_data) + 1]
            avg_volume_5 = recent_volume_5.mean()
            new_rows.loc[idx, 'Volume_change_5'] = (volume - avg_volume_5) / (avg_volume_5 + 1e-9)

            # Add cum_return_5 calculation
            if idx >= 4:
                close_5_ago = new_rows.loc[idx-4, 'Close'] if idx >= 4 else extended_data['Close'].iloc[-5]
                new_rows.loc[idx, 'cum_return_5'] = (close - close_5_ago) / (close_5_ago + 1e-9)
            else:
                close_5_ago = extended_data['Close'].iloc[max(0, len(extended_data) - 5)]
                new_rows.loc[idx, 'cum_return_5'] = (close - close_5_ago) / (close_5_ago + 1e-9)

            # Add is_up_trend calculation
            new_rows.loc[idx, 'is_up_trend'] = 1 if new_rows.loc[idx, 'SMA'] > new_rows.loc[idx, 'EMA'] else 0

            # Add price_vol_corr calculation
            recent_data_corr = extended_data.iloc[max(0, idx + len(prediction_data) - 8):idx + len(prediction_data) + 1]
            if len(recent_data_corr) > 3:
                new_rows.loc[idx, 'price_vol_corr'] = recent_data_corr['Close'].corr(recent_data_corr['Volume'])
                if pd.isna(new_rows.loc[idx, 'price_vol_corr']):
                    new_rows.loc[idx, 'price_vol_corr'] = 0
            else:
                new_rows.loc[idx, 'price_vol_corr'] = 0

            # Add return_3 calculation
            if idx >= 2:
                close_3_ago = new_rows.loc[idx-2, 'Close'] if idx >= 2 else extended_data['Close'].iloc[-3]
                new_rows.loc[idx, 'return_3'] = (close - close_3_ago) / (close_3_ago + 1e-9)
            else:
                close_3_ago = extended_data['Close'].iloc[max(0, len(extended_data) - 3)]
                new_rows.loc[idx, 'return_3'] = (close - close_3_ago) / (close_3_ago + 1e-9)

            # Add volatility_5 calculation
            recent_close_5 = extended_data['Close'].iloc[max(0, idx + len(prediction_data) - 4):idx + len(prediction_data) + 1]
            returns_5 = recent_close_5.pct_change().dropna()
            new_rows.loc[idx, 'volatility_5'] = returns_5.std() if len(returns_5) > 1 else 0

            # Add range_spike calculation
            avg_range = recent_data['High'].subtract(recent_data['Low']).mean()
            current_range = high - low
            new_rows.loc[idx, 'range_spike'] = (current_range - avg_range) / (avg_range + 1e-9)

            # Add range_spike_flag calculation
            new_rows.loc[idx, 'range_spike_flag'] = 1 if new_rows.loc[idx, 'range_spike'] > 1.5 else 0

            # Add near_upper_band and near_lower_band calculations
            bb_range = new_rows.loc[idx, 'BB_upper'] - new_rows.loc[idx, 'BB_lower']
            new_rows.loc[idx, 'near_upper_band'] = 1 if (new_rows.loc[idx, 'BB_upper'] - close) / (bb_range + 1e-9) < 0.1 else 0
            new_rows.loc[idx, 'near_lower_band'] = 1 if (close - new_rows.loc[idx, 'BB_lower']) / (bb_range + 1e-9) < 0.1 else 0

            # Add breakout_pressure calculation
            new_rows.loc[idx, 'breakout_pressure'] = (volume / (recent_volume.mean() + 1e-9)) * (current_range / (avg_range + 1e-9))

            # Add MACD_slope calculation
            if idx > 0 and 'MACD' in new_rows.columns:
                prev_macd = new_rows.loc[idx-1, 'MACD'] if idx > 0 else extended_data['MACD'].iloc[-1] if 'MACD' in extended_data.columns else 0
                new_rows.loc[idx, 'MACD_slope'] = new_rows.loc[idx, 'MACD'] - prev_macd
            else:
                new_rows.loc[idx, 'MACD_slope'] = 0

            # Add bullish_divergence calculation
            recent_rsi = extended_data['RSI'].iloc[max(0, idx + len(prediction_data) - 3):idx + len(prediction_data) + 1] if 'RSI' in extended_data.columns else pd.Series([new_rows.loc[idx, 'RSI']])
            rsi_trend = 1 if len(recent_rsi) > 1 and recent_rsi.iloc[-1] > recent_rsi.iloc[0] else 0
            price_trend = 1 if recent_close.iloc[-1] > recent_close.iloc[0] else 0
            new_rows.loc[idx, 'bullish_divergence'] = 1 if rsi_trend > price_trend else 0

            # Add price_acceleration calculation
            if idx >= 1:
                prev_momentum = new_rows.loc[idx-1, 'Close_momentum_3'] if idx > 0 else 0
                new_rows.loc[idx, 'price_acceleration'] = new_rows.loc[idx, 'Close_momentum_3'] - prev_momentum
            else:
                new_rows.loc[idx, 'price_acceleration'] = new_rows.loc[idx, 'Close_momentum_3']

            # Add z_score_close calculation
            close_std = recent_close.std()
            close_mean = recent_close.mean()
            new_rows.loc[idx, 'z_score_close'] = (close - close_mean) / (close_std + 1e-9)

            # Add z_score_spike calculation
            new_rows.loc[idx, 'z_score_spike'] = 1 if abs(new_rows.loc[idx, 'z_score_close']) > 2 else 0

            # Add relative_volume calculation
            new_rows.loc[idx, 'relative_volume'] = volume / (recent_volume.mean() + 1e-9)

            # Add volume_spike calculation
            new_rows.loc[idx, 'volume_spike'] = 1 if new_rows.loc[idx, 'relative_volume'] > 2 else 0

            # Add price_jump calculation
            new_rows.loc[idx, 'price_jump'] = abs(new_rows.loc[idx, 'z_score_close']) * new_rows.loc[idx, 'relative_volume']

            # Add jump_with_volume calculation
            new_rows.loc[idx, 'jump_with_volume'] = 1 if (new_rows.loc[idx, 'z_score_spike'] == 1 and new_rows.loc[idx, 'volume_spike'] == 1) else 0

            # Add chaikin_vol calculation
            hl_mean = recent_data[['High', 'Low']].mean(axis=1).mean()
            hl_current = (high + low) / 2
            new_rows.loc[idx, 'chaikin_vol'] = ((hl_current - hl_mean) / (hl_mean + 1e-9)) * new_rows.loc[idx, 'relative_volume']

            # Volatility context (avoid division by zero)
            vol_scale = max(new_rows.loc[idx, 'volatility_5'], 1e-6)

            # Recent trend dynamics
            jump_factor = new_rows.loc[idx, 'price_jump']
            accel = new_rows.loc[idx, 'price_acceleration']
            volume_spike = new_rows.loc[idx, 'volume_spike']

            # Normalize and clip contextual factors
            norm_vol = np.clip(vol_scale, 0.0001, 0.05)
            norm_jump = np.clip(jump_factor, -0.03, 0.03)
            norm_accel = np.clip(accel, -0.03, 0.03)
            norm_volume_spike = np.clip(volume_spike, 0, 5)

            # Amplified base scale: volatility-aware
            base_price_noise = 0.0333 * norm_vol  # was 0.033
            base_volume_noise = 0.033 * (1 + norm_volume_spike)

            # Add sharper influence from jump + acceleration
            raw_price_noise_scale = base_price_noise * (1 + 8.88 * (np.abs(norm_jump) + np.abs(norm_accel)))  # was 16.66
            price_noise_scale = np.clip(raw_price_noise_scale, 0, 0.08)

            # Apply heavier noise
            new_rows.loc[idx, 'Close'] += np.random.normal(0, price_noise_scale * close)
            new_rows.loc[idx, 'Open'] += np.random.normal(0, price_noise_scale * open_price)
            new_rows.loc[idx, 'High'] += np.random.normal(0, price_noise_scale * high)
            new_rows.loc[idx, 'Low'] += np.random.normal(0, price_noise_scale * low)
            new_rows.loc[idx, 'Volume'] += np.random.normal(0, base_volume_noise * volume)


            new_rows.loc[idx, 'Volume'] = max(new_rows.loc[idx, 'Volume'], 1)
            new_rows.loc[idx, 'Low'] = max(new_rows.loc[idx, 'Low'], 0)
            new_rows.loc[idx, 'Open'] = max(new_rows.loc[idx, 'Open'], 0)
            new_rows.loc[idx, 'Close'] = max(new_rows.loc[idx, 'Close'], 0)
            new_rows.loc[idx, 'High'] = max(new_rows.loc[idx, 'High'], new_rows.loc[idx, 'Low'])

        prediction_data = pd.concat([prediction_data, new_rows], ignore_index=True)
        prediction_data = prediction_data.iloc[-lookback_window:].reset_index(drop=True)
        current_timestamp = prediction_data['timestamp'].iloc[-1]

    future_df = pd.DataFrame(future_predictions)
    future_df['datetime'] = pd.to_datetime(future_df['timestamp'], utc=True)
    future_df['datetime'] = future_df['datetime'].dt.tz_convert('US/Eastern')
    future_df['datetime'] = future_df['datetime'].dt.strftime('%Y-%m-%d %I:%M %p %Z')
    future_df = future_df[['datetime', 'Close', 'Open', 'High', 'Low', 'Volume']]
    future_df.to_csv(f'{sym_one}_future_predictions.csv', index=False)

    print(f"Saved {len(future_df)} predictions to {sym_one}_future_predictions.csv")
    return future_df