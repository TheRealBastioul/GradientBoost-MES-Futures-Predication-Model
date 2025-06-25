import pandas as pd
import numpy as np
from getFinanceData import get_finance_data

def engineer_features(first_symbol, second_symbol, third_symbol, period='1d', interval='5m'):
    
    # Faster formula creation too lazy to type first_symbol[]
    close = first_symbol['Close']
    open = first_symbol['Open']
    high = first_symbol['High']
    low = first_symbol['Low']
    volume = first_symbol['Volume']
    n_close = second_symbol['Close']
    n_open = second_symbol['Open']
    n_high = second_symbol['High']
    n_low = second_symbol['Low']
    n_volume = second_symbol['Volume']
    j_close = third_symbol['Close']
    j_open = third_symbol['Open']
    j_high = third_symbol['High']
    j_low = third_symbol['Low']
    j_volume = third_symbol['Volume']
    n = 8            
    k = 2               
    short_period = 3   
    long_period = 8    
    signal_period = 5
    
    PI = np.pi  # Define PI for transformations

    # Original features with enhancements
    first_symbol['PER'] = (.5 * volume * (close - open)**2) * 1000 / PI
    first_symbol['ISO'] = (((close - low) - (high - close)) / (high - low)) * 1000 / PI**0.5
    first_symbol['VII'] = (((high - low) / (abs(close - open) + 1e-9)) * (1 / np.log10(volume + 1))) * 100
    first_symbol['VPR'] = (((close - low) / (high - low + 1e-9)) * (volume / volume.rolling(14).mean())) ** 3
    first_symbol['p_up'] = ((close - open) / (high - low + 1e-9)) * 1000 / PI
    first_symbol['p_down'] = (1 - first_symbol['p_up']) * 1000 / PI
    
    # Entropy-based feature - enhanced
    first_symbol['cei'] = -(first_symbol['p_up'] * np.log(np.clip(first_symbol['p_up'], 1e-9, 1)) + 
                           first_symbol['p_down'] * np.log(np.clip(first_symbol['p_down'], 1e-9, 1))) * 1000
    
    # Super juice transformations for these core metrics
    first_symbol['vaa_juiced'] = (((close - open) - (close.shift(1) - open.shift(1))) / 
                               (volume - volume.shift(1) + 1e-9)) * 1000 / PI**3
    first_symbol['fps_juiced'] = (np.abs((high - close) / (open - low + 1e-9) - volume / (close + 1e-9))) ** 3 * 10

    # Keep original versions too
    first_symbol['vaa'] = ((close - open) - (close.shift(1) - open.shift(1))) / (volume - volume.shift(1) + 1e-9)
    first_symbol['fps'] = np.abs((high - close) / (open - low + 1e-9) - volume / (close + 1e-9))
    
    # Moving averages - enhance with PI-normalized versions
    first_symbol['SMA'] = close.rolling(window=n).mean()
    first_symbol['SMA_juiced'] = first_symbol['SMA'] * 1000 / PI**0.5
    
    # Exponential Moving Average (EMA)
    first_symbol['EMA'] = close.ewm(span=n, adjust=False).mean()
    first_symbol['EMA_juiced'] = first_symbol['EMA'] * 1000 / PI**0.5

    # Relative Strength Index (RSI) - enhanced with cubic transformation
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    rs = avg_gain / avg_loss
    first_symbol['RSI'] = 100 - (100 / (1 + rs))
    first_symbol['RSI_cubed'] = (first_symbol['RSI'] / 100)**3 * 1000  # Cubic transformation to amplify signals

    # Bollinger Bands - Enhance spread metrics
    middle_band = close.rolling(window=n).mean()
    std_dev = close.rolling(window=n).std()
    first_symbol['BB_upper'] = middle_band + (k * std_dev)
    first_symbol['BB_middle'] = middle_band
    first_symbol['BB_lower'] = middle_band - (k * std_dev)
    
    # Add enhanced BB metrics
    first_symbol['BB_spread'] = (first_symbol['BB_upper'] - first_symbol['BB_lower']) * 1000 / PI
    first_symbol['BB_position'] = ((close - first_symbol['BB_lower']) / 
                                 (first_symbol['BB_upper'] - first_symbol['BB_lower'] + 1e-9)) ** 3 * 100

    # MACD - Enhanced with PI transformations
    ema_short = close.ewm(span=short_period, adjust=False).mean()
    ema_long = close.ewm(span=long_period, adjust=False).mean()
    first_symbol['MACD'] = ema_short - ema_long
    first_symbol['MACD_signal'] = first_symbol['MACD'].ewm(span=signal_period, adjust=False).mean()
    first_symbol['MACD_juiced'] = first_symbol['MACD'] * 1000 / PI
    first_symbol['MACD_histogram'] = (first_symbol['MACD'] - first_symbol['MACD_signal']) ** 3 * 10

    # Volume Weighted Average Price (VWAP)
    typical_price = (high + low + close) / 3
    cum_vp = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    first_symbol['VWAP'] = cum_vp / cum_vol
    first_symbol['VWAP_distance'] = ((close - first_symbol['VWAP']) / first_symbol['VWAP']) * 1000 / PI**2

    # Lag features
    for lag in [1, 2, 3]:
        first_symbol[f'Close_lag{lag}'] = close.shift(lag)
        first_symbol[f'Volume_lag{lag}'] = volume.shift(lag)
        # Add enhanced lag differentials
        first_symbol[f'Close_lag{lag}_diff'] = (close - close.shift(lag)) * 1000 / PI
        first_symbol[f'Volume_lag{lag}_ratio'] = (volume / (volume.shift(lag) + 1e-9)) ** 3

    # Momentum features - enhanced
    first_symbol['Close_momentum_3'] = close - close.shift(3)
    first_symbol['Close_momentum_3_juiced'] = first_symbol['Close_momentum_3'] * 1000 / PI**3
    first_symbol['Volume_change_5'] = volume.pct_change(5)
    first_symbol['Volume_change_5_cubed'] = first_symbol['Volume_change_5'] ** 3 * 100

    # Cumulative Return - enhanced with cubic transformations
    first_symbol['cum_return_5'] = (close / close.shift(5)) - 1
    first_symbol['cum_return_5_cubed'] = first_symbol['cum_return_5'] ** 3 * 1000
    first_symbol['is_up_trend'] = (first_symbol['cum_return_5'] > 0).astype(int)

    # Rolling Correlations - enhanced
    first_symbol['price_vol_corr'] = close.rolling(10).corr(volume)
    first_symbol['price_vol_corr_juiced'] = (first_symbol['price_vol_corr'] ** 3) * 1000 / PI

    # Return 3 - future prediction target
    first_symbol['return_3'] = (first_symbol['Close'].shift(-3) / first_symbol['Close']) - 1
    first_symbol['return_3_juiced'] = first_symbol['return_3'] * 1000 / PI**2

    # Volatility metrics - enhanced
    first_symbol['volatility_5'] = close.pct_change().rolling(5).std()
    first_symbol['volatility_5_juiced'] = first_symbol['volatility_5'] * 1000 / PI

    # Range spike metrics - enhanced
    first_symbol['range_spike'] = (high - low) / (high.shift(1) - low.shift(1) + 1e-9)
    first_symbol['range_spike_juiced'] = (first_symbol['range_spike'] ** 3) * 10
    first_symbol['range_spike_flag'] = (first_symbol['range_spike'] > 1.5).astype(int)

    # Breakout Pressure - enhanced with PI transformations
    first_symbol['near_upper_band'] = (close >= first_symbol['BB_upper']).astype(int)
    first_symbol['near_lower_band'] = (close <= first_symbol['BB_lower']).astype(int)
    first_symbol['breakout_pressure'] = (first_symbol['near_upper_band'] - first_symbol['near_lower_band'])
    first_symbol['breakout_pressure_juiced'] = first_symbol['breakout_pressure'] * PI * 10

    # Divergence between Momentum and Price - enhanced
    first_symbol['MACD_slope'] = first_symbol['MACD'].diff()
    first_symbol['MACD_slope_juiced'] = first_symbol['MACD_slope'] * 1000 / PI**2
    first_symbol['bullish_divergence'] = ((first_symbol['MACD_slope'] > 0) & 
                                        (first_symbol['Close_momentum_3'] < 0)).astype(int)
    
    # Add an intensity measure for divergence
    first_symbol['divergence_intensity'] = (abs(first_symbol['MACD_slope']) * 
                                          abs(first_symbol['Close_momentum_3'])) * 1000 / PI**3

    # Price Acceleration - enhanced
    first_symbol['price_acceleration'] = first_symbol['Close'].diff().diff()
    first_symbol['price_acceleration_juiced'] = first_symbol['price_acceleration'] * 1000 / PI**2

    # Rolling Z-Score - enhanced
    first_symbol['z_score_close'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    first_symbol['z_score_close_cubed'] = first_symbol['z_score_close'] ** 3 * 10
    first_symbol['z_score_spike'] = (first_symbol['z_score_close'].abs() > 2).astype(int)

    # Relative Volume Spike - enhanced
    first_symbol['relative_volume'] = volume / volume.rolling(20).mean()
    first_symbol['relative_volume_juiced'] = first_symbol['relative_volume'] ** 3
    first_symbol['volume_spike'] = (first_symbol['relative_volume'] > 2).astype(int)

    # Price/Volume Crossover - enhanced with PI transformations
    first_symbol['price_jump'] = close.pct_change().abs()
    first_symbol['price_jump_juiced'] = first_symbol['price_jump'] * 1000 / PI
    first_symbol['jump_with_volume'] = ((first_symbol['price_jump'] > 0.01) & 
                                      (first_symbol['relative_volume'] > 2)).astype(int)
    
    # Intensity of price-volume jumps
    first_symbol['jump_intensity'] = (first_symbol['price_jump'] * 
                                    first_symbol['relative_volume']) * 1000 / PI**2

    # Chaikin Volatility - enhanced
    hl_diff = high - low
    first_symbol['chaikin_vol'] = hl_diff.rolling(10).mean().pct_change(10)
    first_symbol['chaikin_vol_juiced'] = first_symbol['chaikin_vol'] * 1000 / PI**3

    #=========================== NQ=F and JPY=X features ===========================
    first_symbol['n_close'] = n_close
    first_symbol['n_open'] = n_open
    first_symbol['n_high'] = n_high
    first_symbol['n_low'] = n_low
    first_symbol['n_volume'] = n_volume

    first_symbol['j_close'] = j_close
    first_symbol['j_open'] = j_open
    first_symbol['j_high'] = j_high
    first_symbol['j_low'] = j_low
    first_symbol['j_volume'] = j_volume

    # Enhanced cross-asset features
    first_symbol['n_momentum_diff'] = (n_close - n_open) - (close - open)
    first_symbol['n_momentum_diff_juiced'] = first_symbol['n_momentum_diff'] * 1000 / PI**2
    
    first_symbol['j_momentum_diff'] = (j_close - j_open) - (close - open)
    first_symbol['j_momentum_diff_juiced'] = first_symbol['j_momentum_diff'] * 1000 / PI**2

    # Leader-follow lag features - enhanced
    first_symbol['n_lead_return'] = n_close.shift(1) / n_close.shift(2) - 1
    first_symbol['n_lead_return_juiced'] = first_symbol['n_lead_return'] * 1000 / PI
    
    first_symbol['j_lead_return'] = j_close.shift(1) / j_close.shift(2) - 1
    first_symbol['j_lead_return_juiced'] = first_symbol['j_lead_return'] * 1000 / PI
    
    first_symbol['mes_catchup'] = (first_symbol['n_lead_return'] + first_symbol['j_lead_return']) / 2 - ((close / close.shift(1)) - 1)
    first_symbol['mes_catchup_cubed'] = first_symbol['mes_catchup'] ** 3 * 1000

    # Spread divergence - enhanced
    first_symbol['n_spread'] = close - n_close
    first_symbol['j_spread'] = close - j_close
    first_symbol['spread_deviation'] = (first_symbol['n_spread'] - first_symbol['n_spread'].mean()) + (first_symbol['j_spread'] - first_symbol['j_spread'].mean())
    first_symbol['spread_deviation_juiced'] = first_symbol['spread_deviation'] * 1000 / (PI**3)

    # Volume-confirmed momentum - enhanced
    first_symbol['volume_momentum_ratio'] = ((close - open) / volume) / (((n_close - n_open) / n_volume) + ((j_close - j_open) / j_volume))
    first_symbol['volume_momentum_ratio_juiced'] = first_symbol['volume_momentum_ratio'] ** 3 * 10

    # Index Agreement - enhanced with intensity measure
    first_symbol['agreement'] = (
        ((close > open) & (n_close > n_open) & (j_close > j_open)).astype(int) -
        ((close < open) & (n_close < n_open) & (j_close < j_open)).astype(int)
    )
    
    # Add an agreement intensity metric
    first_symbol['agreement_intensity'] = first_symbol['agreement'] * (
        abs(close - open) + abs(n_close - n_open) + abs(j_close - j_open)
    ) * 1000 / PI**2

    # Tri-Index Momentum Entanglement - enhanced
    first_symbol['momentum_entanglement'] = abs((close - open) * (n_close - n_open) * (j_close - j_open)) / (
        abs(close - open) + abs(n_close - n_open) + abs(j_close - j_open) + 1e-9
    )
    first_symbol['momentum_entanglement_juiced'] = first_symbol['momentum_entanglement'] * 1000 / PI**3
    
    # Cross-Volume Weighted Relative Strength - enhanced
    first_symbol['cross_vol_weighted_strength'] = (
        ((close - open) * volume) / (volume + n_volume + j_volume + 1e-9)
    ) - (
        ((n_close - n_open) * n_volume + (j_close - j_open) * j_volume) / (volume + n_volume + j_volume + 1e-9)
    )
    first_symbol['cross_vol_weighted_strength_juiced'] = first_symbol['cross_vol_weighted_strength'] * 1000 / PI**2
    
    # Volatility Sync Index - enhanced
    first_symbol['volatility_sync'] = 1 - abs((high - low) - ((n_high - n_low) + (j_high - j_low)) / 2) / (
        (high - low) + ((n_high - n_low) + (j_high - j_low)) / 2 + 1e-9
    )
    first_symbol['volatility_sync_juiced'] = first_symbol['volatility_sync'] ** 3 * 1000
    
    # Directional Agreement Strength with Volume Amplifier - enhanced
    first_symbol['directional_agreement_volume'] = (
        ((close > open).astype(int) + (n_close > n_open).astype(int) + (j_close > j_open).astype(int)) / 3
    ) * (volume + n_volume + j_volume) / (volume.max() + n_volume.max() + j_volume.max() + 1e-9)
    first_symbol['directional_agreement_volume_juiced'] = first_symbol['directional_agreement_volume'] * 1000 / PI**2
    
    # Tri-Index Momentum Ratio Oscillator - enhanced
    first_symbol['momentum_ratio_oscillator'] = (
        ((close - open) * (n_close - n_open) + 1e-9) / (abs(j_close - j_open) + 1e-9)
    )
    first_symbol['momentum_ratio_oscillator_juiced'] = first_symbol['momentum_ratio_oscillator'] * 1000 / (PI**3)
    
    # New Trigonometric Transformation Features
    first_symbol['sin_price_cycle'] = np.sin(close.pct_change() * 1000) * 100
    first_symbol['cos_volume_cycle'] = np.cos(volume.pct_change() * 1000) * 100
    first_symbol['tan_momentum_cycle'] = np.tan(np.clip(first_symbol['Close_momentum_3'] * 10, -1.5, 1.5)) * 10
    
    # Logarithmic transformations of key metrics
    first_symbol['log_price_range'] = np.log1p(high - low) * 1000 / PI
    first_symbol['log_volume_spike'] = np.log1p(first_symbol['relative_volume']) * 1000 / PI
    
    # Power transformations
    first_symbol['power_RSI'] = (first_symbol['RSI'] / 50 - 1) ** 5 * 100  # Emphasizes extremes
    first_symbol['power_volume'] = (volume / volume.rolling(20).mean()) ** 4  # Super-emphasizes volume spikes
    
    # Exponential transformations
    first_symbol['exp_momentum'] = np.exp(np.clip(close.pct_change() * 20, -4, 4)) - 1  # Emphasizes directional moves
    
    # Cross-asset non-linear relationships
    first_symbol['cross_asset_entropy'] = -((close / n_close) * np.log(close / n_close + 1e-9) + 
                                          (close / j_close) * np.log(close / j_close + 1e-9)) * 1000
    
    # Quadratic trend features
    close_trend = close - close.rolling(10).mean()
    first_symbol['trend_strength_quadratic'] = close_trend ** 2 * np.sign(close_trend) * 1000 / PI
    
    # Hyperbolic features - can detect regime changes
    first_symbol['tanh_price_normalized'] = np.tanh((close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9) * 3) * 1000
    
    # Clean up infinities and NaNs
    first_symbol.replace([np.inf, -np.inf], np.nan, inplace=True)
    first_symbol.ffill(inplace=True)  # forward fill
    
    # Save the enhanced features
    first_symbol.to_csv('engineered_enhanced.csv', index=False)
    print(list(first_symbol.columns))
    
    return first_symbol