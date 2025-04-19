# -*- coding: utf-8 -*-
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that DataFrame has required columns and is not empty."""
    if df is None or len(df) == 0:
        return False
    return all(col in df.columns for col in required_columns)

def safe_division(a, b, default=0):
    """Safely divide two numbers, handling division by zero."""
    try:
        if isinstance(b, (pd.Series, np.ndarray)):
            return np.where(b != 0, a / b, default)
        return a / b if b != 0 else default
    except Exception:
        return default

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> tuple:
    """Calculate Bollinger Bands separately to avoid DataFrame assignment issues."""
    try:
        if not validate_dataframe(df, ['Close']):
            raise ValueError("Invalid DataFrame for Bollinger Bands calculation")
        
        middle_band = df['Close'].rolling(window=window).mean()
        std_dev = df['Close'].rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return middle_band, upper_band, lower_band
    except Exception as e:
        logger.error(f"Error in Bollinger Bands calculation: {str(e)}")
        raise

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various technical indicators."""
    try:
        if not validate_dataframe(df, ['Close', 'Volume']):
            raise ValueError("DataFrame missing required columns")

        # Create a copy of the dataframe
        df = df.copy()
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = safe_division(avg_gain, avg_loss, 1)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df)
        df['BB_middle'] = bb_middle
        df['BB_upper'] = bb_upper
        df['BB_lower'] = bb_lower
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = safe_division(df['Volume'], df['Volume_MA'])
        
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise

def get_stock_data(symbol: str) -> pd.DataFrame:
    """Fetch stock data and calculate technical indicators."""
    try:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Invalid stock symbol")

        logger.info(f"Fetching data for symbol: {symbol}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download stock data
        df = yf.download(symbol.upper(), start=start_date, end=end_date)
        if len(df) == 0:
            raise ValueError(f"No data found for symbol {symbol}")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not validate_dataframe(df, required_columns):
            raise ValueError(f"Missing required columns in data: {required_columns}")
            
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Add technical indicators
        df = calculate_technical_indicators(df)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Successfully processed data for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise

def generate_market_analysis(df: pd.DataFrame) -> dict:
    """Generate comprehensive market analysis."""
    try:
        if not validate_dataframe(df, ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_upper', 'BB_lower', 'Volume_Ratio']):
            raise ValueError("Missing required technical indicators")

        latest = df.iloc[-1]
        
        # Ensure all Series are properly aligned for comparison
        close = latest['Close']
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        
        # Trend Analysis with proper numeric comparison
        trend = "Upward" if (float(close) > float(sma_50) and float(sma_20) > float(sma_50)) else \
                "Downward" if (float(close) < float(sma_50) and float(sma_20) < float(sma_50)) else "Sideways"
        
        # RSI Analysis with numeric comparison
        rsi = float(latest['RSI'])
        rsi_signal = "Overbought" if rsi > 70 else \
                     "Oversold" if rsi < 30 else "Neutral"
        
        # MACD Analysis with numeric comparison
        macd = float(latest['MACD'])
        signal_line = float(latest['Signal_Line'])
        macd_signal = "Bullish" if macd > signal_line else "Bearish"
        
        # Bollinger Bands Analysis with numeric comparison
        bb_upper = float(latest['BB_upper'])
        bb_lower = float(latest['BB_lower'])
        current_close = float(latest['Close'])
        
        bb_range = bb_upper - bb_lower
        bb_position = 50 if bb_range == 0 else \
                     ((current_close - bb_lower) / bb_range) * 100
        
        bb_signal = "Overbought" if bb_position > 80 else \
                    "Oversold" if bb_position < 20 else "Neutral"
        
        # Volume Analysis with numeric comparison
        volume_ratio = float(latest['Volume_Ratio'])
        volume_signal = "High" if volume_ratio > 1.5 else \
                       "Low" if volume_ratio < 0.5 else "Normal"
        
        return {
            'trend': trend,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'bb_signal': bb_signal,
            'volume_signal': volume_signal
        }
    except Exception as e:
        logger.error(f"Error generating market analysis: {str(e)}")
        raise

def create_advanced_visualization(stock_data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create advanced visualization with multiple technical indicators."""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(
            f'{symbol} Price and Predictions',
            'Volume Analysis',
            'RSI',
            'MACD'
        )
    )

    dates = pd.to_datetime(stock_data['Date'])

    # Main price chart with candlesticks and predictions
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Add predicted prices
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['Predicted'],
            name='Predicted',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ),
        row=1, col=1
    )

    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['BB_upper'],
            name='BB Upper',
            line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['BB_lower'],
            name='BB Lower',
            line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 204, 255, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )

    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['SMA_20'],
            name='20-day MA',
            line=dict(color='#3498db', width=1.5)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['SMA_50'],
            name='50-day MA',
            line=dict(color='#9b59b6', width=1.5)
        ),
        row=1, col=1
    )

    # Volume chart with color based on price change
    colors = ['#26a69a' if close >= open_ else '#ef5350' 
             for close, open_ in zip(stock_data['Close'], stock_data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=stock_data['Volume'],
            name='Volume',
            marker=dict(color=colors),
            showlegend=False
        ),
        row=2, col=1
    )

    # Add volume MA
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['Volume_MA'],
            name='Volume MA',
            line=dict(color='#ffd700', width=2),
            showlegend=False
        ),
        row=2, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['RSI'],
            name='RSI',
            line=dict(color='#2ecc71', width=2),
            showlegend=False
        ),
        row=3, col=1
    )

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['MACD'],
            name='MACD',
            line=dict(color='#3498db', width=2),
            showlegend=False
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stock_data['Signal_Line'],
            name='Signal Line',
            line=dict(color='#e74c3c', width=2),
            showlegend=False
        ),
        row=4, col=1
    )

    # Calculate MACD histogram
    macd_hist = stock_data['MACD'] - stock_data['Signal_Line']
    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in macd_hist]
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=macd_hist,
            name='MACD Histogram',
            marker=dict(color=colors),
            showlegend=False
        ),
        row=4, col=1
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f'{symbol} Advanced Technical Analysis',
        template='plotly_dark',
        hovermode='x unified',
        height=1000,  # Increased height for better visibility
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        margin=dict(t=50, b=20, l=50, r=50)
    )

    # Update axes
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    # Add range buttons
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )

    return fig

def predict_stock_price(symbol: str) -> dict:
    """Predict stock prices with advanced technical analysis."""
    try:
        logger.info(f"Starting prediction for symbol: {symbol}")
        
        # Get stock data with indicators
        stock_data = get_stock_data(symbol)
        
        # Prepare features
        stock_data['Target'] = stock_data['Close'].shift(-1)
        stock_data = stock_data.dropna()

        # Enhanced feature set
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line',
                   'Volume_Ratio', 'BB_middle']
        
        if not validate_dataframe(stock_data, features + ['Target']):
            raise ValueError("Missing required features for prediction")

        X = stock_data[features].copy()  # Create explicit copy
        y = stock_data['Target'].copy()  # Create explicit copy

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)  # Preserve index

        # Split data while preserving index
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        stock_data.loc[:, 'Predicted'] = model.predict(X_scaled)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        # Get latest values using numeric comparison
        last_price = float(stock_data['Close'].iloc[-1])
        predicted_price = float(stock_data['Predicted'].iloc[-1])
        price_change = ((predicted_price - last_price) / last_price) * 100

        # Get market analysis
        analysis = generate_market_analysis(stock_data)
        
        # Generate signal using numeric comparisons
        signal = "BUY" if (price_change > 1 and analysis['trend'] == "Upward" and 
                          analysis['rsi_signal'] != "Overbought" and 
                          analysis['macd_signal'] == "Bullish") else \
                "SELL" if (price_change < -1 and analysis['trend'] == "Downward" and 
                          analysis['rsi_signal'] != "Oversold" and 
                          analysis['macd_signal'] == "Bearish") else "HOLD"

        # Create enhanced visualization
        fig = create_advanced_visualization(stock_data, symbol)

        logger.info(f"Successfully completed prediction for {symbol}")
        
        return {
            'plot': fig,
            'last_price': f"${last_price:.2f}",
            'predicted_price': f"${predicted_price:.2f}",
            'price_change': f"{price_change:.2f}%",
            'signal': signal,
            'accuracy': f"{accuracy*100:.2f}%",
            'analysis': {
                'Trend': analysis['trend'],
                'RSI': analysis['rsi_signal'],
                'MACD': analysis['macd_signal'],
                'Bollinger': analysis['bb_signal'],
                'Volume': analysis['volume_signal']
            }
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise 