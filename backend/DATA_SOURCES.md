# Historical Options Data Sources

## Overview
To train your ML models (LightGBM for IV prediction, LSTM for price movement), you need **historical options data** with:
1. All 66 features (OHLCV, technical indicators, volatility metrics, option-specific features)
2. **Target variables**: Actual implied volatility or next-day price movement
3. Multiple samples across different stocks, strikes, expirations, and time periods

**Minimum Requirements:**
- **For LightGBM**: 1,000+ samples (more is better)
- **For LSTM**: 10,000+ samples (needs more data for sequence learning)
- **Quality over quantity**: Clean, accurate data is crucial

---

## Data Source Options

### Option 1: **CBOE DataShop** (Recommended - Most Complete)
**URL**: https://www.cboe.com/us/options/market_statistics/

**What You Get:**
- Historical options quotes (bid/ask/IV) for all strikes and expirations
- End-of-day data going back years
- Includes actual implied volatility (your target variable!)
- Covers major indices (SPX, VIX) and individual stocks

**Pricing:**
- **Free**: Limited historical samples for research
- **Paid**: ~$200-500/month for comprehensive historical data
- **Academic**: Often free for university research projects

**Data Format:**
- CSV files with columns: Date, Ticker, Strike, Expiration, Type, Bid, Ask, IV, Volume, Open Interest
- Easy to process with pandas

**How to Use:**
1. Register at https://datashop.cboe.com/
2. Download historical data for AAPL (or other tickers)
3. For each option quote:
   - Call your `/api/ml-features-mock` endpoint (modify to accept custom date/price)
   - Calculate all 66 features for that historical moment
   - Use the actual IV from CBOE as your target
4. Build training dataset with 66 features + 1 target column

---

### Option 2: **Polygon.io** (Good for Developers)
**URL**: https://polygon.io/options

**What You Get:**
- Options OHLC data (Open, High, Low, Close for options contracts)
- Historical quotes with bid/ask spreads
- Greeks calculations (delta, gamma, theta, vega, rho)
- Real-time and historical data via REST API

**Pricing:**
- **Starter**: $49/month - 5 years historical, 100 API calls/minute
- **Developer**: $99/month - 20 years historical, 500 calls/minute
- **Advanced**: $199/month - unlimited history, 1000 calls/minute
- **Free Trial**: 14 days with limited data

**Data Format:**
- JSON API responses
- Example endpoint: `GET /v2/aggs/ticker/O:AAPL251219C00270000/range/1/day/2024-01-01/2024-12-31`
- Returns: timestamp, open, high, low, close, volume, vwap

**How to Use:**
```python
import requests

API_KEY = 'your_polygon_key'
ticker = 'O:AAPL251219C00270000'  # Format: O:TICKER{YYMMDD}{C/P}{strike*1000}
url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2024-12-31?apiKey={API_KEY}'

response = requests.get(url)
data = response.json()

# Process each historical day
for bar in data['results']:
    # Get stock price for that date
    # Calculate your 66 features
    # Store with target variable
```

---

### Option 3: **ThetaData** (Specialized Options Data)
**URL**: https://thetadata.net/

**What You Get:**
- Historical options tick data (every trade/quote)
- End-of-day option prices
- Includes IV calculations
- Bulk downloads available

**Pricing:**
- **Standard**: $50/month - EOD data, 10 years history
- **Pro**: $150/month - Tick data, full history
- **Free**: 20 requests per day (good for testing)

**Data Format:**
- CSV bulk downloads
- Python API client available
- Includes: Date, Strike, Expiration, Bid, Ask, IV, Greeks, Volume

**How to Use:**
```python
from thetadata import ThetaClient

client = ThetaClient(username='your_email', passwd='your_password')
client.connect()

# Get historical options chain for AAPL on a specific date
hist_opts = client.get_hist_option(
    root='AAPL',
    exp='20251219',
    start_date='20240101',
    end_date='20241231'
)

# Process the data
for option in hist_opts:
    # Calculate features, extract IV as target
    pass
```

---

### Option 4: **IQFeed** (Professional-Grade)
**URL**: https://www.iqfeed.net/

**What You Get:**
- Comprehensive historical options data
- Real-time streaming (if needed later)
- High reliability, used by professional traders
- Includes all US options

**Pricing:**
- **Options Real-Time**: $85/month
- **Historical Data**: $70/month (one-time)
- Combined: ~$155/month

**Data Format:**
- TCP/IP streaming protocol
- Python libraries available (pyiqfeed)
- CSV export for historical data

---

### Option 5: **Interactive Brokers (IBKR) API** (If You Have Account)
**URL**: https://www.interactivebrokers.com/en/index.php?f=5041

**What You Get:**
- Free if you have IBKR account
- Historical options data via TWS API
- Can request greeks, IV, and historical prices
- Real trades, not just quotes

**Pricing:**
- **Free** with funded account ($0-10k deposit)
- Market data fees: ~$10-30/month for options quotes

**Data Format:**
- Python API (ib_insync library)
- Request historical data programmatically

**How to Use:**
```python
from ib_insync import IB, Option, util

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define option contract
contract = Option('AAPL', '20251219', 270, 'C', 'SMART')

# Request historical data
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 Y',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True
)

# Process bars
for bar in bars:
    print(bar.date, bar.close)
```

---

### Option 6: **Alpha Vantage** (Limited Options Data)
**URL**: https://www.alphavantage.co/documentation/#options

**What You Get:**
- Current options chain data
- Very limited historical data
- Not ideal for training (too sparse)

**Pricing:**
- **Free**: 25 calls/day (already using for stock data)
- **Premium**: $50/month for 1200 calls/day

**Limitation**: No historical options data - only current chains

---

### Option 7: **Build Your Own Dataset** (Free but Time-Consuming)
**Method**: Collect data over time using your current API setup

**Steps:**
1. Every day, call your `/api/ml-features-mock` endpoint for multiple options:
   - AAPL $260 Call, 30 DTE
   - AAPL $270 Call, 30 DTE
   - AAPL $280 Call, 30 DTE
   - Different expirations (7, 30, 60, 90 DTE)
2. Store the 66 features + current IV (from yfinance or manual calculation)
3. Next day, check what the actual price movement was (target variable)
4. After 30-60 days, you'll have enough data to start testing models

**Storage Structure:**
```csv
date,ticker,strike,expiration,type,feature_1,...,feature_66,target_iv,next_day_return
2024-01-01,AAPL,270,2024-02-01,call,268.47,...,0.21,0.324,0.012
2024-01-01,AAPL,280,2024-02-01,call,268.47,...,0.18,0.287,-0.005
...
```

**Advantages:**
- Free
- You control data quality
- Real-time learning as you collect

**Disadvantages:**
- Takes weeks/months to gather enough data
- Can't backtest historical strategies

---

## Recommended Approach

### For Immediate Development (Next 24-48 hours):
**Use Synthetic Data**
- Generate realistic mock options data with known patterns
- Train models on synthetic data to validate pipeline
- Test normalization, model architecture, prediction flow
- Example script below:

```python
# Generate 10,000 synthetic options samples
import numpy as np
import pandas as pd

n_samples = 10000
synthetic_data = []

for i in range(n_samples):
    base_price = np.random.uniform(200, 300)
    strike = np.random.choice([base_price * 0.95, base_price, base_price * 1.05])
    dte = np.random.randint(7, 90)
    
    # Calculate features (using your existing functions)
    features = {
        'close': base_price,
        'delta': calculate_delta(base_price, strike, dte),
        # ... all 66 features
    }
    
    # Synthetic target (IV correlates with ATM-ness and DTE)
    moneyness = strike / base_price
    target_iv = 0.25 + (abs(1 - moneyness) * 0.5) + (0.1 * (90 - dte) / 90)
    
    features['target_iv'] = target_iv
    synthetic_data.append(features)

# Save for training
df = pd.DataFrame(synthetic_data)
df.to_csv('backend/data/synthetic_training.csv', index=False)
```

### For Production (Within 1-2 weeks):
**Recommended: Polygon.io Developer Plan ($99/month)**
- Good balance of cost/quality
- 20 years of historical data
- API-first (easy integration)
- Can cancel after collecting data you need

**Alternative: ThetaData Standard ($50/month)**
- More affordable
- Sufficient for training models
- Good for academic projects

---

## Data Collection Script Template

Here's a script to collect historical data from Polygon.io:

```python
# backend/collect_historical_data.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from feature_engineering import prepare_feature_matrix
import sys
sys.path.append('.')
from app import calculate_technical_indicators, calculate_historical_volatility, calculate_option_features

POLYGON_API_KEY = 'your_key_here'

def collect_option_data(ticker, start_date, end_date):
    """
    Collect historical options data and calculate ML features
    """
    training_data = []
    
    # Get historical stock prices first
    stock_url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}'
    stock_data = requests.get(stock_url).json()
    
    # Convert to DataFrame
    stock_df = pd.DataFrame(stock_data['results'])
    stock_df['date'] = pd.to_datetime(stock_df['t'], unit='ms')
    
    # For each date, get options chain
    for idx, row in stock_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        current_price = row['c']  # Close price
        
        # Get options for common expirations
        for dte in [30, 60, 90]:
            exp_date = (row['date'] + timedelta(days=dte)).strftime('%Y%m%d')
            
            # Get options at different strikes
            for strike_offset in [-0.05, 0, 0.05]:
                strike = round(current_price * (1 + strike_offset))
                
                for option_type in ['C', 'P']:
                    # Polygon format: O:AAPL251219C00270000
                    option_ticker = f"O:{ticker}{exp_date}{option_type}{str(int(strike*1000)).zfill(8)}"
                    
                    # Get option historical data
                    opt_url = f'https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{date_str}/{date_str}?apiKey={POLYGON_API_KEY}'
                    opt_response = requests.get(opt_url)
                    
                    if opt_response.status_code == 200 and 'results' in opt_response.json():
                        opt_data = opt_response.json()['results'][0]
                        
                        # Calculate all 66 features for this option at this date
                        # (Use historical stock data up to this date)
                        historical_prices = stock_df[stock_df['date'] <= row['date']].tail(504)
                        
                        tech_indicators = calculate_technical_indicators(historical_prices)
                        hv_metrics = calculate_historical_volatility(historical_prices)
                        opt_features = calculate_option_features(...)
                        
                        # Combine features + target
                        sample = {
                            **opt_features,
                            'target_iv': opt_data.get('iv', None),  # If Polygon provides IV
                            'actual_option_price': opt_data['c']
                        }
                        
                        training_data.append(sample)
                        
                        time.sleep(0.1)  # Rate limiting
    
    return pd.DataFrame(training_data)

# Usage
df = collect_option_data('AAPL', '2024-01-01', '2024-12-31')
df.to_csv('backend/data/aapl_historical_options.csv', index=False)
print(f"Collected {len(df)} samples")
```

---

## Summary

**Best Free Option**: Build your own dataset over time (but slow)

**Best Paid Option (Immediate)**: Polygon.io Developer ($99/month) or ThetaData Standard ($50/month)

**Best Academic Option**: CBOE DataShop (often free for students)

**For Your Hackathon Timeline**: Use synthetic data for demo, upgrade to real data post-event

Let me know which option you want to pursue, and I can help you build the data collection pipeline!
