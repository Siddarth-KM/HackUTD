"""
Historical Options Data Collection Script
Collects training data from Polygon.io for ML model training

This script:
1. Fetches historical stock prices
2. For each historical date, simulates multiple option contracts
3. Calculates all 66 features for each option
4. Saves to CSV for model training

Run this ONCE after getting Polygon API key to collect ~10,000 training samples
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon_client import PolygonClient
from feature_engineering import prepare_feature_matrix
import time

# Import your existing feature calculation functions
from app import (
    calculate_technical_indicators,
    calculate_historical_volatility,
    calculate_option_features
)


class TrainingDataCollector:
    """Collect historical options training data from Polygon"""
    
    def __init__(self, polygon_api_key=None):
        """
        Initialize data collector
        
        Args:
            polygon_api_key: Polygon API key (optional, reads from .env if not provided)
        """
        self.client = PolygonClient(polygon_api_key)
        self.training_data = []
    
    def collect_for_ticker(self, ticker, start_date, end_date, strikes_config=None):
        """
        Collect training data for one ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            strikes_config: Dict with strike configuration
                {
                    'offsets': [-0.10, -0.05, 0, 0.05, 0.10],  # % from current price
                    'dte_values': [7, 14, 30, 45, 60, 90],     # Days to expiration
                    'option_types': ['call', 'put']
                }
        
        Returns:
            int: Number of samples collected
        """
        if strikes_config is None:
            strikes_config = {
                'offsets': [-0.10, -0.05, 0, 0.05, 0.10],  # 10% OTM to 10% ITM
                'dte_values': [7, 14, 30, 45, 60, 90],
                'option_types': ['call', 'put']
            }
        
        print(f"\n{'='*60}")
        print(f"Collecting training data for {ticker}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"{'='*60}\n")
        
        # Step 1: Get full historical stock data
        print(f"Fetching stock data for {ticker}...")
        stock_df = self.client.get_stock_data(ticker, start_date, end_date)
        
        if stock_df.empty:
            print(f"No stock data found for {ticker}")
            return 0
        
        print(f"✓ Retrieved {len(stock_df)} days of stock data")
        
        # Step 2: Get ticker details for sector
        details = self.client.get_ticker_details(ticker)
        sector = details.get('sic_description', 'Technology')
        
        # Step 3: For each date (sampling every N days to avoid too much data)
        sample_every_n_days = 5  # Sample every 5 days to keep dataset manageable
        dates_to_process = stock_df.index[::sample_every_n_days]
        
        print(f"Processing {len(dates_to_process)} dates (sampling every {sample_every_n_days} days)...")
        
        samples_collected = 0
        
        for idx, date in enumerate(dates_to_process):
            try:
                # Get historical data up to this date (for technical indicators)
                historical_up_to_date = stock_df.loc[:date].tail(504)  # Last 2 years
                
                if len(historical_up_to_date) < 100:
                    continue  # Need enough history for indicators
                
                current_price = float(historical_up_to_date['Close'].iloc[-1])
                
                # Calculate technical indicators and HV for this date
                tech_indicators = calculate_technical_indicators(historical_up_to_date)
                hv_metrics = calculate_historical_volatility(historical_up_to_date)
                
                if tech_indicators.empty or hv_metrics is None:
                    continue
                
                # For each strike configuration
                for offset in strikes_config['offsets']:
                    strike = round(current_price * (1 + offset), 2)
                    
                    for dte in strikes_config['dte_values']:
                        expiration_date = date + timedelta(days=dte)
                        
                        for option_type in strikes_config['option_types']:
                            # Calculate option features
                            try:
                                # Mock earnings data (in production, you'd fetch real earnings dates)
                                earnings_data = {
                                    'next_earnings_date': (date + timedelta(days=90)).strftime('%Y-%m-%d'),
                                    'days_until_earnings': 90
                                }
                                
                                # Calculate all option features
                                option_features = calculate_option_features(
                                    ticker=ticker,
                                    strike_price=strike,
                                    expiration_date=expiration_date.strftime('%Y-%m-%d'),
                                    option_type=option_type,
                                    current_price=current_price,
                                    sector=sector,
                                    hv_metrics=hv_metrics,
                                    earnings_data=earnings_data,
                                    risk_free_rate=0.04,  # Mock (in production, use actual rates)
                                    vix=19.0  # Mock (in production, fetch real VIX)
                                )
                                
                                # Combine all features
                                latest_tech = tech_indicators.iloc[-1]
                                
                                sample = {
                                    # Metadata
                                    'date': date.strftime('%Y-%m-%d'),
                                    'ticker': ticker,
                                    'strike': strike,
                                    'expiration': expiration_date.strftime('%Y-%m-%d'),
                                    'option_type': option_type,
                                    
                                    # OHLCV
                                    'open': float(historical_up_to_date['Open'].iloc[-1]),
                                    'high': float(historical_up_to_date['High'].iloc[-1]),
                                    'low': float(historical_up_to_date['Low'].iloc[-1]),
                                    'close': current_price,
                                    'volume': int(historical_up_to_date['Volume'].iloc[-1]),
                                    
                                    # Technical indicators (29 features)
                                    **{col.lower(): float(latest_tech[col]) for col in tech_indicators.columns},
                                    
                                    # Historical volatility (9 features)
                                    'hv_30': hv_metrics['HV_30'],
                                    'hv_60': hv_metrics['HV_60'],
                                    'hv_90': hv_metrics['HV_90'],
                                    'daily_volatility': hv_metrics['daily_volatility'],
                                    'hv_slope': hv_metrics['HV_slope'],
                                    'hv_regime': 1 if hv_metrics['HV_regime'] == 'expanding' else 0,
                                    'hv_acceleration': hv_metrics['HV_acceleration'],
                                    'hv_percentile': hv_metrics['HV_percentile'],
                                    'hv_z_score': hv_metrics['HV_z_score'],
                                    
                                    # Option features (27 features)
                                    **option_features,
                                    
                                    # TARGET VARIABLES (to be filled with actual data later)
                                    # In production, you'd fetch the actual option price from Polygon
                                    # and calculate actual IV, or use next-day stock movement
                                    'target_next_day_return': 0.0,  # Placeholder
                                    'target_actual_iv': 0.0  # Placeholder
                                }
                                
                                self.training_data.append(sample)
                                samples_collected += 1
                                
                            except Exception as e:
                                print(f"Error calculating features: {e}")
                                continue
                
                # Progress update
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(dates_to_process)} dates, {samples_collected} samples collected")
                
            except Exception as e:
                print(f"Error processing date {date}: {e}")
                continue
        
        print(f"\n✓ Collected {samples_collected} samples for {ticker}")
        return samples_collected
    
    def save_to_csv(self, filename='training_data.csv'):
        """
        Save collected data to CSV
        
        Args:
            filename: Output filename
        """
        if not self.training_data:
            print("No data to save!")
            return
        
        df = pd.DataFrame(self.training_data)
        
        # Save to backend/data/ directory
        output_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Saved {len(df)} samples to {output_path}")
        print(f"  Features: {len(df.columns)} columns")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"{'='*60}")
        
        return output_path
    
    def collect_multi_ticker(self, tickers, start_date, end_date):
        """
        Collect data for multiple tickers
        
        Args:
            tickers: List of ticker symbols ['AAPL', 'MSFT', 'GOOGL', ...]
            start_date: Start date
            end_date: End date
        
        Returns:
            str: Path to saved CSV file
        """
        total_samples = 0
        total_tickers = len(tickers)
        
        for ticker_idx, ticker in enumerate(tickers, 1):
            try:
                print(f"\n{'#'*60}")
                print(f"TICKER {ticker_idx}/{total_tickers}: {ticker}")
                print(f"{'#'*60}")
                
                samples = self.collect_for_ticker(ticker, start_date, end_date)
                total_samples += samples
                print(f"\n✓ {ticker}: Collected {samples} samples")
                print(f"✓ Total samples so far: {total_samples}")
                print(f"✓ Progress: {ticker_idx}/{total_tickers} tickers ({ticker_idx/total_tickers*100:.1f}%)")
                
                # Save intermediate results after each ticker (in case of crash)
                if total_samples > 0:
                    temp_file = self.save_to_csv()
                    print(f"✓ Intermediate save: {temp_file}")
                
                # Rate limit pause between tickers (Polygon free tier: 5 calls/min)
                if ticker_idx < total_tickers:
                    print(f"\nWaiting 15 seconds before next ticker (rate limit protection)...")
                    time.sleep(15)
                
            except Exception as e:
                print(f"\n❌ Error collecting data for {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return self.save_to_csv()


def main():
    """
    Main collection script
    Run this to collect training data
    """
    print("\n" + "="*60)
    print("Options Training Data Collection")
    print("="*60)
    
    # Initialize collector
    collector = TrainingDataCollector()
    
    # Configuration - MAG 7 + Major ETFs
    tickers = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Alphabet
        'AMZN',   # Amazon
        'NVDA',   # Nvidia
        'TSLA',   # Tesla
        'META',   # Meta
        'QQQ',    # Nasdaq-100 ETF
        'SPY'     # S&P 500 ETF
    ]
    
    # Date range: Last 1 year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {', '.join(tickers)} ({len(tickers)} total)")
    print(f"  Date range: {start_str} to {end_str}")
    print(f"  Strikes: 5 offsets (-10%, -5%, 0%, +5%, +10%)")
    print(f"  DTEs: 6 values (7, 14, 30, 45, 60, 90 days)")
    print(f"  Types: Call & Put")
    print(f"  Sampling: Every 5 days")
    
    # Calculate expected samples
    days_in_range = (end_date - start_date).days
    sample_dates = days_in_range // 5
    samples_per_ticker = sample_dates * 5 * 6 * 2  # strikes × DTEs × types
    total_expected = len(tickers) * samples_per_ticker
    
    print(f"  Expected samples: ~{total_expected:,} ({len(tickers)} tickers × ~{samples_per_ticker:,} each)")
    print(f"  Estimated time: ~{len(tickers) * 5} minutes (5 min/ticker with rate limits)")
    print()
    
    # Collect data
    output_file = collector.collect_multi_ticker(tickers, start_str, end_str)
    
    print("\n✓ Data collection complete!")
    print(f"\nNext steps:")
    print(f"1. Load the data: df = pd.read_csv('{output_file}')")
    print(f"2. Fill target variables (actual IV or next-day returns)")
    print(f"3. Train models using feature_engineering.py functions")
    print(f"4. Save trained models for prediction API")


if __name__ == "__main__":
    main()
