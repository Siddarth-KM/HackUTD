"""
Polygon.io API Client
Handles all data fetching from Polygon.io for stocks and options
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

class PolygonClient:
    """Client for Polygon.io REST API"""
    
    def __init__(self, api_key=None):
        """
        Initialize Polygon client
        
        Args:
            api_key: Polygon API key (if None, reads from .env POLYGON_API_KEY)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key not found. Set POLYGON_API_KEY in .env file")
        
        self.base_url = 'https://api.polygon.io'
        self.session = requests.Session()
        self.rate_limit_delay = 0.12  # 500 calls/min = ~0.12s between calls
    
    def _make_request(self, endpoint, params=None):
        """
        Make API request with error handling and rate limiting
        
        Args:
            endpoint: API endpoint (e.g., '/v2/aggs/ticker/AAPL/range/1/day/...')
            params: Query parameters dict
        
        Returns:
            dict: JSON response
        """
        if params is None:
            params = {}
        
        params['apiKey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            data = response.json()
            
            # Check for API errors
            if data.get('status') == 'ERROR':
                raise Exception(f"Polygon API Error: {data.get('error', 'Unknown error')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise
    
    def get_stock_data(self, ticker, start_date, end_date, timespan='day'):
        """
        Get historical stock OHLCV data
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            timespan: 'minute', 'hour', 'day', 'week', 'month'
        
        Returns:
            pd.DataFrame with columns: Date (index), Open, High, Low, Close, Volume
        
        Example:
            client = PolygonClient()
            df = client.get_stock_data('AAPL', '2023-01-01', '2025-01-01')
        """
        endpoint = f'/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}'
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            data = self._make_request(endpoint, params)
            
            if 'results' not in data or not data['results']:
                print(f"No data returned for {ticker} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Rename columns to match your existing code
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'vw': 'VWAP',
                'n': 'num_trades'
            })
            
            # Convert timestamp to datetime
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('Date')
            
            # Select only needed columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            print(f"Fetched {len(df)} bars for {ticker}")
            return df
            
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_option_data(self, ticker, strike, expiration_date, option_type, start_date, end_date):
        """
        Get historical options OHLCV data
        
        Args:
            ticker: Underlying stock symbol (e.g., 'AAPL')
            strike: Strike price (e.g., 270.0)
            expiration_date: Expiration as datetime or 'YYYY-MM-DD' or 'YYMMDD'
            option_type: 'call' or 'put' (or 'C' or 'P')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            pd.DataFrame with option OHLCV data
        
        Example:
            df = client.get_option_data('AAPL', 270, '2025-12-19', 'call', '2024-01-01', '2024-12-31')
        """
        # Format option ticker: O:AAPL251219C00270000
        # Format: O:{underlying}{YYMMDD}{C/P}{strike*1000 padded to 8 digits}
        
        # Parse expiration date
        if isinstance(expiration_date, str):
            if len(expiration_date) == 10:  # YYYY-MM-DD
                exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
            elif len(expiration_date) == 6:  # YYMMDD
                exp_dt = datetime.strptime(expiration_date, '%y%m%d')
            else:
                raise ValueError(f"Invalid expiration date format: {expiration_date}")
        else:
            exp_dt = expiration_date
        
        exp_str = exp_dt.strftime('%y%m%d')
        
        # Parse option type
        opt_type = option_type[0].upper()  # 'C' or 'P'
        
        # Format strike (multiply by 1000 and pad to 8 digits)
        strike_str = str(int(strike * 1000)).zfill(8)
        
        # Build option ticker
        option_ticker = f"O:{ticker}{exp_str}{opt_type}{strike_str}"
        
        endpoint = f'/v2/aggs/ticker/{option_ticker}/range/1/day/{start_date}/{end_date}'
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            data = self._make_request(endpoint, params)
            
            if 'results' not in data or not data['results']:
                print(f"No option data for {option_ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'vw': 'VWAP'
            })
            
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('Date')
            
            print(f"Fetched {len(df)} option bars for {option_ticker}")
            return df
            
        except Exception as e:
            print(f"Error fetching option data for {option_ticker}: {e}")
            return pd.DataFrame()
    
    def get_ticker_details(self, ticker):
        """
        Get company details (sector, market cap, etc.)
        
        Args:
            ticker: Stock symbol
        
        Returns:
            dict with company info
        
        Example:
            details = client.get_ticker_details('AAPL')
            sector = details.get('sector', 'Technology')
        """
        endpoint = f'/v3/reference/tickers/{ticker}'
        
        try:
            data = self._make_request(endpoint)
            
            if 'results' not in data:
                print(f"No details found for {ticker}")
                return {}
            
            results = data['results']
            
            return {
                'ticker': results.get('ticker'),
                'name': results.get('name'),
                'market': results.get('market'),
                'locale': results.get('locale'),
                'primary_exchange': results.get('primary_exchange'),
                'type': results.get('type'),
                'active': results.get('active'),
                'currency_name': results.get('currency_name'),
                'cik': results.get('cik'),
                'composite_figi': results.get('composite_figi'),
                'share_class_figi': results.get('share_class_figi'),
                'market_cap': results.get('market_cap'),
                'phone_number': results.get('phone_number'),
                'address': results.get('address'),
                'description': results.get('description'),
                'sic_code': results.get('sic_code'),
                'sic_description': results.get('sic_description'),
                'ticker_root': results.get('ticker_root'),
                'homepage_url': results.get('homepage_url'),
                'total_employees': results.get('total_employees'),
                'list_date': results.get('list_date'),
                'branding': results.get('branding'),
                'share_class_shares_outstanding': results.get('share_class_shares_outstanding'),
                'weighted_shares_outstanding': results.get('weighted_shares_outstanding'),
                'round_lot': results.get('round_lot'),
            }
            
        except Exception as e:
            print(f"Error fetching ticker details for {ticker}: {e}")
            return {}
    
    def get_previous_close(self, ticker):
        """
        Get previous day's close price
        
        Args:
            ticker: Stock symbol
        
        Returns:
            dict with previous close data
        """
        endpoint = f'/v2/aggs/ticker/{ticker}/prev'
        
        try:
            data = self._make_request(endpoint)
            
            if 'results' not in data or not data['results']:
                return {}
            
            result = data['results'][0]
            
            return {
                'ticker': result.get('T'),
                'open': result.get('o'),
                'high': result.get('h'),
                'low': result.get('l'),
                'close': result.get('c'),
                'volume': result.get('v'),
                'vwap': result.get('vw'),
                'timestamp': result.get('t'),
                'num_trades': result.get('n')
            }
            
        except Exception as e:
            print(f"Error fetching previous close for {ticker}: {e}")
            return {}


# Utility functions for easy use
def fetch_stock_data_polygon(ticker, period='2y'):
    """
    Fetch stock data using Polygon (replaces yfinance)
    
    Args:
        ticker: Stock symbol
        period: Time period ('1y', '2y', '5y', etc.)
    
    Returns:
        pd.DataFrame with OHLCV data
    
    Example:
        df = fetch_stock_data_polygon('AAPL', period='2y')
    """
    client = PolygonClient()
    
    # Calculate date range
    end_date = datetime.now()
    
    if period.endswith('y'):
        years = int(period[:-1])
        start_date = end_date - timedelta(days=years * 365)
    elif period.endswith('mo'):
        months = int(period[:-2])
        start_date = end_date - timedelta(days=months * 30)
    elif period.endswith('d'):
        days = int(period[:-1])
        start_date = end_date - timedelta(days=days)
    else:
        # Default to 2 years
        start_date = end_date - timedelta(days=730)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    return client.get_stock_data(ticker, start_str, end_str)


def fetch_option_data_polygon(ticker, strike, expiration, option_type, period='1y'):
    """
    Fetch historical option data
    
    Args:
        ticker: Underlying stock
        strike: Strike price
        expiration: Expiration date
        option_type: 'call' or 'put'
        period: How far back to get data
    
    Returns:
        pd.DataFrame with option OHLCV
    """
    client = PolygonClient()
    
    end_date = datetime.now()
    
    if period.endswith('y'):
        years = int(period[:-1])
        start_date = end_date - timedelta(days=years * 365)
    else:
        start_date = end_date - timedelta(days=365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    return client.get_option_data(ticker, strike, expiration, option_type, start_str, end_str)


# Test the client
if __name__ == "__main__":
    print("="*60)
    print("Testing Polygon Client")
    print("="*60)
    
    try:
        client = PolygonClient()
        
        # Test 1: Get stock data
        print("\n1. Fetching AAPL stock data (last 30 days)...")
        end = datetime.now()
        start = end - timedelta(days=30)
        df = client.get_stock_data('AAPL', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        print(f"   Retrieved {len(df)} days of data")
        print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")
        
        # Test 2: Get ticker details
        print("\n2. Fetching AAPL company details...")
        details = client.get_ticker_details('AAPL')
        print(f"   Name: {details.get('name')}")
        print(f"   Market Cap: ${details.get('market_cap', 0):,.0f}")
        
        # Test 3: Get previous close
        print("\n3. Fetching previous close...")
        prev = client.get_previous_close('AAPL')
        print(f"   Previous close: ${prev.get('close', 0):.2f}")
        
        print("\n" + "="*60)
        print("Polygon Client Test Complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to set POLYGON_API_KEY in your .env file")
