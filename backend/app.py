''' 
Options Pricing and Stock Prediction Backend Application, 
using machine learning models such as random forest, lightgbm, 
and LSTM neural networks. Features a React frontend and Flask backend.
Backend handles data fetching, preprocessing, model training,
and prediction serving, written in python.
'''
import os
import time
import pickle
from polygon_client import PolygonClient
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import joblib
from dotenv import load_dotenv
from fredapi import Fred
from scipy.stats import norm
from flask import Flask, jsonify, request
from flask_cors import CORS
import ta
from tensorflow import keras
from feature_engineering import (
    prepare_feature_matrix,
    create_feature_scaler,
    load_feature_scaler,
    normalize_features
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Cache Manager - stores fetched data with expiration to limit API calls

# Persistent cache file location
CACHE_FILE = 'backend/cache_data.pkl'

# Global cache dictionary - load from disk if exists
def load_cache_from_disk():
    """Load cache from disk file if it exists"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                loaded_cache = pickle.load(f)
                print(f"Loaded {len(loaded_cache)} items from cache file")
                return loaded_cache
        except Exception as e:
            print(f"Error loading cache from disk: {e}")
            return {}
    return {}

def save_cache_to_disk():
    """Save cache to disk file"""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache to disk: {e}")

# Load existing cache on startup
cache = load_cache_from_disk()

# Time-to-live constants (in seconds)
TTL_STOCK_PRICE = 3600     # 1 hour - for development/testing (normally 5 min)
TTL_SENTIMENT = 3600       # 1 hour - news doesn't change that fast
TTL_EARNINGS = 86400       # 24 hours - earnings dates rarely change
TTL_OPTIONS = 900          # 15 minutes - options data moderately volatile
TTL_VIX = 3600             # 1 hour - for development (normally 5 min)
TTL_RISK_FREE = 3600       # 1 hour - treasury rates

def is_cached(key):
    """
    Check if data exists in cache and is still fresh
    
    Args:
        key: Cache key string
    
    Returns:
        bool: True if cached and fresh, False otherwise
    """
    if key not in cache:
        return False
    
    # Check if expired
    cached_item = cache[key]
    current_time = datetime.now().timestamp()
    age = current_time - cached_item['timestamp']
    
    if age > cached_item['ttl']:
        # Expired - remove from cache
        del cache[key]
        return False
    
    return True

def get_from_cache(key):
    """
    Retrieve data from cache
    
    Args:
        key: Cache key string
    
    Returns:
        Cached data or None if not found/expired
    """
    if is_cached(key):
        return cache[key]['data']
    return None

def add_to_cache(key, data, ttl):
    """
    Add data to cache with expiration time and persist to disk
    
    Args:
        key: Cache key string
        data: Data to cache
        ttl: Time to live in seconds
    """
    cache[key] = {
        'data': data,
        'timestamp': datetime.now().timestamp(),
        'ttl': ttl
    }
    
    # Save to disk for persistence
    save_cache_to_disk()
    print(f"Cached {key} (TTL: {ttl}s) - Total cache items: {len(cache)}")


# Custom exception for API rate limits
class RateLimitError(Exception):
    """Custom exception for API rate limits"""
    pass


# BLACK-SCHOLES OPTION PRICING

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price
    
    Black-Scholes assumes:
    - Stock follows Geometric Brownian Motion (GBM): dS = μS dt + σS dW
    - No dividends
    - European-style exercise
    - Constant volatility and risk-free rate
    - No transaction costs
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        sigma: Implied volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Theoretical option price
    """
    if T <= 0:
        # At expiration, option worth intrinsic value only
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    if sigma <= 0:
        # Zero volatility - option either worthless or worth intrinsic
        if option_type == 'call':
            return max(0, S - K * np.exp(-r * T))
        else:
            return max(0, K * np.exp(-r * T) - S)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all option Greeks
    
    Greeks measure sensitivity of option price to various factors:
    - Delta (Δ): Change in option price per $1 change in stock price
    - Gamma (Γ): Change in delta per $1 change in stock price
    - Theta (Θ): Change in option price per day (time decay)
    - Vega (ν): Change in option price per 1% change in volatility
    - Rho (ρ): Change in option price per 1% change in interest rate
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        sigma: Implied volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        dict with all Greeks
    """
    if T <= 0:
        # At expiration, Greeks are deterministic
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        
        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    if sigma <= 0:
        # Zero volatility edge case
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Standard normal PDF and CDF
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    # Delta
    if option_type == 'call':
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1
    
    # Gamma (same for calls and puts)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    
    # Theta (daily decay)
    if option_type == 'call':
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * cdf_d2) / 365
    else:
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (per 1% volatility change)
    vega = (S * np.sqrt(T) * pdf_d1) / 100
    
    # Rho (per 1% interest rate change)
    if option_type == 'call':
        rho = (K * T * np.exp(-r * T) * cdf_d2) / 100
    else:
        rho = -(K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
    
    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta),
        'vega': float(vega),
        'rho': float(rho)
    }


# Sentiment Analysis - fetches and analyzes news sentiment

# Track last API call for rate limiting
LAST_NEWS_API_CALL = None

def get_alpha_vantage_news(ticker, limit=50):
    """
    Get news from Alpha Vantage News API with rate limiting
    
    Args:
        ticker: Stock symbol to get news for
        limit: Maximum number of articles to retrieve
    
    Returns:
        list: Articles with sentiment data
    """
    global LAST_NEWS_API_CALL
    
    # Rate limiting - ensure at least 12 seconds between calls
    # Alpha Vantage: 25 calls/day = need to be conservative
    current_time = datetime.now().timestamp()
    if LAST_NEWS_API_CALL is not None:
        time_since_last = current_time - LAST_NEWS_API_CALL
        if time_since_last < 12:
            time.sleep(12 - time_since_last)
    
    LAST_NEWS_API_CALL = datetime.now().timestamp()
    
    try:
        url = "https://www.alphavantage.co/query"
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': os.getenv('ALPHA_VANTAGE_KEY'),
            'limit': min(limit, 50),  # API limit is 50
            'sort': 'LATEST'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            return []
        
        if 'Note' in data:
            # Rate limit hit
            raise RateLimitError("Alpha Vantage rate limit reached")
            
        if 'Information' in data:
            # API key issue or invalid input
            return []
        
        # Extract articles
        articles = data.get('feed', [])
        
        # Filter and format articles (last 7 days only)
        filtered_articles = []
        for article in articles:
            try:
                time_published = article.get('time_published', '')
                if len(time_published) >= 8:
                    # Format: YYYYMMDDTHHMMSS
                    pub_datetime = datetime.strptime(time_published[:8], '%Y%m%d')
                    
                    # Only include articles from last 7 days
                    days_old = (datetime.now() - pub_datetime).days
                    if days_old > 7:
                        continue
                    
                    filtered_articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'time_published': time_published,
                        'days_old': days_old,
                        'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'overall_sentiment_label': article.get('overall_sentiment_label', 'Neutral')
                    })
                    
            except Exception:
                # Skip malformed articles
                continue
        
        return filtered_articles
        
    except RateLimitError:
        raise  # Re-raise rate limit errors
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def calculate_time_decay_weight(days_old):
    if days_old <= 0:
        return 1.0
    elif days_old >= 7:
        return 0.1
    else:
        # Linear decay from 1.0 (today) to 0.1 (7 days old)
        return 1.0 - (days_old * 0.9 / 7.0)

def fetch_sentiment(ticker, use_cache=True):
    """
    Fetch and analyze sentiment for a ticker with caching and rate limit handling
    Args:
        ticker: Stock symbol
        use_cache: Whether to use cached data (default True)
    
    Returns:
        dict with sentiment scores and metadata
    """
    cache_key = f"{ticker}_sentiment"
    
    # Check cache first
    if use_cache and is_cached(cache_key):
        cached_data = get_from_cache(cache_key)
        cached_data['cached'] = True
        cached_data['source'] = 'cache'
        return cached_data
    
    # Try to fetch live sentiment
    try:
        # Fetch news articles
        articles = get_alpha_vantage_news(ticker, limit=50)
        
        if not articles:
            # No articles found
            result = {
                'company_sentiment': 0.0,
                'source': 'no_data',
                'cached': False,
                'article_count': 0
            }
            return result
        
        # Calculate weighted sentiment using Alpha Vantage's scores
        total_weighted_sentiment = 0.0
        total_weight = 0.0
        
        for article in articles:
            # Get Alpha Vantage's sentiment score (-1 to +1)
            sentiment_score = article['overall_sentiment_score']
            
            # Apply time decay weighting
            weight = calculate_time_decay_weight(article['days_old'])
            
            total_weighted_sentiment += sentiment_score * weight
            total_weight += weight
        
        # Calculate average weighted sentiment
        if total_weight > 0:
            company_sentiment = total_weighted_sentiment / total_weight
        else:
            company_sentiment = 0.0
        
        result = {
            'company_sentiment': company_sentiment,
            'article_count': len(articles),
            'source': 'live',
            'cached': False
        }
        
        # Cache for 1 hour
        add_to_cache(cache_key, result, TTL_SENTIMENT)
        
        return result
        
    except RateLimitError:
        # Rate limit hit and not cached
        result = {
            'company_sentiment': 0.0,
            'source': 'rate_limited',
            'cached': False,
            'article_count': 0
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching sentiment: {e}")
        
        # Return neutral sentiment
        result = {
            'company_sentiment': 0.0,
            'source': 'error',
            'cached': False,
            'article_count': 0
        }
        
        return result

# Stock Data Fetcher - downloads current price and historical data

def fetch_stock_data(ticker):
    """
    Fetch stock data using Polygon.io (replaces yfinance)

    Args:
        ticker: Stock symbol (e.g., 'AAPL')
    Returns:
        dict: {
            'current_price': float,
            'historical_prices': DataFrame with OHLCV data,
            'sector': str
        }
    """
    cache_key = f"{ticker}_stock_data"
    
    # Check cache first
    if is_cached(cache_key):
        cached_data = get_from_cache(cache_key)
        # Return a deep copy to prevent cache corruption
        return {
            'current_price': cached_data['current_price'],
            'historical_prices': cached_data['historical_prices'].copy(),
            'sector': cached_data['sector']
        }
    
    try:
        print(f"Fetching stock data for {ticker} using Polygon.io...")
        
        # Try Polygon first
        try:
            
            client = PolygonClient()
            
            # Get 2 years of historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            
            historical = client.get_stock_data(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if historical.empty or len(historical) < 100:
                raise ValueError(f"Insufficient data from Polygon: {len(historical)} rows")
            
            # Get current price (latest close)
            current_price = float(historical['Close'].iloc[-1])
            
            # Get sector from ticker details
            details = client.get_ticker_details(ticker)
            sector = details.get('sic_description', 'Unknown')
            if sector == 'Unknown' or not sector:
                # Fallback sector mapping
                sector_map = {
                    'AAPL': 'Technology',
                    'MSFT': 'Technology',
                    'GOOGL': 'Technology',
                    'AMZN': 'Consumer Cyclical',
                    'TSLA': 'Automotive',
                    'SPY': 'Index Fund',
                    'QQQ': 'Index Fund'
                }
                sector = sector_map.get(ticker, 'Unknown')
            
            print(f"✓ Polygon: Fetched {len(historical)} days for {ticker} (${current_price:.2f})")
            
        except Exception as polygon_error:
            print(f"Polygon error: {polygon_error}")
            print(f"Falling back to yfinance...")
            
            # Fallback to yfinance
            historical = yf.download(
                ticker, 
                period='2y', 
                interval='1d', 
                progress=False,
                auto_adjust=False
            )
            
            if historical.empty or len(historical) < 100:
                raise ValueError(f"Insufficient data from yfinance: {len(historical)} rows")
            
            historical = historical[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            current_price = float(historical['Close'].iloc[-1])
            
            # Get sector from yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', 'Unknown')
            except:
                sector = 'Unknown'
            
            print(f"✓ yfinance: Fetched {len(historical)} days for {ticker} (${current_price:.2f})")
        
        # Package the data
        result = {
            'current_price': current_price,
            'historical_prices': historical,
            'sector': sector
        }
        
        # Cache for 1 hour
        add_to_cache(cache_key, result, TTL_STOCK_PRICE)
        
        return result
        
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        raise ValueError(f"Could not fetch stock data for {ticker}: {str(e)}")


# VIX Data Fetcher - fetches current market volatility index

def fetch_vix():
    """
    Fetch current VIX value - using Alpha Vantage as fallback
    
    Returns:
        float: Current VIX value (e.g., 18.5)
    """
    cache_key = "vix_current"
    
    # Check cache first
    if is_cached(cache_key):
        return get_from_cache(cache_key)
    
    # Try Alpha Vantage first (more reliable)
    try:
        api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if api_key:
            time.sleep(1)
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'VIX',
                'apikey': api_key,
                'outputsize': 'compact'
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'Time Series (Daily)' in data:
                    latest_date = list(data['Time Series (Daily)'].keys())[0]
                    vix_value = float(data['Time Series (Daily)'][latest_date]['4. close'])
                    add_to_cache(cache_key, vix_value, TTL_VIX)
                    return vix_value
    except Exception as e:
        print(f"Alpha Vantage VIX error: {e}")
    
    # Return current VIX as default
    default_vix = 19.08  # Current VIX as of Nov 8, 2025
    print(f"Using default VIX: {default_vix}")
    add_to_cache(cache_key, default_vix, TTL_VIX)
    return default_vix

# Risk-Free Rate Fetcher - fetches 10-year Treasury yield from FRED

def fetch_risk_free_rate():
    """
    Fetch current risk-free rate (10-year Treasury) from FRED
    
    Returns:
        float: Annual rate as decimal (e.g., 0.045 for 4.5%)
    """
    cache_key = "risk_free_rate"
    
    # Check cache first
    if is_cached(cache_key):
        return get_from_cache(cache_key)
    
    try:
        # Initialize FRED API
        fred_api_key = os.getenv('FRED_KEY')
        if not fred_api_key:
            raise ValueError("FRED_KEY not found in environment variables")
        
        fred = Fred(api_key=fred_api_key)
        
        # Fetch 10-Year Treasury Constant Maturity Rate (DGS10)
        treasury_data = fred.get_series('DGS10')
        
        if len(treasury_data) == 0:
            raise ValueError("No Treasury data available")
        
        # Get most recent rate and convert from percentage to decimal
        latest_rate = float(treasury_data.iloc[-1])
        rate_decimal = latest_rate / 100.0
        
        
        # Cache for 1 hour
        add_to_cache(cache_key, rate_decimal, TTL_RISK_FREE)
        
        return rate_decimal
        
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        # Return reasonable default (current ~4.5%)
        return 0.045


# Earnings Date Fetcher - fetches next earnings announcement date

def fetch_earnings_date(ticker):
    """
    Fetch next earnings announcement date from Alpha Vantage
    
    Args:
        ticker: Stock symbol
    
    Returns:
        dict: {
            'next_earnings_date': 'YYYY-MM-DD' or None,
            'days_until_earnings': int or None
        }
    """
    cache_key = f"{ticker}_earnings"
    
    # Check cache first
    if is_cached(cache_key):
        return get_from_cache(cache_key)
    
    try:
        api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_KEY not found in environment variables")
        
        time.sleep(1)  # Rate limiting
        
        # Alpha Vantage EARNINGS endpoint
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'EARNINGS',
            'symbol': ticker.upper(),
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors
        if 'Error Message' in data:
            raise ValueError(data['Error Message'])
        if 'Note' in data:
            raise ValueError("Alpha Vantage rate limit reached")
        
        # Get quarterly earnings (future dates)
        quarterly_earnings = data.get('quarterlyEarnings', [])
        
        if not quarterly_earnings:
            result = {
                'next_earnings_date': None,
                'days_until_earnings': None
            }
        else:
            # Find the next future earnings date
            today = datetime.now().date()
            future_earnings = []
            
            for earning in quarterly_earnings:
                try:
                    # reportedDate is the actual earnings date
                    date_str = earning.get('reportedDate')
                    if not date_str:
                        continue
                    
                    earning_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Alpha Vantage shows historical, so we need to estimate next one
                    # Typically earnings are quarterly (90 days apart)
                    if earning_date >= today:
                        future_earnings.append({'date': date_str, 'earning_date': earning_date})
                except Exception:
                    continue
            
            if future_earnings:
                # Sort and get the earliest
                future_earnings.sort(key=lambda x: x['earning_date'])
                next_earning = future_earnings[0]
                days_until = (next_earning['earning_date'] - today).days
                
                result = {
                    'next_earnings_date': next_earning['date'],
                    'days_until_earnings': days_until
                }
            else:
                # No future earnings found - estimate based on last earnings
                # Most companies report quarterly (every ~90 days)
                if quarterly_earnings:
                    last_earning_str = quarterly_earnings[0].get('reportedDate')
                    if last_earning_str:
                        try:
                            last_date = datetime.strptime(last_earning_str, '%Y-%m-%d').date()
                            # Estimate next earnings as ~90 days from last
                            estimated_next = last_date + timedelta(days=90)
                            
                            # Keep adding quarters until we get a future date
                            while estimated_next < today:
                                estimated_next += timedelta(days=90)
                            
                            days_until = (estimated_next - today).days
                            
                            result = {
                                'next_earnings_date': estimated_next.strftime('%Y-%m-%d'),
                                'days_until_earnings': days_until
                            }
                        except Exception:
                            result = {
                                'next_earnings_date': None,
                                'days_until_earnings': None
                            }
                    else:
                        result = {
                            'next_earnings_date': None,
                            'days_until_earnings': None
                        }
                else:
                    result = {
                        'next_earnings_date': None,
                        'days_until_earnings': None
                    }
        
        # Cache for 24 hours
        add_to_cache(cache_key, result, TTL_EARNINGS)
        
        return result
        
    except Exception as e:
        print(f"Error fetching earnings date for {ticker}: {e}")
        # Return None values if error
        return {
            'next_earnings_date': None,
            'days_until_earnings': None
        }


# Feature Engineering - Phase 2

def calculate_technical_indicators(df):
    """
    Calculate technical indicators optimized for short-term options (up to 30 days)
    
    Args:
        df: DataFrame with OHLCV data (from Alpha Vantage)
    
    Returns:
        DataFrame with added technical indicator columns
    """
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure all price columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. RSI - Relative Strength Index (14-day standard)
        # Values: 0-100 | >70 = overbought, <30 = oversold
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi_indicator.rsi()
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            df['RSI'] = np.nan
        
        # 2. MACD - Moving Average Convergence Divergence
        # Trend and momentum indicator
        try:
            macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            df['MACD'] = df['MACD_signal'] = df['MACD_histogram'] = np.nan
        
        # 3. Bollinger Bands (20-day standard)
        # Volatility and support/resistance
        try:
            bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            df['BB_upper'] = df['BB_lower'] = df['BB_middle'] = df['BB_width'] = df['BB_position'] = np.nan
        
        # 4. Moving Averages - Short-term for options trading
        # SMA_10, SMA_20 more relevant than SMA_50/200 for 30-day options
        try:
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages (more weight to recent prices)
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            df['SMA_10'] = df['SMA_20'] = df['SMA_50'] = df['EMA_10'] = df['EMA_20'] = np.nan
        
        # 5. ADX - Average Directional Index (trend strength)
        # >25 = strong trend, <20 = weak/choppy
        try:
            adx = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['ADX'] = adx.adx()
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            df['ADX'] = np.nan
        
        # 6. ATR - Average True Range (volatility measure)
        # Higher ATR = higher volatility = wider price swings
        try:
            atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['ATR'] = atr.average_true_range()
            # Normalize ATR by price for comparison across stocks
            df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            df['ATR'] = df['ATR_pct'] = np.nan
        
        # 7. Volume indicators
        try:
            # Volume moving average
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            # Volume ratio (current vs average)
            df['Volume_ratio'] = df['Volume'] / df['Volume_MA_20']
            # Volume change
            df['Volume_change'] = df['Volume'].pct_change()
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
            df['Volume_MA_20'] = df['Volume_ratio'] = df['Volume_change'] = np.nan
        
        # 8. Momentum indicators (important for short-term options)
        try:
            # Rate of Change - 10 day
            df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
            # Price momentum (10-day)
            df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
            # Recent returns
            df['Returns_1d'] = df['Close'].pct_change(1)
            df['Returns_5d'] = df['Close'].pct_change(5)
            df['Returns_10d'] = df['Close'].pct_change(10)
        except Exception as e:
            print(f"Error calculating momentum indicators: {e}")
            df['ROC_10'] = df['Momentum_10'] = df['Returns_1d'] = df['Returns_5d'] = df['Returns_10d'] = np.nan
        
        # 9. Stochastic Oscillator (momentum indicator for short-term)
        # Values: 0-100 | >80 = overbought, <20 = oversold
        try:
            stoch = ta.momentum.StochasticOscillator(
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'], 
                window=14, 
                smooth_window=3
            )
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            df['Stoch_K'] = df['Stoch_D'] = np.nan
        
        # 10. Price position relative to moving averages (key for trend identification)
        try:
            df['Price_vs_SMA10'] = ((df['Close'] - df['SMA_10']) / df['SMA_10']) * 100
            df['Price_vs_SMA20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
        except Exception as e:
            print(f"Error calculating price vs MA: {e}")
            df['Price_vs_SMA10'] = df['Price_vs_SMA20'] = np.nan
        
        return df
        
    except Exception as e:
        print(f"Error in calculate_technical_indicators: {e}")
        return df


def calculate_historical_volatility(df, windows=[30, 60, 90]):
    """
    Calculate comprehensive historical volatility metrics
    
    Args:
        df: DataFrame with Close prices
        windows: List of lookback periods in days (default: [30, 60, 90])
    
    Returns:
        dict: {
            'HV_30': float (annualized volatility),
            'HV_60': float,
            'HV_90': float,
            'HV_slope': float (trend direction),
            'HV_regime': str ('expanding' or 'contracting'),
            'HV_percentile': float (0-1),
            'HV_z_score': float (standard deviations from mean),
            'daily_volatility': float (current daily vol),
            'HV_acceleration': float (rate of vol change)
        }
    """
    try:
        # Calculate daily returns (log returns are more accurate for volatility)
        returns = np.log(df['Close'] / df['Close'].shift(1))
        returns = returns.dropna()
        
        if len(returns) < max(windows):
            print(f"Insufficient data for volatility calculation. Need {max(windows)} days, have {len(returns)}")
            return None
        
        result = {}
        
        # Calculate HV for each window
        hv_values = {}
        for window in windows:
            # Standard deviation of returns over the window
            std_dev = returns.rolling(window=window).std()
            
            # Annualize: multiply by sqrt(252 trading days)
            annualized_vol = std_dev * 15.8745  # sqrt(252)
            
            # Get the most recent value
            latest_hv = annualized_vol.iloc[-1]
            
            # Store in result
            key = f'HV_{window}'
            result[key] = float(latest_hv)
            hv_values[window] = float(latest_hv)
        
        # Current daily volatility (not annualized)
        result['daily_volatility'] = float(returns.std())
        if 30 in hv_values and 90 in hv_values and hv_values[90] > 0:
            result['HV_slope'] = (hv_values[30] - hv_values[90]) / hv_values[90]
        else:
            result['HV_slope'] = 0.0
        
        # HV Regime
        if result['HV_slope'] > 0.05:  # More than 5% increase
            result['HV_regime'] = 'expanding'
        elif result['HV_slope'] < -0.05:  # More than 5% decrease
            result['HV_regime'] = 'contracting'
        else:
            result['HV_regime'] = 'stable'
        
        # HV Acceleration
        # (HV_30 - HV_60) - (HV_60 - HV_90)
        if 30 in hv_values and 60 in hv_values and 90 in hv_values:
            recent_change = hv_values[30] - hv_values[60]
            older_change = hv_values[60] - hv_values[90]
            result['HV_acceleration'] = recent_change - older_change
        else:
            result['HV_acceleration'] = 0.0
        
        # Calculate HV_30 for entire history
        std_dev_30 = returns.rolling(window=30).std()
        annualized_vol_30 = std_dev_30 * 15.8745  # sqrt(252)

        # Get last year of HV values (252 trading days)
        hv_history = annualized_vol_30.dropna().tail(252)
        
        if len(hv_history) > 0:
            current_hv = hv_values[30]
            min_hv = hv_history.min()
            max_hv = hv_history.max()
            
            if max_hv > min_hv:
                result['HV_percentile'] = (current_hv - min_hv) / (max_hv - min_hv)
            else:
                result['HV_percentile'] = 0.5
            
            # HV Z-Score: How many standard deviations from mean?
            mean_hv = hv_history.mean()
            std_hv = hv_history.std()
            
            if std_hv > 0:
                result['HV_z_score'] = (current_hv - mean_hv) / std_hv
            else:
                result['HV_z_score'] = 0.0
        else:
            result['HV_percentile'] = 0.5
            result['HV_z_score'] = 0.0
        
        return result
        
    except Exception as e:
        print(f"Error calculating historical volatility: {e}")
        return None


def calculate_option_features(strike_price, expiration_date, option_type, 
                              current_price, sector, hv_metrics, earnings_data, 
                              risk_free_rate, vix, ticker = "AAPL"):
    """
    Calculate option-specific features for ML models
    Args:
        ticker: Stock symbol
        strike_price: Option strike price
        expiration_date: Option expiration (datetime or string 'YYYY-MM-DD')
        option_type: 'call' or 'put'
        current_price: Current stock price
        sector: Company sector
        hv_metrics: Dict from calculate_historical_volatility()
        earnings_data: Dict from fetch_earnings_date()
        risk_free_rate: Current risk-free rate (decimal)
        vix: Current VIX level
        
    Returns:
        Dict with 27 option-specific features
    """
    # Convert expiration to datetime if string
    if isinstance(expiration_date, str):
        expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    
    # Calculate DTE
    dte = (expiration_date - datetime.now()).days
    time_to_expiration = dte / 365.25
    
    # 1. MONEYNESS FEATURES (4 features)
    moneyness = current_price / strike_price
    
    # ITM Amount (intrinsic value)
    if option_type == 'call':
        itm_amount = max(0, current_price - strike_price)
    else:  # put
        itm_amount = max(0, strike_price - current_price)
    
    itm_percent = itm_amount / strike_price if itm_amount > 0 else 0
    otm_distance = abs(current_price - strike_price) / current_price

    
    # Time regime classification
    if dte > 60:
        time_regime_code = 2  # long_term
    elif dte > 30:
        time_regime_code = 1  # medium_term
    else:
        time_regime_code = 0  # short_term
    
    # Theta multiplier (time decay accelerates near expiration)
    theta_multiplier = 1 / np.sqrt(dte) if dte > 0 else 0
    
    # 3. EARNINGS IMPACT FEATURES (3 features)
    
    days_to_earnings = earnings_data.get('days_until_earnings')
    
    # Will earnings occur before expiration?
    earnings_before_expiration = 1 if (days_to_earnings is not None and 
                                       days_to_earnings < dte) else 0
    
    # Earnings risk premium (options more expensive if earnings before expiration)
    if earnings_before_expiration and days_to_earnings is not None:
        earnings_risk = min(1.0, (dte - days_to_earnings) / dte)
    else:
        earnings_risk = 0.0
    
    hv_30 = hv_metrics['HV_30']
    
    # Black-Scholes d1 and d2
    if time_to_expiration > 0 and hv_30 > 0:
        d1 = (np.log(current_price / strike_price) + 
              (risk_free_rate + 0.5 * hv_30**2) * time_to_expiration) / \
             (hv_30 * np.sqrt(time_to_expiration))
        
        d2 = d1 - hv_30 * np.sqrt(time_to_expiration)
    else:
        d1 = 0
        d2 = 0
    
    # Delta approximation
    if time_to_expiration > 0:
        if option_type == 'call':
            delta_approx = norm.cdf(d1)
        else:  # put
            delta_approx = norm.cdf(d1) - 1
    else:
        # At expiration, delta is binary
        if option_type == 'call':
            delta_approx = 1 if current_price > strike_price else 0
        else:
            delta_approx = -1 if current_price < strike_price else 0
    
    # Gamma approximation
    gamma_approx = norm.pdf(d1) / (current_price * hv_30 * np.sqrt(time_to_expiration)) \
                   if time_to_expiration > 0 and hv_30 > 0 else 0
    
    # Vega approximation (sensitivity to volatility)
    vega_approx = current_price * np.sqrt(time_to_expiration) * norm.pdf(d1) / 100 \
                  if time_to_expiration > 0 else 0
    
    # Theta approximation (daily decay)
    if time_to_expiration > 0:
        theta_approx = -(current_price * hv_30 * norm.pdf(d1)) / (2 * np.sqrt(time_to_expiration))
    else:
        theta_approx = 0
    
    # Rho approximation (interest rate sensitivity)
    if time_to_expiration > 0:
        if option_type == 'call':
            rho_approx = strike_price * time_to_expiration * np.exp(-risk_free_rate * time_to_expiration) * \
                         norm.cdf(d2) / 100
        else:
            rho_approx = -strike_price * time_to_expiration * np.exp(-risk_free_rate * time_to_expiration) * \
                         norm.cdf(-d2) / 100
    else:
        rho_approx = 0
    
    hv_percentile = hv_metrics['HV_percentile']
    vol_skew = hv_metrics['HV_slope']
    vix_ratio = vix / 20.0  # Normalize to typical VIX ~20

    # Sector encoding (label encoding for ML)
    sector_codes = {
        'Technology': 0,
        'Healthcare': 1,
        'Financial Services': 2,
        'Consumer Cyclical': 3,
        'Communication Services': 4,
        'Industrials': 5,
        'Consumer Defensive': 6,
        'Energy': 7,
        'Basic Materials': 8,
        'Real Estate': 9,
        'Utilities': 10,
        'Unknown': 11
    }
    sector_code = sector_codes.get(sector, 11)
    
    # Enhanced VIX regime classification (granular levels)
    if vix < 14:
        vix_regime_code = 0   # Very Low (complacent market)
    elif vix < 15:
        vix_regime_code = 1   # Low
    elif vix < 16:
        vix_regime_code = 2
    elif vix < 17:
        vix_regime_code = 3
    elif vix < 18:
        vix_regime_code = 4
    elif vix < 19:
        vix_regime_code = 5
    elif vix < 20:
        vix_regime_code = 6   # Normal low
    elif vix < 21:
        vix_regime_code = 7
    elif vix < 22:
        vix_regime_code = 8
    elif vix < 23:
        vix_regime_code = 9
    elif vix < 24:
        vix_regime_code = 10
    elif vix < 25:
        vix_regime_code = 11  # Elevated
    else:
        vix_regime_code = 12  # High (fear/uncertainty)
    
    is_call = 1 if option_type == 'call' else 0
    
    # Return all features
    return {
        # Moneyness (4)
        'moneyness': moneyness,
        'itm_amount': itm_amount,
        'itm_percent': itm_percent,
        'otm_distance': otm_distance,
        
        # Time (4)
        'dte': dte,
        'time_to_expiration': time_to_expiration,
        'time_regime': time_regime_code,
        'theta_multiplier': theta_multiplier,
        
        # Earnings (3)
        'days_to_earnings': days_to_earnings if days_to_earnings is not None else -1,
        'earnings_before_expiration': earnings_before_expiration,
        'earnings_risk': earnings_risk,
        
        # Greeks (5)
        'delta': delta_approx,
        'gamma': gamma_approx,
        'vega': vega_approx,
        'theta': theta_approx,
        'rho': rho_approx,
        
        # Volatility Context (3)
        'hv_percentile': hv_percentile,
        'vol_skew': vol_skew,
        'vix_ratio': vix_ratio,
        
        # Market Context (5)
        'sector_code': sector_code,
        'risk_free_rate': risk_free_rate,
        'vix': vix,
        'vix_regime': vix_regime_code,
        
        # Option Type (1)
        'is_call': is_call
    }


# Master Data Orchestrator - combines all data fetchers

def fetch_all_market_data(ticker):
    """
    Master function to fetch all required market data (user provides option details separately)
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
    
    Returns:
        dict: Combined data from all sources with error handling
    """
    results = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'stock_data': None,
        'sentiment': None,
        'earnings': None,
        'vix': None,
        'risk_free_rate': None,
        'errors': []
    }
    
    # Fetch stock data
    try:
        results['stock_data'] = fetch_stock_data(ticker)
    except Exception as e:
        results['errors'].append(f"Stock data error: {str(e)}")
    
    # Fetch sentiment
    try:
        results['sentiment'] = fetch_sentiment(ticker, use_cache=True)
    except Exception as e:
        results['errors'].append(f"Sentiment error: {str(e)}")
        results['sentiment'] = {'company_sentiment': 0.0, 'source': 'error', 'cached': False, 'article_count': 0}
    
    # Fetch earnings date
    try:
        results['earnings'] = fetch_earnings_date(ticker)
    except Exception as e:
        results['errors'].append(f"Earnings error: {str(e)}")
        results['earnings'] = {'next_earnings_date': None, 'days_until_earnings': None}
    
    # Fetch VIX
    try:
        results['vix'] = fetch_vix()
    except Exception as e:
        results['errors'].append(f"VIX error: {str(e)}")
        results['vix'] = 19.08
    
    # Fetch risk-free rate
    try:
        results['risk_free_rate'] = fetch_risk_free_rate()
    except Exception as e:
        results['errors'].append(f"Risk-free rate error: {str(e)}")
        results['risk_free_rate'] = 0.045
    
    return results


# Flask Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_size': len(cache),
        'cache_file': CACHE_FILE,
        'cache_file_exists': os.path.exists(CACHE_FILE)
    })


@app.route('/api/cache/info', methods=['GET'])
def cache_info():
    """Get detailed cache information"""
    cache_details = []
    current_time = datetime.now().timestamp()
    
    for key, value in cache.items():
        age = current_time - value['timestamp']
        ttl_remaining = value['ttl'] - age
        cache_details.append({
            'key': key,
            'age_seconds': round(age, 1),
            'ttl_seconds': value['ttl'],
            'ttl_remaining': round(ttl_remaining, 1),
            'expired': ttl_remaining <= 0
        })
    
    return jsonify({
        'success': True,
        'total_items': len(cache),
        'cache_file': CACHE_FILE,
        'cache_file_exists': os.path.exists(CACHE_FILE),
        'items': cache_details
    })


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cache (use with caution)"""
    global cache
    cache = {}
    save_cache_to_disk()
    return jsonify({
        'success': True,
        'message': 'Cache cleared'
    })


@app.route('/api/test', methods=['GET'])
def test_data_pipeline():
    """Test endpoint to verify data fetching works"""
    try:
        # Test with AAPL
        data = fetch_all_market_data(ticker='AAPL')
        
        # Convert DataFrame to dict for JSON serialization
        if data.get('stock_data') and 'historical_prices' in data['stock_data']:
            hist = data['stock_data']['historical_prices']
            data['stock_data']['historical_prices'] = {
                'rows': len(hist),
                'columns': list(hist.columns),
                'latest_date': str(hist.index[-1]),
                'latest_close': float(hist['Close'].iloc[-1])
            }
        
        return jsonify({
            'success': True,
            'data': data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test-indicators', methods=['GET'])
def test_technical_indicators():
    """Test endpoint to verify technical indicators calculation"""
    try:
        # Fetch stock data
        stock_data = fetch_stock_data('AAPL')
        
        # Get historical DataFrame
        hist_df = stock_data['historical_prices']
        
        # Calculate technical indicators
        enriched_df = calculate_technical_indicators(hist_df)
        
        # Get the latest row with all indicators
        latest = enriched_df.iloc[-1]
        
        # Build response with all calculated indicators
        indicators = {
            'date': str(enriched_df.index[-1]),
            'price_data': {
                'Close': float(latest['Close']),
                'Open': float(latest['Open']),
                'High': float(latest['High']),
                'Low': float(latest['Low']),
                'Volume': int(latest['Volume'])
            },
            'momentum_indicators': {
                'RSI': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
                'MACD': float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
                'MACD_signal': float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None,
                'MACD_histogram': float(latest['MACD_histogram']) if not pd.isna(latest['MACD_histogram']) else None,
                'Stoch_K': float(latest['Stoch_K']) if not pd.isna(latest['Stoch_K']) else None,
                'Stoch_D': float(latest['Stoch_D']) if not pd.isna(latest['Stoch_D']) else None,
                'ROC_10': float(latest['ROC_10']) if not pd.isna(latest['ROC_10']) else None,
                'Momentum_10': float(latest['Momentum_10']) if not pd.isna(latest['Momentum_10']) else None
            },
            'trend_indicators': {
                'SMA_10': float(latest['SMA_10']) if not pd.isna(latest['SMA_10']) else None,
                'SMA_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
                'SMA_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
                'EMA_10': float(latest['EMA_10']) if not pd.isna(latest['EMA_10']) else None,
                'EMA_20': float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None,
                'ADX': float(latest['ADX']) if not pd.isna(latest['ADX']) else None,
                'Price_vs_SMA10': float(latest['Price_vs_SMA10']) if not pd.isna(latest['Price_vs_SMA10']) else None,
                'Price_vs_SMA20': float(latest['Price_vs_SMA20']) if not pd.isna(latest['Price_vs_SMA20']) else None
            },
            'volatility_indicators': {
                'BB_upper': float(latest['BB_upper']) if not pd.isna(latest['BB_upper']) else None,
                'BB_lower': float(latest['BB_lower']) if not pd.isna(latest['BB_lower']) else None,
                'BB_middle': float(latest['BB_middle']) if not pd.isna(latest['BB_middle']) else None,
                'BB_width': float(latest['BB_width']) if not pd.isna(latest['BB_width']) else None,
                'BB_position': float(latest['BB_position']) if not pd.isna(latest['BB_position']) else None,
                'ATR': float(latest['ATR']) if not pd.isna(latest['ATR']) else None,
                'ATR_pct': float(latest['ATR_pct']) if not pd.isna(latest['ATR_pct']) else None
            },
            'volume_indicators': {
                'Volume_MA_20': float(latest['Volume_MA_20']) if not pd.isna(latest['Volume_MA_20']) else None,
                'Volume_ratio': float(latest['Volume_ratio']) if not pd.isna(latest['Volume_ratio']) else None,
                'Volume_change': float(latest['Volume_change']) if not pd.isna(latest['Volume_change']) else None
            },
            'returns': {
                'Returns_1d': float(latest['Returns_1d']) if not pd.isna(latest['Returns_1d']) else None,
                'Returns_5d': float(latest['Returns_5d']) if not pd.isna(latest['Returns_5d']) else None,
                'Returns_10d': float(latest['Returns_10d']) if not pd.isna(latest['Returns_10d']) else None
            }
        }
        
        # Add summary stats
        summary = {
            'total_indicators_calculated': len([col for col in enriched_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]),
            'rows_with_indicators': len(enriched_df),
            'date_range': {
                'start': str(enriched_df.index[0]),
                'end': str(enriched_df.index[-1])
            }
        }
        
        return jsonify({
            'success': True,
            'ticker': 'AAPL',
            'latest_indicators': indicators,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test-volatility', methods=['GET'])
def test_historical_volatility():
    """Test endpoint to verify historical volatility calculation"""
    try:
        # Fetch stock data
        stock_data = fetch_stock_data('AAPL')
        
        # Get historical DataFrame
        hist_df = stock_data['historical_prices']
        
        # Calculate historical volatility
        hv_metrics = calculate_historical_volatility(hist_df, windows=[30, 60, 90])
        
        if hv_metrics is None:
            return jsonify({
                'success': False,
                'error': 'Could not calculate historical volatility'
            }), 500
        
        # Calculate technical indicators for context
        enriched_df = calculate_technical_indicators(hist_df)
        latest = enriched_df.iloc[-1]
        
        # Build comprehensive response
        response = {
            'success': True,
            'ticker': 'AAPL',
            'current_price': float(stock_data['current_price']),
            'timestamp': datetime.now().isoformat(),
            
            # Historical Volatility Metrics
            'historical_volatility': {
                'HV_30': {
                    'value': round(hv_metrics['HV_30'], 4),
                    'percentage': f"{round(hv_metrics['HV_30'] * 100, 2)}%",
                    'description': '30-day annualized historical volatility'
                },
                'HV_60': {
                    'value': round(hv_metrics['HV_60'], 4),
                    'percentage': f"{round(hv_metrics['HV_60'] * 100, 2)}%",
                    'description': '60-day annualized historical volatility'
                },
                'HV_90': {
                    'value': round(hv_metrics['HV_90'], 4),
                    'percentage': f"{round(hv_metrics['HV_90'] * 100, 2)}%",
                    'description': '90-day annualized historical volatility'
                },
                'daily_volatility': {
                    'value': round(hv_metrics['daily_volatility'], 4),
                    'percentage': f"{round(hv_metrics['daily_volatility'] * 100, 2)}%",
                    'description': 'Current daily volatility (not annualized)'
                }
            },
            
            # Volatility Analysis
            'volatility_analysis': {
                'regime': {
                    'status': hv_metrics['HV_regime'],
                    'description': {
                        'expanding': 'Volatility is INCREASING - options getting more expensive',
                        'contracting': 'Volatility is DECREASING - options getting cheaper',
                        'stable': 'Volatility is STABLE - consistent pricing'
                    }[hv_metrics['HV_regime']]
                },
                'slope': {
                    'value': round(hv_metrics['HV_slope'], 4),
                    'percentage': f"{round(hv_metrics['HV_slope'] * 100, 2)}%",
                    'description': 'HV_30 vs HV_90 trend (positive = expanding)'
                },
                'acceleration': {
                    'value': round(hv_metrics['HV_acceleration'], 4),
                    'description': 'Rate of volatility change (positive = accelerating expansion)',
                    'interpretation': 'Explosive' if hv_metrics['HV_acceleration'] > 0.02 else 
                                    'Steady' if abs(hv_metrics['HV_acceleration']) <= 0.02 else 
                                    'Decelerating'
                },
                'percentile': {
                    'value': round(hv_metrics['HV_percentile'], 4),
                    'percentage': f"{round(hv_metrics['HV_percentile'] * 100, 1)}%",
                    'description': 'Where current HV ranks in 1-year range',
                    'interpretation': 'Extremely High' if hv_metrics['HV_percentile'] > 0.8 else
                                    'High' if hv_metrics['HV_percentile'] > 0.6 else
                                    'Average' if hv_metrics['HV_percentile'] > 0.4 else
                                    'Low' if hv_metrics['HV_percentile'] > 0.2 else
                                    'Extremely Low'
                },
                'z_score': {
                    'value': round(hv_metrics['HV_z_score'], 2),
                    'description': 'Standard deviations from 1-year mean HV',
                    'interpretation': 'Extreme' if abs(hv_metrics['HV_z_score']) > 2 else
                                    'Elevated' if abs(hv_metrics['HV_z_score']) > 1 else
                                    'Normal'
                }
            },
            
            # Expected Price Movement (based on HV_30)
            'expected_moves': {
                '30_day': {
                    'one_std_dev': round(stock_data['current_price'] * hv_metrics['HV_30'] * np.sqrt(30/252), 2),
                    'range_lower': round(stock_data['current_price'] * (1 - hv_metrics['HV_30'] * np.sqrt(30/252)), 2),
                    'range_upper': round(stock_data['current_price'] * (1 + hv_metrics['HV_30'] * np.sqrt(30/252)), 2),
                    'probability': '68%',
                    'description': '68% probability price stays within this range in 30 days'
                },
                '1_day': {
                    'one_std_dev': round(stock_data['current_price'] * hv_metrics['daily_volatility'], 2),
                    'range_lower': round(stock_data['current_price'] * (1 - hv_metrics['daily_volatility']), 2),
                    'range_upper': round(stock_data['current_price'] * (1 + hv_metrics['daily_volatility']), 2),
                    'probability': '68%',
                    'description': 'Expected daily price range'
                }
            },
            
            # Related Technical Indicators
            'related_indicators': {
                'ATR': {
                    'value': round(float(latest['ATR']), 2),
                    'percentage': round(float(latest['ATR_pct']), 2),
                    'description': 'Average True Range (actual daily price swings)'
                },
                'BB_width': {
                    'value': round(float(latest['BB_width']), 4),
                    'percentage': f"{round(float(latest['BB_width']) * 100, 2)}%",
                    'description': 'Bollinger Band width (another volatility measure)'
                }
            },
            
            # Trading Implications
            'options_strategy_hints': {
                'current_state': f"HV is at {round(hv_metrics['HV_percentile'] * 100, 1)}th percentile - " +
                                f"Volatility is {hv_metrics['HV_regime']}",
                'recommendation': 'Consider SELLING options (collect high premium)' if hv_metrics['HV_percentile'] > 0.7
                                else 'Consider BUYING options (cheaper premiums)' if hv_metrics['HV_percentile'] < 0.3
                                else 'Neutral - use directional strategies',
                'iv_expectation': 'IV likely elevated - options expensive' if hv_metrics['HV_regime'] == 'expanding'
                                else 'IV likely falling - options getting cheaper' if hv_metrics['HV_regime'] == 'contracting'
                                else 'IV likely stable'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/features/<ticker>', methods=['GET'])
def get_ml_features(ticker):
    """
    Get ML-ready features (full DataFrames + calculated metrics)
    This endpoint returns ONLY numerical values needed for ML models
    
    Optional query params for option-specific features:
        - strike: Strike price
        - expiration: Expiration date (YYYY-MM-DD)
        - type: call or put
    """
    try:
        ticker = ticker.upper()
        
        # Fetch all market data
        market_data = fetch_all_market_data(ticker)
        
        # Check if we have required data
        if not market_data.get('stock_data'):
            return jsonify({
                'success': False,
                'error': 'Could not fetch stock data',
                'details': market_data.get('errors', [])
            }), 500
        
        # Get historical DataFrame
        stock_data = market_data['stock_data']
        hist_df = stock_data['historical_prices']
        
        # Calculate technical indicators (returns full DataFrame)
        enriched_df = calculate_technical_indicators(hist_df)
        
        # Calculate historical volatility metrics
        hv_metrics = calculate_historical_volatility(hist_df, windows=[30, 60, 90])
        
        # Get latest values for summary
        latest = enriched_df.iloc[-1]
        current_price = stock_data['current_price']
        sector = stock_data['sector']
        
        # Build ML-ready response (NO strings, only numbers and DataFrames)
        response = {
            'success': True,
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            
            # Stock info
            'current_price': float(current_price),
            'sector': sector,
            
            # Full DataFrame with all indicators (for ML training/prediction)
            'dataframe': {
                'rows': len(enriched_df),
                'columns': list(enriched_df.columns),
                'date_range': {
                    'start': str(enriched_df.index[0]),
                    'end': str(enriched_df.index[-1])
                },
                # Convert to records for JSON (array of dicts, one per date)
                'data': enriched_df.reset_index().to_dict('records')
            },
            
            # Latest values (for display/quick reference)
            'latest_values': {
                'date': str(enriched_df.index[-1]),
                'price': {
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'close': float(latest['Close']),
                    'volume': int(latest['Volume'])
                },
                'momentum': {
                    'RSI': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
                    'MACD': float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
                    'MACD_signal': float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None,
                    'MACD_histogram': float(latest['MACD_histogram']) if not pd.isna(latest['MACD_histogram']) else None,
                    'Stoch_K': float(latest['Stoch_K']) if not pd.isna(latest['Stoch_K']) else None,
                    'Stoch_D': float(latest['Stoch_D']) if not pd.isna(latest['Stoch_D']) else None,
                    'ROC_10': float(latest['ROC_10']) if not pd.isna(latest['ROC_10']) else None,
                    'Momentum_10': float(latest['Momentum_10']) if not pd.isna(latest['Momentum_10']) else None
                },
                'trend': {
                    'SMA_10': float(latest['SMA_10']) if not pd.isna(latest['SMA_10']) else None,
                    'SMA_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
                    'SMA_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
                    'EMA_10': float(latest['EMA_10']) if not pd.isna(latest['EMA_10']) else None,
                    'EMA_20': float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None,
                    'ADX': float(latest['ADX']) if not pd.isna(latest['ADX']) else None,
                    'Price_vs_SMA10': float(latest['Price_vs_SMA10']) if not pd.isna(latest['Price_vs_SMA10']) else None,
                    'Price_vs_SMA20': float(latest['Price_vs_SMA20']) if not pd.isna(latest['Price_vs_SMA20']) else None
                },
                'volatility': {
                    'BB_upper': float(latest['BB_upper']) if not pd.isna(latest['BB_upper']) else None,
                    'BB_lower': float(latest['BB_lower']) if not pd.isna(latest['BB_lower']) else None,
                    'BB_middle': float(latest['BB_middle']) if not pd.isna(latest['BB_middle']) else None,
                    'BB_width': float(latest['BB_width']) if not pd.isna(latest['BB_width']) else None,
                    'BB_position': float(latest['BB_position']) if not pd.isna(latest['BB_position']) else None,
                    'ATR': float(latest['ATR']) if not pd.isna(latest['ATR']) else None,
                    'ATR_pct': float(latest['ATR_pct']) if not pd.isna(latest['ATR_pct']) else None
                },
                'volume': {
                    'Volume_MA_20': float(latest['Volume_MA_20']) if not pd.isna(latest['Volume_MA_20']) else None,
                    'Volume_ratio': float(latest['Volume_ratio']) if not pd.isna(latest['Volume_ratio']) else None,
                    'Volume_change': float(latest['Volume_change']) if not pd.isna(latest['Volume_change']) else None
                },
                'returns': {
                    'Returns_1d': float(latest['Returns_1d']) if not pd.isna(latest['Returns_1d']) else None,
                    'Returns_5d': float(latest['Returns_5d']) if not pd.isna(latest['Returns_5d']) else None,
                    'Returns_10d': float(latest['Returns_10d']) if not pd.isna(latest['Returns_10d']) else None
                }
            },
            
            # Historical Volatility (pure numbers only)
            'historical_volatility': {
                'HV_30': hv_metrics['HV_30'],
                'HV_60': hv_metrics['HV_60'],
                'HV_90': hv_metrics['HV_90'],
                'daily_volatility': hv_metrics['daily_volatility'],
                'HV_slope': hv_metrics['HV_slope'],
                'HV_regime': hv_metrics['HV_regime'],  # 'expanding', 'contracting', or 'stable'
                'HV_acceleration': hv_metrics['HV_acceleration'],
                'HV_percentile': hv_metrics['HV_percentile'],
                'HV_z_score': hv_metrics['HV_z_score']
            }
        }
        
        # Optional: Calculate option-specific features if params provided
        strike = request.args.get('strike', type=float)
        expiration = request.args.get('expiration', type=str)
        option_type = request.args.get('type', type=str)
        
        if strike and expiration and option_type:
            option_features = calculate_option_features(
                ticker=ticker,
                strike_price=strike,
                expiration_date=expiration,
                option_type=option_type,
                current_price=current_price,
                sector=sector,
                hv_metrics=hv_metrics,
                earnings_data=market_data['earnings'],
                risk_free_rate=market_data['risk_free_rate'],
                vix=market_data['vix']
            )
            response['option_features'] = option_features
            response['total_features'] = 43 + len(option_features)  # 43 stock features + 27 option features
        else:
            response['option_features'] = None
            response['total_features'] = 43  # Just stock features

        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
def test_historical_volatility():
    """Test endpoint to verify historical volatility calculation"""
    try:
        # Fetch stock data
        stock_data = fetch_stock_data('AAPL')
        
        # Get historical DataFrame
        hist_df = stock_data['historical_prices']
        
        # Calculate historical volatility
        hv_metrics = calculate_historical_volatility(hist_df, windows=[30, 60, 90])
        
        if hv_metrics is None:
            return jsonify({
                'success': False,
                'error': 'Could not calculate historical volatility'
            }), 500
        
        # Calculate technical indicators for context
        enriched_df = calculate_technical_indicators(hist_df)
        latest = enriched_df.iloc[-1]
        
        # Build comprehensive response
        response = {
            'success': True,
            'ticker': 'AAPL',
            'current_price': float(stock_data['current_price']),
            'timestamp': datetime.now().isoformat(),
            
            # Historical Volatility Metrics
            'historical_volatility': {
                'HV_30': {
                    'value': round(hv_metrics['HV_30'], 4),
                    'percentage': f"{round(hv_metrics['HV_30'] * 100, 2)}%",
                    'description': '30-day annualized historical volatility'
                },
                'HV_60': {
                    'value': round(hv_metrics['HV_60'], 4),
                    'percentage': f"{round(hv_metrics['HV_60'] * 100, 2)}%",
                    'description': '60-day annualized historical volatility'
                },
                'HV_90': {
                    'value': round(hv_metrics['HV_90'], 4),
                    'percentage': f"{round(hv_metrics['HV_90'] * 100, 2)}%",
                    'description': '90-day annualized historical volatility'
                },
                'daily_volatility': {
                    'value': round(hv_metrics['daily_volatility'], 4),
                    'percentage': f"{round(hv_metrics['daily_volatility'] * 100, 2)}%",
                    'description': 'Current daily volatility (not annualized)'
                }
            },
            
            # Volatility Analysis
            'volatility_analysis': {
                'regime': {
                    'status': hv_metrics['HV_regime'],
                    'description': {
                        'expanding': 'Volatility is INCREASING - options getting more expensive',
                        'contracting': 'Volatility is DECREASING - options getting cheaper',
                        'stable': 'Volatility is STABLE - consistent pricing'
                    }[hv_metrics['HV_regime']]
                },
                'slope': {
                    'value': round(hv_metrics['HV_slope'], 4),
                    'percentage': f"{round(hv_metrics['HV_slope'] * 100, 2)}%",
                    'description': 'HV_30 vs HV_90 trend (positive = expanding)'
                },
                'acceleration': {
                    'value': round(hv_metrics['HV_acceleration'], 4),
                    'description': 'Rate of volatility change (positive = accelerating expansion)',
                    'interpretation': 'Explosive' if hv_metrics['HV_acceleration'] > 0.02 else 
                                    'Steady' if abs(hv_metrics['HV_acceleration']) <= 0.02 else 
                                    'Decelerating'
                },
                'percentile': {
                    'value': round(hv_metrics['HV_percentile'], 4),
                    'percentage': f"{round(hv_metrics['HV_percentile'] * 100, 1)}%",
                    'description': 'Where current HV ranks in 1-year range',
                    'interpretation': 'Extremely High' if hv_metrics['HV_percentile'] > 0.8 else
                                    'High' if hv_metrics['HV_percentile'] > 0.6 else
                                    'Average' if hv_metrics['HV_percentile'] > 0.4 else
                                    'Low' if hv_metrics['HV_percentile'] > 0.2 else
                                    'Extremely Low'
                },
                'z_score': {
                    'value': round(hv_metrics['HV_z_score'], 2),
                    'description': 'Standard deviations from 1-year mean HV',
                    'interpretation': 'Extreme' if abs(hv_metrics['HV_z_score']) > 2 else
                                    'Elevated' if abs(hv_metrics['HV_z_score']) > 1 else
                                    'Normal'
                }
            },
            
            # Expected Price Movement (based on HV_30)
            'expected_moves': {
                '30_day': {
                    'one_std_dev': round(stock_data['current_price'] * hv_metrics['HV_30'] * np.sqrt(30/252), 2),
                    'range_lower': round(stock_data['current_price'] * (1 - hv_metrics['HV_30'] * np.sqrt(30/252)), 2),
                    'range_upper': round(stock_data['current_price'] * (1 + hv_metrics['HV_30'] * np.sqrt(30/252)), 2),
                    'probability': '68%',
                    'description': '68% probability price stays within this range in 30 days'
                },
                '1_day': {
                    'one_std_dev': round(stock_data['current_price'] * hv_metrics['daily_volatility'], 2),
                    'range_lower': round(stock_data['current_price'] * (1 - hv_metrics['daily_volatility']), 2),
                    'range_upper': round(stock_data['current_price'] * (1 + hv_metrics['daily_volatility']), 2),
                    'probability': '68%',
                    'description': 'Expected daily price range'
                }
            },
            
            # Related Technical Indicators
            'related_indicators': {
                'ATR': {
                    'value': round(float(latest['ATR']), 2),
                    'percentage': round(float(latest['ATR_pct']), 2),
                    'description': 'Average True Range (actual daily price swings)'
                },
                'BB_width': {
                    'value': round(float(latest['BB_width']), 4),
                    'percentage': f"{round(float(latest['BB_width']) * 100, 2)}%",
                    'description': 'Bollinger Band width (another volatility measure)'
                }
            },
            
            # Trading Implications
            'options_strategy_hints': {
                'current_state': f"HV is at {round(hv_metrics['HV_percentile'] * 100, 1)}th percentile - " +
                                f"Volatility is {hv_metrics['HV_regime']}",
                'recommendation': 'Consider SELLING options (collect high premium)' if hv_metrics['HV_percentile'] > 0.7
                                else 'Consider BUYING options (cheaper premiums)' if hv_metrics['HV_percentile'] < 0.3
                                else 'Neutral - use directional strategies',
                'iv_expectation': 'IV likely elevated - options expensive' if hv_metrics['HV_regime'] == 'expanding'
                                else 'IV likely falling - options getting cheaper' if hv_metrics['HV_regime'] == 'contracting'
                                else 'IV likely stable'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test-option-features-mock', methods=['GET'])
def test_option_features_mock():
    """
    Test option features with MOCK data (for development when API is rate-limited)
    This endpoint uses simulated data to test the option feature calculations
    """
    try:
        # Get parameters
        strike_price = float(request.args.get('strike', 270))
        days_out = int(request.args.get('days', 30))
        option_type = request.args.get('type', 'call').lower()
        
        # Mock data
        current_price = 268.47
        ticker = 'AAPL'
        sector = 'Technology'
        
        # Create mock HV metrics
        hv_metrics = {
            'HV_30': 0.2019,
            'HV_60': 0.2273,
            'HV_90': 0.2332,
            'daily_volatility': 0.0127,
            'HV_slope': -0.134,
            'HV_regime': 'contracting',
            'HV_acceleration': -0.0194,
            'HV_percentile': 0.1327,
            'HV_z_score': -0.59
        }
        
        # Mock earnings data
        earnings_data = {
            'next_earnings_date': '2026-01-28',
            'days_until_earnings': 81
        }
        
        # Calculate expiration date
        expiration_date = (datetime.now() + timedelta(days=days_out)).strftime('%Y-%m-%d')
        
        # Calculate option features
        option_features = calculate_option_features(
            ticker=ticker,
            strike_price=strike_price,
            expiration_date=expiration_date,
            option_type=option_type,
            current_price=current_price,
            sector=sector,
            hv_metrics=hv_metrics,
            earnings_data=earnings_data,
            risk_free_rate=0.0411,
            vix=19.08
        )
        
        # Build response
        return jsonify({
            'success': True,
            'note': 'Using MOCK data for development',
            'ticker': ticker,
            'option': {
                'type': option_type,
                'strike': strike_price,
                'expiration': expiration_date,
                'dte': option_features['dte']
            },
            'market_data': {
                'current_price': current_price,
                'sector': sector,
                'vix': 19.08,
                'risk_free_rate': 4.11,
                'next_earnings': '2026-01-28',
                'days_to_earnings': 81
            },
            'option_features': {
                'moneyness': {
                    'moneyness_ratio': round(option_features['moneyness'], 4),
                    'status': 'ITM' if option_features['moneyness'] > 1.0 else 'OTM' if option_features['moneyness'] < 1.0 else 'ATM',
                    'itm_amount': round(option_features['itm_amount'], 2),
                    'itm_percent': round(option_features['itm_percent'] * 100, 2),
                    'otm_distance': round(option_features['otm_distance'] * 100, 2)
                },
                'time_decay': {
                    'dte': option_features['dte'],
                    'time_to_expiration': round(option_features['time_to_expiration'], 4),
                    'time_regime': ['short_term', 'medium_term', 'long_term'][option_features['time_regime']],
                    'theta_multiplier': round(option_features['theta_multiplier'], 4)
                },
                'earnings_impact': {
                    'days_to_earnings': option_features['days_to_earnings'],
                    'earnings_before_expiration': bool(option_features['earnings_before_expiration']),
                    'earnings_risk': round(option_features['earnings_risk'], 4)
                },
                'greeks': {
                    'delta': round(option_features['delta'], 4),
                    'gamma': round(option_features['gamma'], 6),
                    'vega': round(option_features['vega'], 4),
                    'theta': round(option_features['theta'], 4),
                    'rho': round(option_features['rho'], 4)
                },
                'volatility_context': {
                    'hv_percentile': round(option_features['hv_percentile'] * 100, 2),
                    'vol_skew': round(option_features['vol_skew'], 4),
                    'vix_ratio': round(option_features['vix_ratio'], 4)
                },
                'market_context': {
                    'sector_code': option_features['sector_code'],
                    'vix': option_features['vix'],
                    'vix_regime': option_features['vix_regime'],
                    'vix_regime_label': [
                        'Very Low (<14)', 'Low (14-15)', '15-16', '16-17', '17-18', 
                        '18-19', 'Normal (19-20)', '20-21', '21-22', '22-23', 
                        '23-24', 'Elevated (24-25)', 'High (>25)'
                    ][option_features['vix_regime']],
                    'risk_free_rate': round(option_features['risk_free_rate'] * 100, 2)
                }
            },
            'feature_count': len(option_features),
            'raw_features': option_features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/ml-features', methods=['GET'])
def ml_features():
    """
    Get COMPLETE ML feature set using REAL data from Polygon.io (70+ features)
    Returns: OHLCV + Technical Indicators + Historical Volatility + Option Features
    """
    try:
        # Get parameters from query
        ticker = request.args.get('ticker', 'AAPL').upper()
        strike_price = float(request.args.get('strike', 270))
        days_out = int(request.args.get('days', 30))
        option_type = request.args.get('type', 'call').lower()
        
        print(f"\n{'='*60}")
        print(f"Fetching ML features for {ticker} ${strike_price} {option_type.upper()} {days_out}DTE")
        print(f"{'='*60}")
        
        # Fetch REAL stock data using Polygon (through existing function)
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data['current_price']
        historical_prices = stock_data['historical_prices']
        sector = stock_data['sector']
        
        print(f"✓ Fetched {len(historical_prices)} days of real stock data")
        print(f"  Current price: ${current_price:.2f}")
        print(f"  Sector: {sector}")
        
        # Calculate expiration
        expiration_date = (datetime.now() + timedelta(days=days_out)).strftime('%Y-%m-%d')
        
        # Calculate ALL feature sets using REAL data
        technical_indicators = calculate_technical_indicators(historical_prices)
        hv_metrics = calculate_historical_volatility(historical_prices)
        
        # Prepare HV data for option features
        hv_data = {
            'HV_30': hv_metrics['HV_30'],
            'HV_60': hv_metrics['HV_60'],
            'HV_90': hv_metrics['HV_90'],
            'daily_volatility': hv_metrics['daily_volatility'],
            'HV_slope': hv_metrics['HV_slope'],
            'HV_regime': hv_metrics['HV_regime'],
            'HV_acceleration': hv_metrics['HV_acceleration'],
            'HV_percentile': hv_metrics['HV_percentile'],
            'HV_z_score': hv_metrics['HV_z_score']
        }
        
        # Get real VIX and risk-free rate
        vix = fetch_vix()
        risk_free_rate = fetch_risk_free_rate()
        
        # Mock earnings (in production, fetch real earnings dates)
        earnings_data = {
            'next_earnings_date': '2026-01-28',
            'days_until_earnings': 81
        }
        
        option_features = calculate_option_features(
            ticker=ticker,
            strike_price=strike_price,
            expiration_date=expiration_date,
            option_type=option_type,
            current_price=current_price,
            sector=sector,
            hv_metrics=hv_data,
            earnings_data=earnings_data,
            risk_free_rate=risk_free_rate,
            vix=vix
        )
        
        # Build complete feature vector for ML
        ml_features = {}
        
        # OHLCV (5 features) - REAL data
        latest = historical_prices.iloc[-1]
        ml_features.update({
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'close': float(latest['Close']),
            'volume': int(latest['Volume'])
        })
        
        # Technical Indicators (29 features) - calculated from REAL data
        if not technical_indicators.empty:
            latest_indicators = technical_indicators.iloc[-1]
            for col in technical_indicators.columns:
                ml_features[col.lower()] = float(latest_indicators[col])
        
        # Historical Volatility (9 features) - calculated from REAL data
        ml_features.update({
            'hv_30': hv_metrics['HV_30'],
            'hv_60': hv_metrics['HV_60'],
            'hv_90': hv_metrics['HV_90'],
            'daily_volatility': hv_metrics['daily_volatility'],
            'hv_slope': hv_metrics['HV_slope'],
            'hv_regime': 1 if hv_metrics['HV_regime'] == 'expanding' else 0,
            'hv_acceleration': hv_metrics['HV_acceleration'],
            'hv_percentile': hv_metrics['HV_percentile'],
            'hv_z_score': hv_metrics['HV_z_score']
        })
        
        # Option Features (27 features) - calculated from REAL data
        ml_features.update(option_features)
        
        print(f"✓ Calculated {len(ml_features)} ML features from REAL market data")
        
        return jsonify({
            'success': True,
            'note': 'REAL DATA from Polygon.io - Complete ML feature set',
            'ticker': ticker,
            'data_source': 'Polygon.io API',
            'feature_breakdown': {
                'ohlcv': 5,
                'technical_indicators': 29,
                'historical_volatility': 9,
                'option_features': 27,
                'total_features': len(ml_features)
            },
            'ml_features': ml_features,
            'feature_names': sorted(ml_features.keys()),
            'option_params': {
                'strike': strike_price,
                'expiration': expiration_date,
                'dte': option_features['dte'],
                'type': option_type
            },
            'market_context': {
                'current_price': current_price,
                'sector': sector,
                'vix': vix,
                'risk_free_rate': risk_free_rate,
                'historical_days': len(historical_prices)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/test-option-features', methods=['GET'])
def test_option_features():
    """
    Test option-specific feature calculation
    Example: AAPL $270 Call expiring 30 days out
    
    Query params (optional):
        - ticker: Stock symbol (default: AAPL)
        - strike: Strike price (default: 270)
        - days: Days to expiration (default: 30)
        - type: call or put (default: call)
    """
    try:
        # Get parameters from query string or use defaults
        ticker = request.args.get('ticker', 'AAPL').upper()
        strike_price = float(request.args.get('strike', 270))
        days_out = int(request.args.get('days', 30))
        option_type = request.args.get('type', 'call').lower()
        
        # Validate option type
        if option_type not in ['call', 'put']:
            return jsonify({
                'success': False,
                'error': 'Option type must be "call" or "put"'
            }), 400
        
        # Fetch market data
        market_data = fetch_all_market_data(ticker)
        
        # Check if we have required data
        if not market_data.get('stock_data'):
            return jsonify({
                'success': False,
                'error': 'Could not fetch stock data',
                'details': market_data.get('errors', [])
            }), 500
        
        # Calculate technical indicators
        df = market_data['stock_data']['historical_prices'].copy()
        enriched_df = calculate_technical_indicators(df)
        
        # Calculate HV
        hv_metrics = calculate_historical_volatility(enriched_df)
        if not hv_metrics:
            return jsonify({
                'success': False,
                'error': 'Could not calculate historical volatility'
            }), 500
        
        # Calculate expiration date
        expiration_date = (datetime.now() + timedelta(days=days_out)).strftime('%Y-%m-%d')
        
        # Get current price
        current_price = market_data['stock_data']['current_price']
        
        # Calculate option features
        option_features = calculate_option_features(
            ticker=ticker,
            strike_price=strike_price,
            expiration_date=expiration_date,
            option_type=option_type,
            current_price=current_price,
            sector=market_data['stock_data']['sector'],
            hv_metrics=hv_metrics,
            earnings_data=market_data['earnings'],
            risk_free_rate=market_data['risk_free_rate'],
            vix=market_data['vix']
        )
        
        # Build response with interpretations
        return jsonify({
            'success': True,
            'ticker': ticker,
            'option': {
                'type': option_type,
                'strike': strike_price,
                'expiration': expiration_date,
                'dte': option_features['dte']
            },
            'market_data': {
                'current_price': current_price,
                'sector': market_data['stock_data']['sector'],
                'vix': market_data['vix'],
                'risk_free_rate': round(market_data['risk_free_rate'] * 100, 2),
                'next_earnings': market_data['earnings']['next_earnings_date'],
                'days_to_earnings': market_data['earnings']['days_until_earnings']
            },
            'option_features': {
                'moneyness': {
                    'moneyness_ratio': round(option_features['moneyness'], 4),
                    'status': 'ITM' if option_features['moneyness'] > 1.0 else 'OTM' if option_features['moneyness'] < 1.0 else 'ATM',
                    'itm_amount': round(option_features['itm_amount'], 2),
                    'itm_percent': round(option_features['itm_percent'] * 100, 2),
                    'otm_distance': round(option_features['otm_distance'] * 100, 2)
                },
                'time_decay': {
                    'dte': option_features['dte'],
                    'time_to_expiration': round(option_features['time_to_expiration'], 4),
                    'time_regime': ['short_term', 'medium_term', 'long_term'][option_features['time_regime']],
                    'theta_multiplier': round(option_features['theta_multiplier'], 4)
                },
                'earnings_impact': {
                    'days_to_earnings': option_features['days_to_earnings'],
                    'earnings_before_expiration': bool(option_features['earnings_before_expiration']),
                    'earnings_risk': round(option_features['earnings_risk'], 4)
                },
                'greeks': {
                    'delta': round(option_features['delta'], 4),
                    'gamma': round(option_features['gamma'], 6),
                    'vega': round(option_features['vega'], 4),
                    'theta': round(option_features['theta'], 4),
                    'rho': round(option_features['rho'], 4)
                },
                'volatility_context': {
                    'hv_percentile': round(option_features['hv_percentile'] * 100, 2),
                    'vol_skew': round(option_features['vol_skew'], 4),
                    'vix_ratio': round(option_features['vix_ratio'], 4)
                },
                'market_context': {
                    'sector_code': option_features['sector_code'],
                    'vix': option_features['vix'],
                    'vix_regime': option_features['vix_regime'],
                    'vix_regime_label': [
                        'Very Low (<14)', 'Low (14-15)', '15-16', '16-17', '17-18', 
                        '18-19', 'Normal (19-20)', '20-21', '21-22', '22-23', 
                        '23-24', 'Elevated (24-25)', 'High (>25)'
                    ][option_features['vix_regime']],
                    'risk_free_rate': round(option_features['risk_free_rate'] * 100, 2)
                }
            },
            'feature_count': len(option_features),
            'raw_features': option_features  # For ML model input
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test-normalization', methods=['GET'])
def test_normalization():
    """
    Test feature normalization pipeline with mock data
    Demonstrates: raw features → normalized features → ready for ML
    """
    try:
        # Get raw features from mock endpoint
        strike = float(request.args.get('strike', 270))
        days = int(request.args.get('days', 30))
        option_type = request.args.get('type', 'call')
        
        # Simulate getting features from ml-features-mock endpoint
        # In production, you'd call the endpoint, here we'll generate mock data
        
        # Create mock raw features (this would come from /api/ml-features-mock)
        mock_raw_features = {
            'open': 268.47, 'high': 270.0, 'low': 267.0, 'close': 268.47,
            'volume': 76796693, 'rsi': 50.5, 'macd': 4.68, 'sma_10': 276.97,
            'sma_20': 272.65, 'sma_50': 253.94, 'ema_10': 272.95, 'ema_20': 270.68,
            'stoch_k': 7.48, 'stoch_d': 18.29, 'bb_upper': 296.93, 'bb_middle': 272.65,
            'bb_lower': 248.36, 'bb_width': 0.178, 'bb_position': 0.414, 'atr': 6.43,
            'atr_pct': 2.40, 'adx': 17.21, 'momentum_10': -16.19, 'roc_10': -5.69,
            'returns_1d': 0.001, 'returns_5d': -0.025, 'returns_10d': -0.057,
            'volume_ratio': 0.85, 'volume_change': -0.008, 'volume_ma_20': 90292490,
            'price_vs_sma10': -3.07, 'price_vs_sma20': -1.53, 'macd_signal': 7.23,
            'macd_histogram': -2.55, 'hv_30': 0.324, 'hv_60': 0.332, 'hv_90': 0.337,
            'daily_volatility': 0.021, 'hv_slope': -0.039, 'hv_regime': 0,
            'hv_acceleration': -0.003, 'hv_percentile': 0.387, 'hv_z_score': -0.540,
            'moneyness': 0.994, 'itm_amount': 0, 'itm_percent': 0, 'otm_distance': 0.006,
            'dte': days, 'time_to_expiration': days/365, 'time_regime': 0,
            'theta_multiplier': 0.186, 'days_to_earnings': 81, 'earnings_before_expiration': 0,
            'earnings_risk': 0.0, 'delta': 0.508, 'gamma': 0.016, 'vega': 0.302,
            'theta': -61.48, 'rho': 0.101, 'vol_skew': -0.039, 'vix_ratio': 0.954,
            'vix': 19.08, 'vix_regime': 6, 'risk_free_rate': 0.0411, 'sector_code': 0,
            'is_call': 1 if option_type == 'call' else 0
        }
        
        # Step 1: Prepare feature matrix (convert dict → DataFrame)
        features_df = prepare_feature_matrix(mock_raw_features, return_dataframe=True)
        
        # Step 2: Check if scalers exist
        scaler_path = 'backend/scalers/feature_scalers.pkl'
        if not os.path.exists(scaler_path):
            # Create mock training data to fit scalers (first time only)
            print("Creating scalers for first time...")
            mock_training = []
            for i in range(100):
                # Generate 100 variations of features
                variation = mock_raw_features.copy()
                variation['close'] = variation['close'] * np.random.uniform(0.95, 1.05)
                variation['rsi'] = np.random.uniform(30, 70)
                variation['delta'] = np.random.uniform(0.3, 0.7)
                mock_training.append(variation)
            
            training_df = prepare_feature_matrix(mock_training)
            scaler_dict = create_feature_scaler(training_df, save_path=scaler_path)
        else:
            # Load existing scalers
            scaler_dict = load_feature_scaler(scaler_path)
        
        # Step 3: Normalize features
        normalized_df = normalize_features(features_df, scaler_dict, return_dataframe=True)
        
        # Step 4: Convert to formats for different models
        normalized_array = normalized_df.values  # For numpy-based models
        normalized_dict = normalized_df.iloc[0].to_dict()  # For inspection
        
        return jsonify({
            'success': True,
            'note': 'Feature normalization pipeline test',
            'raw_features': {
                'shape': list(features_df.shape),
                'sample': features_df.iloc[0].to_dict(),
                'stats': {
                    'min_value': float(features_df.min().min()),
                    'max_value': float(features_df.max().max()),
                    'mean_value': float(features_df.mean().mean())
                }
            },
            'normalized_features': {
                'shape': list(normalized_df.shape),
                'sample': normalized_dict,
                'stats': {
                    'min_value': float(normalized_df.min().min()),
                    'max_value': float(normalized_df.max().max()),
                    'mean_value': float(normalized_df.mean().mean())
                }
            },
            'ready_for_ml': {
                'dataframe_shape': list(normalized_df.shape),
                'numpy_array_shape': list(normalized_array.shape),
                'feature_count': len(normalized_dict),
                'feature_names': list(normalized_df.columns)
            },
            'scaler_info': {
                'path': scaler_path,
                'created_at': scaler_dict.get('created_at'),
                'trained_on_samples': scaler_dict.get('n_samples'),
                'minmax_features': len(scaler_dict['feature_groups']['minmax_features']),
                'standard_features': len(scaler_dict['feature_groups']['standard_features']),
                'categorical_features': len(scaler_dict['feature_groups']['categorical_features'])
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/predict-iv', methods=['GET'])
def predict_iv():
    """
    Predict implied volatility using trained LightGBM model
    
    Query params:
        - ticker: Stock symbol (e.g., AAPL)
        - strike: Strike price (e.g., 270)
        - days: Days to expiration (e.g., 30)
        - type: call or put
    
    Returns:
        Predicted IV with confidence interval
    """
    try:
        # Get parameters
        ticker = request.args.get('ticker', 'AAPL').upper()
        strike_price = float(request.args.get('strike', 270))
        days_out = int(request.args.get('days', 30))
        option_type = request.args.get('type', 'call').lower()
        
        print(f"\n{'='*60}")
        print(f"IV Prediction Request: {ticker} ${strike_price} {option_type.upper()} {days_out}DTE")
        print(f"{'='*60}")
        
        
        model_path = 'backend/models/lightgbm_iv_predictor.pkl'
        scaler_path = 'backend/scalers/feature_scalers.pkl'
        metadata_path = 'backend/models/lightgbm_metadata.pkl'
        
        if not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'error': 'Model not found. Run train_lightgbm.py first'
            }), 500
        
        print("Loading model and scaler...")
        model = joblib.load(model_path)
        scaler_dict = joblib.load(scaler_path)
        metadata = joblib.load(metadata_path)
        
        # Get features from /api/ml-features endpoint logic
        # (Reuse the same feature calculation)
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data['current_price']
        historical_prices = stock_data['historical_prices']
        sector = stock_data['sector']
        
        print(f"✓ Stock data fetched: ${current_price:.2f}")
        
        # Calculate expiration
        expiration_date = (datetime.now() + timedelta(days=days_out)).strftime('%Y-%m-%d')
        
        # Calculate ALL feature sets
        technical_indicators = calculate_technical_indicators(historical_prices)
        hv_metrics = calculate_historical_volatility(historical_prices)
        
        # Prepare HV data
        hv_data = {
            'HV_30': hv_metrics['HV_30'],
            'HV_60': hv_metrics['HV_60'],
            'HV_90': hv_metrics['HV_90'],
            'daily_volatility': hv_metrics['daily_volatility'],
            'HV_slope': hv_metrics['HV_slope'],
            'HV_regime': hv_metrics['HV_regime'],
            'HV_acceleration': hv_metrics['HV_acceleration'],
            'HV_percentile': hv_metrics['HV_percentile'],
            'HV_z_score': hv_metrics['HV_z_score']
        }
        
        # Get real VIX and risk-free rate
        vix = fetch_vix()
        risk_free_rate = fetch_risk_free_rate()
        
        # Mock earnings
        earnings_data = {
            'next_earnings_date': '2026-01-28',
            'days_until_earnings': 81
        }
        
        option_features = calculate_option_features(
            ticker=ticker,
            strike_price=strike_price,
            expiration_date=expiration_date,
            option_type=option_type,
            current_price=current_price,
            sector=sector,
            hv_metrics=hv_data,
            earnings_data=earnings_data,
            risk_free_rate=risk_free_rate,
            vix=vix
        )
        
        # Build feature vector (same as training)
        ml_features = {}
        
        # OHLCV
        latest = historical_prices.iloc[-1]
        ml_features.update({
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'close': float(latest['Close']),
            'volume': int(latest['Volume'])
        })
        
        # Technical Indicators
        if not technical_indicators.empty:
            latest_indicators = technical_indicators.iloc[-1]
            for col in technical_indicators.columns:
                ml_features[col.lower()] = float(latest_indicators[col])
        
        # Historical Volatility
        ml_features.update({
            'hv_30': hv_metrics['HV_30'],
            'hv_60': hv_metrics['HV_60'],
            'hv_90': hv_metrics['HV_90'],
            'daily_volatility': hv_metrics['daily_volatility'],
            'hv_slope': hv_metrics['HV_slope'],
            'hv_regime': 1 if hv_metrics['HV_regime'] == 'expanding' else 0,
            'hv_acceleration': hv_metrics['HV_acceleration'],
            'hv_percentile': hv_metrics['HV_percentile'],
            'hv_z_score': hv_metrics['HV_z_score']
        })
        
        # Option Features
        ml_features.update(option_features)
        
        print(f"✓ Calculated {len(ml_features)} features")
        
        # Normalize features
        feature_df = prepare_feature_matrix(ml_features, return_dataframe=True)
        normalized_features = normalize_features(feature_df, scaler_dict, return_dataframe=False)
        
        print(f"✓ Features normalized")
        
        # Make prediction
        predicted_iv = model.predict(normalized_features)[0]
        
        print(f"✓ Predicted IV: {predicted_iv:.4f} ({predicted_iv*100:.2f}%)")
        
        # Calculate confidence interval (using test MAE from metadata)
        test_mae = metadata.get('test_mae', 0.0233)
        confidence_lower = max(0.05, predicted_iv - test_mae)
        confidence_upper = min(1.5, predicted_iv + test_mae)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'option': {
                'strike': strike_price,
                'expiration': expiration_date,
                'dte': days_out,
                'type': option_type
            },
            'prediction': {
                'implied_volatility': float(predicted_iv),
                'iv_percentage': float(predicted_iv * 100),
                'confidence_interval': {
                    'lower': float(confidence_lower),
                    'upper': float(confidence_upper),
                    'lower_pct': float(confidence_lower * 100),
                    'upper_pct': float(confidence_upper * 100)
                }
            },
            'market_context': {
                'current_price': current_price,
                'historical_volatility_30d': hv_metrics['HV_30'],
                'vix': vix,
                'moneyness': strike_price / current_price
            },
            'model_info': {
                'model_type': 'LightGBM',
                'test_mae': test_mae,
                'test_mape': metadata.get('test_mape', 0.0),
                'features_used': len(ml_features)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/analyze-option', methods=['GET'])
def analyze_option():
    """
    Comprehensive option analysis combining IV prediction and price forecasting
    
    Query params:
        - ticker: Stock symbol (e.g., AAPL)
        - strike: Strike price (e.g., 270)
        - days: Days to expiration (e.g., 30)
        - type: call or put
    
    Returns:
        Complete analysis with IV prediction, price forecast, Greeks, and scenarios
    """
    try:
        # Get parameters
        ticker = request.args.get('ticker', 'AAPL').upper()
        strike_price = float(request.args.get('strike', 270))
        days_out = int(request.args.get('days', 30))
        option_type = request.args.get('type', 'call').lower()
        
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE OPTION ANALYSIS")
        print(f"  {ticker} ${strike_price} {option_type.upper()} {days_out}DTE")
        print(f"{'='*70}")
    
        
        # Helper function to convert numpy types to Python types
        def convert_to_python_type(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python_type(item) for item in obj]
            return obj
        
        # Load LightGBM (IV prediction)
        lgbm_model_path = 'backend/models/lightgbm_iv_predictor.pkl'
        lgbm_metadata_path = 'backend/models/lightgbm_metadata.pkl'
        
        # Load LSTM (price prediction)
        lstm_model_path = 'backend/models/lstm_price_predictor.h5'
        lstm_metadata_path = 'backend/models/lstm_metadata.pkl'
        
        # Load scaler
        scaler_path = 'backend/scalers/feature_scalers.pkl'
        
        print("\n[1] Loading models...")
        lgbm_model = joblib.load(lgbm_model_path)
        lgbm_metadata = joblib.load(lgbm_metadata_path)
        lstm_model = keras.models.load_model(lstm_model_path)
        lstm_metadata = joblib.load(lstm_metadata_path)
        scaler_dict = joblib.load(scaler_path)
        print("  ✓ LightGBM, LSTM, and scaler loaded")
        
        # [2] Fetch Market Data

        print("\n[2] Fetching market data...")
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data['current_price']
        historical_prices = stock_data['historical_prices']
        sector = stock_data['sector']
        
        print(f"  ✓ Current price: ${current_price:.2f}")
        print(f"  ✓ Historical data: {len(historical_prices)} days")
        
        # Calculate expiration
        expiration_date = (datetime.now() + timedelta(days=days_out)).strftime('%Y-%m-%d')
        

        # [3] Calculate Features

        print("\n[3] Calculating features...")
        
        # Technical indicators
        technical_indicators = calculate_technical_indicators(historical_prices)
        
        # Historical volatility
        hv_metrics = calculate_historical_volatility(historical_prices)
        
        # VIX and risk-free rate
        vix = fetch_vix()
        risk_free_rate = fetch_risk_free_rate()
        
        # Mock earnings data
        earnings_data = {
            'next_earnings_date': '2026-01-28',
            'days_until_earnings': 81
        }
        
        # Prepare HV data structure
        hv_data = {
            'HV_30': hv_metrics['HV_30'],
            'HV_60': hv_metrics['HV_60'],
            'HV_90': hv_metrics['HV_90'],
            'daily_volatility': hv_metrics['daily_volatility'],
            'HV_slope': hv_metrics['HV_slope'],
            'HV_regime': hv_metrics['HV_regime'],
            'HV_acceleration': hv_metrics['HV_acceleration'],
            'HV_percentile': hv_metrics['HV_percentile'],
            'HV_z_score': hv_metrics['HV_z_score']
        }
        
        # Option-specific features
        option_features = calculate_option_features(
            ticker=ticker,
            strike_price=strike_price,
            expiration_date=expiration_date,
            option_type=option_type,
            current_price=current_price,
            sector=sector,
            hv_metrics=hv_data,
            earnings_data=earnings_data,
            risk_free_rate=risk_free_rate,
            vix=vix
        )
        
        # Build complete feature vector
        ml_features = {}
        
        # OHLCV
        latest = historical_prices.iloc[-1]
        ml_features.update({
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'close': float(latest['Close']),
            'volume': int(latest['Volume'])
        })
        
        # Technical Indicators
        if not technical_indicators.empty:
            latest_indicators = technical_indicators.iloc[-1]
            for col in technical_indicators.columns:
                ml_features[col.lower()] = float(latest_indicators[col])
        
        # Historical Volatility
        ml_features.update({
            'hv_30': hv_metrics['HV_30'],
            'hv_60': hv_metrics['HV_60'],
            'hv_90': hv_metrics['HV_90'],
            'daily_volatility': hv_metrics['daily_volatility'],
            'hv_slope': hv_metrics['HV_slope'],
            'hv_regime': 1 if hv_metrics['HV_regime'] == 'expanding' else 0,
            'hv_acceleration': hv_metrics['HV_acceleration'],
            'hv_percentile': hv_metrics['HV_percentile'],
            'hv_z_score': hv_metrics['HV_z_score']
        })
        
        # Option Features
        ml_features.update(option_features)
        
        print(f"  ✓ Calculated {len(ml_features)} features")
        

        # [4] Predict IV (LightGBM)

        print("\n[4] Predicting implied volatility...")
        
        feature_df = prepare_feature_matrix(ml_features, return_dataframe=True)
        normalized_single = normalize_features(feature_df, scaler_dict, return_dataframe=False)
        
        predicted_iv = lgbm_model.predict(normalized_single)[0]
        lgbm_mae = lgbm_metadata.get('test_mae', 0.0221)
        
        iv_confidence = {
            'lower': max(0.05, predicted_iv - lgbm_mae),
            'upper': min(1.5, predicted_iv + lgbm_mae)
        }
        
        print(f"  ✓ Predicted IV: {predicted_iv:.4f} ({predicted_iv*100:.2f}%)")
        

        # [4.5] Calculate Black-Scholes Pricing & Greeks

        print("\n[4.5] Calculating Black-Scholes pricing...")
        
        time_to_expiration = days_out / 365.0
        
        # Theoretical price using predicted IV
        theoretical_price = black_scholes_price(
            S=current_price,
            K=strike_price,
            T=time_to_expiration,
            r=risk_free_rate,
            sigma=predicted_iv,
            option_type=option_type
        )
        
        # Theoretical price using historical volatility (for comparison)
        theoretical_price_hv = black_scholes_price(
            S=current_price,
            K=strike_price,
            T=time_to_expiration,
            r=risk_free_rate,
            sigma=hv_metrics['HV_30'],
            option_type=option_type
        )
        
        # Calculate Greeks using predicted IV
        greeks = calculate_greeks(
            S=current_price,
            K=strike_price,
            T=time_to_expiration,
            r=risk_free_rate,
            sigma=predicted_iv,
            option_type=option_type
        )
        
        print(f"  ✓ Theoretical price (predicted IV): ${theoretical_price:.2f}")
        print(f"  ✓ Theoretical price (HV): ${theoretical_price_hv:.2f}")
        print(f"  ✓ Greeks: Δ={greeks['delta']:.3f}, Γ={greeks['gamma']:.4f}, Θ={greeks['theta']:.4f}")
        
        iv_confidence = {
            'lower': max(0.05, predicted_iv - lgbm_mae),
            'upper': min(1.5, predicted_iv + lgbm_mae)
        }
        
        print(f"  ✓ Predicted IV: {predicted_iv:.4f} ({predicted_iv*100:.2f}%)")
        

        # [5] Predict Price Movement (LSTM)

        print("\n[5] Predicting price movement...")
        
        # Get last 30 days of features for LSTM sequence
        sequence_length = lstm_metadata['sequence_length']
        
        if len(historical_prices) < sequence_length:
            print(f"  ⚠ Not enough historical data for LSTM ({len(historical_prices)} < {sequence_length})")
            price_prediction = None
        else:
            # Calculate features for last 30 days
            sequence_features = []
            
            for i in range(-sequence_length, 0):
                day_features = {}
                day_data = historical_prices.iloc[i]
                
                # OHLCV
                day_features.update({
                    'open': float(day_data['Open']),
                    'high': float(day_data['High']),
                    'low': float(day_data['Low']),
                    'close': float(day_data['Close']),
                    'volume': int(day_data['Volume'])
                })
                
                # Technical indicators for this day
                if i + len(technical_indicators) >= 0:
                    tech_idx = i + len(technical_indicators)
                    if 0 <= tech_idx < len(technical_indicators):
                        day_tech = technical_indicators.iloc[tech_idx]
                        for col in technical_indicators.columns:
                            day_features[col.lower()] = float(day_tech[col])
                
                # Use same HV and option features for all days (simplified)
                day_features.update({
                    'hv_30': hv_metrics['HV_30'],
                    'hv_60': hv_metrics['HV_60'],
                    'hv_90': hv_metrics['HV_90'],
                    'daily_volatility': hv_metrics['daily_volatility'],
                    'hv_slope': hv_metrics['HV_slope'],
                    'hv_regime': 1 if hv_metrics['HV_regime'] == 'expanding' else 0,
                    'hv_acceleration': hv_metrics['HV_acceleration'],
                    'hv_percentile': hv_metrics['HV_percentile'],
                    'hv_z_score': hv_metrics['HV_z_score']
                })
                day_features.update(option_features)
                
                sequence_features.append(day_features)
            
            # Convert to DataFrame and normalize
            sequence_df = pd.DataFrame(sequence_features)
            sequence_normalized = normalize_features(sequence_df, scaler_dict, return_dataframe=False)
            
            # Reshape for LSTM: (1, sequence_length, n_features)
            sequence_input = sequence_normalized.reshape(1, sequence_length, -1)
            
            # Predict
            predicted_return = lstm_model.predict(sequence_input, verbose=0)[0][0]
            predicted_price = current_price * (1 + predicted_return)
            
            lstm_mae = lstm_metadata.get('test_mae', 0.0272)
            direction_accuracy = lstm_metadata.get('direction_accuracy', 0.6465)
            
            price_prediction = {
                'predicted_return': float(predicted_return),
                'predicted_price': float(predicted_price),
                'confidence_interval': {
                    'lower': float(current_price * (1 + predicted_return - lstm_mae)),
                    'upper': float(current_price * (1 + predicted_return + lstm_mae))
                },
                'direction_probability': float(direction_accuracy),
                'model_mae': float(lstm_mae)
            }
            
            print(f"  ✓ Predicted return: {predicted_return*100:.2f}%")
            print(f"  ✓ Predicted price: ${predicted_price:.2f}")
        

        # [6] Calculate Option Scenarios

        print("\n[6] Calculating scenarios...")
        
        moneyness = strike_price / current_price
        
        # Scenario analysis
        scenarios = {
            'bull_case': {
                'stock_price': current_price * 1.10,
                'probability': 0.30
            },
            'base_case': {
                'stock_price': current_price * (1 + (predicted_return if price_prediction else 0)),
                'probability': 0.40
            },
            'bear_case': {
                'stock_price': current_price * 0.90,
                'probability': 0.30
            }
        }
        
        # Calculate intrinsic value for each scenario
        for scenario_name, scenario in scenarios.items():
            price = scenario['stock_price']
            if option_type == 'call':
                intrinsic = max(0, price - strike_price)
            else:
                intrinsic = max(0, strike_price - price)
            
            scenario['intrinsic_value'] = float(intrinsic)
            scenario['in_the_money'] = intrinsic > 0
        

        # [7] Build Response

        print("\n[7] Building response...")
        
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            
            # Option Details
            'option': {
                'ticker': ticker,
                'strike': strike_price,
                'expiration': expiration_date,
                'dte': days_out,
                'type': option_type,
                'moneyness': float(moneyness)
            },
            
            # Market Data
            'market': {
                'current_price': float(current_price),
                'sector': sector,
                'vix': float(vix),
                'risk_free_rate': float(risk_free_rate),
                'historical_volatility': {
                    'hv_30d': float(hv_metrics['HV_30']),
                    'hv_60d': float(hv_metrics['HV_60']),
                    'hv_90d': float(hv_metrics['HV_90']),
                    'regime': hv_metrics['HV_regime']
                }
            },
            
            # Black-Scholes Option Pricing
            'option_pricing': {
                'theoretical_price': float(theoretical_price),
                'theoretical_price_hv': float(theoretical_price_hv),
                'greeks': {
                    'delta': float(greeks['delta']),
                    'gamma': float(greeks['gamma']),
                    'theta': float(greeks['theta']),
                    'vega': float(greeks['vega']),
                    'rho': float(greeks['rho'])
                }
            },
            
            # IV Prediction
            'iv_prediction': {
                'predicted_iv': float(predicted_iv),
                'confidence_interval': {
                    'lower': float(iv_confidence['lower']),
                    'upper': float(iv_confidence['upper'])
                },
                'vs_historical_vol': {
                    'difference': float(predicted_iv - hv_metrics['HV_30'])
                },
                'model': {
                    'type': 'LightGBM',
                    'test_mae': float(lgbm_metadata.get('test_mae', 0.0)),
                    'test_r2': float(lgbm_metadata.get('test_r2', 0.0)),
                    'test_mape': float(lgbm_metadata.get('test_mape', 0.0))
                }
            },
            
            # Price Prediction
            'price_prediction': price_prediction,
            
            # Scenarios
            'scenarios': scenarios
        }
        
        print(f"\n✓ Analysis complete!")
        print(f"{'='*70}\n")
        
        # Convert all numpy types to Python native types before returning
        response = convert_to_python_type(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"\n❌ Error in analysis: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)




