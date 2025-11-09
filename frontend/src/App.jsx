import React, { useState } from 'react';
import { TrendingUp, TrendingDown, AlertCircle, Calendar, DollarSign, Activity, BarChart3, LineChart, Info } from 'lucide-react';

// Tooltip Component
const Tooltip = ({ text, position = 'right' }) => {
  const [show, setShow] = useState(false);
  
  const positionClasses = {
    right: '-right-2 top-6',
    left: '-left-2 top-6',
    top: 'bottom-6 left-1/2 -translate-x-1/2',
  };
  
  const arrowClasses = {
    right: '-top-2 right-3 w-3 h-3 bg-gray-900 transform rotate-45',
    left: '-top-2 left-3 w-3 h-3 bg-gray-900 transform rotate-45',
    top: '-bottom-2 left-1/2 -translate-x-1/2 w-3 h-3 bg-gray-900 transform rotate-45',
  };
  
  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="cursor-help"
      >
        <Info size={16} className="text-blue-500" />
      </div>
      {show && (
        <div className={`absolute z-50 w-64 p-3 bg-gray-900 text-white text-sm rounded-lg shadow-lg ${positionClasses[position]}`}>
          <div className={arrowClasses[position]}></div>
          {text}
        </div>
      )}
    </div>
  );
};

const OptionsAnalyzer = () => {
  const [formData, setFormData] = useState({
    ticker: 'AAPL',
    strike: '270',
    optionType: 'call',
    expiration: '2025-12-08',
    premium: '5.50'
  });
  
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Sample data for demo - replace with actual API call
  const sampleData = {
    "iv_prediction": {
      "confidence_interval": {
        "lower": 0.1669156926485391,
        "upper": 0.21101716516719413
      },
      "model": {
        "test_mae": 0.02205073625932753,
        "test_mape": 9.415652509394764,
        "test_r2": 0.9391130626824851,
        "type": "LightGBM"
      },
      "predicted_iv": 0.1889664289078666,
      "vs_historical_vol": {
        "difference": -0.0129724913607622
      }
    },
    "market": {
      "current_price": 268.47,
      "historical_volatility": {
        "hv_30d": 0.2019389202686288,
        "hv_60d": 0.22725505594019935,
        "hv_90d": 0.23317848924615137,
        "regime": "contracting"
      },
      "risk_free_rate": 0.041100000000000005,
      "sector": "ELECTRONIC COMPUTERS",
      "vix": 19.08
    },
    "option": {
      "dte": 30,
      "expiration": "2025-12-08",
      "moneyness": 1.0056989607777405,
      "strike": 270.0,
      "ticker": "AAPL",
      "type": "call"
    },
    "option_pricing": {
      "greeks": {
        "delta": 0.4938348751992371,
        "gamma": 0.027426070601833077,
        "rho": 0.10444601052622157,
        "theta": -0.11100355775104619,
        "vega": 0.30702105617743974
      },
      "theoretical_price": 5.50386947116958,
      "theoretical_price_hv": 5.902165163346638
    },
    "price_prediction": {
      "confidence_interval": {
        "lower": 265.7668151855469,
        "upper": 280.3976745605469
      },
      "direction_probability": 0.6464646464646465,
      "model_mae": 0.02724858566049538,
      "predicted_price": 273.0822448730469,
      "predicted_return": 0.017179684713482857
    },
    "scenarios": {
      "base_case": {
        "in_the_money": true,
        "intrinsic_value": 3.082244873046875,
        "probability": 0.4,
        "stock_price": 273.0822448730469
      },
      "bear_case": {
        "in_the_money": false,
        "intrinsic_value": 0.0,
        "probability": 0.3,
        "stock_price": 241.62300000000002
      },
      "bull_case": {
        "in_the_money": true,
        "intrinsic_value": 25.317000000000064,
        "probability": 0.3,
        "stock_price": 295.31700000000006
      }
    },
    "success": true,
    "timestamp": "2025-11-08T21:32:34.167476"
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Calculate days to expiration
      const expirationDate = new Date(formData.expiration);
      const today = new Date();
      const days = Math.ceil((expirationDate - today) / (1000 * 60 * 60 * 24));
      
      const response = await fetch(`http://localhost:5000/api/analyze-option?ticker=${formData.ticker}&strike=${formData.strike}&days=${days}&type=${formData.optionType}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to fetch analysis. Make sure the backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;
  const formatCurrency = (value) => `$${value.toFixed(2)}`;
  const formatGreek = (value) => value.toFixed(4);

  const getRecommendation = () => {
    if (!analysis) return null;
    
    const currentPrice = analysis.market.current_price;
    const predictedPrice = analysis.price_prediction.predicted_price;
    const predictedPriceLower = analysis.price_prediction.confidence_interval.lower;
    const predictedPriceUpper = analysis.price_prediction.confidence_interval.upper;
    const strike = analysis.option.strike;
    const optionType = analysis.option.type;
    const premium = parseFloat(formData.premium);
    const directionProb = analysis.price_prediction.direction_probability;
    
    // Calculate break-even price
    let breakEvenPrice = 0;
    if (optionType === 'call') {
      breakEvenPrice = strike + premium;
    } else {
      breakEvenPrice = strike - premium;
    }
    
    // Calculate intrinsic value at predicted price
    let predictedIntrinsic = 0;
    if (optionType === 'call') {
      predictedIntrinsic = Math.max(0, predictedPrice - strike);
    } else {
      predictedIntrinsic = Math.max(0, strike - predictedPrice);
    }
    
    // Calculate profit/loss per share and as percentage
    const profitLossPerShare = predictedIntrinsic - premium;
    const profitLossPercent = (profitLossPerShare / premium) * 100;
    
    // Check if predicted price exceeds break-even
    let exceedsBreakEven = false;
    if (optionType === 'call') {
      exceedsBreakEven = predictedPrice > breakEvenPrice;
    } else {
      exceedsBreakEven = predictedPrice < breakEvenPrice;
    }
    
    // Recommendation based on expected profit, probability, and break-even
    if (exceedsBreakEven && profitLossPercent > 50 && directionProb > 0.6) {
      return { text: 'STRONG BUY', color: 'bg-green-600', profitLoss: profitLossPercent, breakEven: breakEvenPrice, exceedsBreakEven };
    }
    if (exceedsBreakEven && profitLossPercent > 20 && directionProb > 0.55) {
      return { text: 'BUY', color: 'bg-green-500', profitLoss: profitLossPercent, breakEven: breakEvenPrice, exceedsBreakEven };
    }
    if (!exceedsBreakEven || profitLossPercent < -50 || directionProb < 0.4) {
      return { text: 'AVOID', color: 'bg-red-600', profitLoss: profitLossPercent, breakEven: breakEvenPrice, exceedsBreakEven };
    }
    if (profitLossPercent < -20) {
      return { text: 'SELL', color: 'bg-red-500', profitLoss: profitLossPercent, breakEven: breakEvenPrice, exceedsBreakEven };
    }
    return { text: 'NEUTRAL', color: 'bg-gray-500', profitLoss: profitLossPercent, breakEven: breakEvenPrice, exceedsBreakEven };
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-red-600 via-red-700 to-blue-700 text-white py-6 shadow-lg">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center gap-3">
            <Activity size={32} />
            <div>
              <h1 className="text-3xl font-bold">Options Pricing Analyzer</h1>
              <p className="text-red-50 text-sm">ML-Powered Fair Value & Probability Analysis</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
              <LineChart size={20} className="text-red-600" />
              Option Details
            </h2>
            <Tooltip text="An option is a contract giving you the right (but not obligation) to buy (call) or sell (put) a stock at a specific price (strike) by a certain date (expiration). You pay a premium upfront for this right. Options are leveraged instruments that can amplify both gains and losses." />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1 flex items-center gap-1">
                Ticker
                <Tooltip position="left" text="The stock symbol (e.g., AAPL for Apple, TSLA for Tesla). This is the underlying asset the option contract is based on." />
              </label>
              <input
                type="text"
                value={formData.ticker}
                onChange={(e) => setFormData({...formData, ticker: e.target.value.toUpperCase()})}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="AAPL"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1 flex items-center gap-1">
                Strike Price
                <Tooltip text="The price at which you can buy (call) or sell (put) the stock if you exercise the option. Compare this to the current stock price to determine if the option is in-the-money (ITM), at-the-money (ATM), or out-of-the-money (OTM)." />
              </label>
              <input
                type="number"
                value={formData.strike}
                onChange={(e) => setFormData({...formData, strike: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                placeholder="270"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1 flex items-center gap-1">
                Premium/Share
                <Tooltip text="The price you pay per share for the option contract. Since one contract = 100 shares, multiply this by 100 to get the total cost. This is your maximum loss if the option expires worthless." />
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.premium}
                onChange={(e) => setFormData({...formData, premium: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                placeholder="5.50"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1 flex items-center gap-1">
                Type
                <Tooltip text="CALL: Right to BUY the stock at the strike price. Profit when stock goes UP. PUT: Right to SELL the stock at the strike price. Profit when stock goes DOWN." />
              </label>
              <select
                value={formData.optionType}
                onChange={(e) => setFormData({...formData, optionType: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
              >
                <option value="call">Call</option>
                <option value="put">Put</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1 flex items-center gap-1">
                Expiration
                <Tooltip text="The date when the option expires. After this date, the option becomes worthless if not exercised. Options lose value over time (time decay/theta), especially as expiration approaches." />
              </label>
              <input
                type="date"
                value={formData.expiration}
                onChange={(e) => setFormData({...formData, expiration: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="w-full bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-bold py-2 px-4 rounded transition-colors disabled:bg-gray-400"
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>
          </div>
          
          {/* Error Message */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
              <p className="font-semibold">Error:</p>
              <p>{error}</p>
            </div>
          )}
        </div>

        {analysis && (
          <>
            {/* Recommendation Banner */}
            <div className={`${getRecommendation().color} text-white rounded-lg shadow-lg p-6 mb-8`}>
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold mb-2">Recommendation: {getRecommendation().text}</h2>
                  <p className="text-white/90">
                    Expected P/L: {getRecommendation().profitLoss > 0 ? '+' : ''}{getRecommendation().profitLoss.toFixed(2)}%
                    <span className="mx-2">•</span>
                    Break-even: ${getRecommendation().breakEven.toFixed(2)}
                    <span className="mx-2">•</span>
                    Predicted: ${analysis.price_prediction.predicted_price.toFixed(2)}
                    {getRecommendation().exceedsBreakEven ? ' ✓' : ' ✗'}
                  </p>
                </div>
                {getRecommendation().profitLoss > 0 ? 
                  <TrendingUp size={48} /> : <TrendingDown size={48} />
                }
              </div>
            </div>

            {/* Market Overview */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-white rounded-lg shadow p-5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-600 text-sm font-semibold flex items-center gap-1">
                    Current Price
                    <Tooltip text="The current market price of the underlying stock. Compare this to your strike price to see if your option is in-the-money." />
                  </span>
                  <DollarSign size={18} className="text-blue-600" />
                </div>
                <p className="text-2xl font-bold text-gray-800">{formatCurrency(analysis.market.current_price)}</p>
                <p className="text-xs text-gray-500 mt-1">Strike: {formatCurrency(analysis.option.strike)}</p>
              </div>
              
              <div className="bg-white rounded-lg shadow p-5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-600 text-sm font-semibold flex items-center gap-1">
                    Days to Expiry
                    <Tooltip text="DTE (Days To Expiration) - How many days until the option expires. Options lose value faster as expiration approaches due to time decay (theta)." />
                  </span>
                  <Calendar size={18} className="text-blue-600" />
                </div>
                <p className="text-2xl font-bold text-gray-800">{analysis.option.dte}</p>
                <p className="text-xs text-gray-500 mt-1">{analysis.option.expiration}</p>
              </div>
              
              <div className="bg-white rounded-lg shadow p-5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-600 text-sm font-semibold flex items-center gap-1">
                    VIX Level
                    <Tooltip position="left" text="The CBOE Volatility Index - Wall Street's 'fear gauge'. Higher VIX (>20) means higher market fear and option premiums. Lower VIX (<15) means calmer markets and cheaper options." />
                  </span>
                  <Activity size={18} className="text-blue-600" />
                </div>
                <p className="text-2xl font-bold text-gray-800">{analysis.market.vix.toFixed(2)}</p>
                <p className="text-xs text-gray-500 mt-1">Market volatility</p>
              </div>
              
              <div className="bg-white rounded-lg shadow p-5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-600 text-sm font-semibold flex items-center gap-1">
                    Moneyness
                    <Tooltip position="left" text="Stock Price / Strike Price ratio. >1.0 = In-the-Money (ITM), ~1.0 = At-the-Money (ATM), <1.0 = Out-of-the-Money (OTM). ITM options have intrinsic value, OTM options only have time value." />
                  </span>
                  <LineChart size={18} className="text-blue-600" />
                </div>
                <p className="text-2xl font-bold text-gray-800">{analysis.option.moneyness.toFixed(3)}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {analysis.option.moneyness > 1 ? 'In the money' : 'Out of money'}
                </p>
              </div>
            </div>

            {/* Main Analysis Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Price Prediction */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <TrendingUp size={20} className="text-blue-600" />
                  Price Prediction (LSTM)
                  <Tooltip text="Long Short-Term Memory (LSTM) neural network predicts future stock price based on historical price patterns and technical indicators." />
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center pb-3 border-b">
                    <span className="text-gray-600 font-medium flex items-center gap-1">
                      Predicted Price
                      <Tooltip text="The expected stock price at option expiration based on our LSTM model. This is used to calculate expected option profit/loss." />
                    </span>
                    <span className="text-xl font-bold text-gray-800">
                      {formatCurrency(analysis.price_prediction.predicted_price)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center pb-3 border-b">
                    <span className="text-gray-600 font-medium flex items-center gap-1">
                      Expected Return
                      <Tooltip text="Predicted percentage change in stock price from current level to expiration. Positive = stock expected to rise, Negative = stock expected to fall." />
                    </span>
                    <span className={`text-xl font-bold ${analysis.price_prediction.predicted_return > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatPercent(analysis.price_prediction.predicted_return)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center pb-3 border-b">
                    <span className="text-gray-600 font-medium flex items-center gap-1">
                      Direction Confidence
                      <Tooltip text="Probability that the model's directional prediction (up/down) is correct based on historical accuracy. Higher percentage = more confidence in the direction." />
                    </span>
                    <span className="text-xl font-bold text-gray-800">
                      {formatPercent(analysis.price_prediction.direction_probability)}
                    </span>
                  </div>
                  <div className="bg-gray-50 p-3 rounded mt-3">
                    <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                      Confidence Interval (95%)
                      <Tooltip text="95% confidence range for the predicted price. There's a 95% probability the actual stock price will fall within this range at expiration." />
                    </p>
                    <p className="text-sm text-gray-700">
                      {formatCurrency(analysis.price_prediction.confidence_interval.lower)} - {formatCurrency(analysis.price_prediction.confidence_interval.upper)}
                    </p>
                  </div>
                  <div className="bg-blue-50 p-3 rounded flex items-start gap-2">
                    <AlertCircle size={16} className="text-blue-600 mt-0.5" />
                    <p className="text-xs text-blue-800">
                      Model MAE: {formatPercent(analysis.price_prediction.model_mae)}
                    </p>
                  </div>
                </div>
              </div>

              {/* IV Prediction */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <BarChart3 size={20} className="text-blue-600" />
                  Implied Volatility (LightGBM)
                  <Tooltip position="left" text="Implied Volatility (IV) is the market's forecast of future stock volatility, derived from option prices. Higher IV = more expensive options. IV represents expected annualized price movement." />
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center pb-3 border-b">
                    <span className="text-gray-600 font-medium flex items-center gap-1">
                      Predicted IV
                      <Tooltip position="left" text="Our ML model's prediction of what the implied volatility should be for this option based on market conditions and historical patterns." />
                    </span>
                    <span className="text-xl font-bold text-gray-800">
                      {formatPercent(analysis.iv_prediction.predicted_iv)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center pb-3 border-b">
                    <span className="text-gray-600 font-medium flex items-center gap-1">
                      30-Day HV
                      <Tooltip position="left" text="Historical Volatility - the actual price volatility of the stock over the past 30 days. Compare to IV to see if options are expensive (IV > HV) or cheap (IV < HV)." />
                    </span>
                    <span className="text-gray-700 font-semibold">
                      {formatPercent(analysis.market.historical_volatility.hv_30d)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center pb-3 border-b">
                    <span className="text-gray-600 font-medium flex items-center gap-1">
                      HV vs Predicted IV
                      <Tooltip position="left" text="Difference between Historical and Implied Volatility. Negative = options may be underpriced, Positive = options may be overpriced relative to historical movements." />
                    </span>
                    <span className={`font-bold ${analysis.iv_prediction.vs_historical_vol.difference < 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {formatPercent(analysis.iv_prediction.vs_historical_vol.difference)}
                    </span>
                  </div>
                  <div className="bg-gray-50 p-3 rounded mt-3">
                    <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                      IV Confidence Interval
                      <Tooltip position="left" text="95% confidence interval for the IV prediction. The true IV has a 95% probability of falling within this range." />
                    </p>
                    <p className="text-sm text-gray-700">
                      {formatPercent(analysis.iv_prediction.confidence_interval.lower)} - {formatPercent(analysis.iv_prediction.confidence_interval.upper)}
                    </p>
                  </div>
                  <div className="bg-purple-50 p-3 rounded">
                    <p className="text-xs text-purple-800">
                      <span className="font-semibold">Model Performance:</span> R² = {analysis.iv_prediction.model.test_r2.toFixed(4)}, 
                      MAPE = {analysis.iv_prediction.model.test_mape.toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Option Pricing */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-8">
              <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <DollarSign size={20} className="text-blue-600" />
                Theoretical Pricing & Greeks
                <Tooltip text="Black-Scholes model calculates fair option value and Greeks (risk metrics). Greeks measure how option price changes with different factors." />
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-4">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                    Fair Value (Pred. IV)
                    <Tooltip text="Theoretical option price calculated using predicted Implied Volatility. Compare to actual market price to find over/undervalued options." />
                  </p>
                  <p className="text-xl font-bold text-gray-800">{formatCurrency(analysis.option_pricing.theoretical_price)}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                    Fair Value (HV)
                    <Tooltip text="Theoretical price using Historical Volatility instead of IV. Useful for comparing current option prices to historical norms." />
                  </p>
                  <p className="text-xl font-bold text-gray-800">{formatCurrency(analysis.option_pricing.theoretical_price_hv)}</p>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                    Delta
                    <Tooltip text="Rate of change in option price per $1 stock move. 0.5 delta = option gains $0.50 when stock gains $1. Also represents approximate probability of expiring ITM. Range: 0 to 1 for calls, 0 to -1 for puts." />
                  </p>
                  <p className="text-lg font-bold text-gray-800">{formatGreek(analysis.option_pricing.greeks.delta)}</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                    Gamma
                    <Tooltip text="Rate of change of Delta per $1 stock move. High gamma = delta changes quickly = higher risk/reward. Peaks for ATM options near expiration." />
                  </p>
                  <p className="text-lg font-bold text-gray-800">{formatGreek(analysis.option_pricing.greeks.gamma)}</p>
                </div>
                <div className="bg-yellow-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                    Theta
                    <Tooltip text="Time decay - how much option loses per day as expiration approaches. Always negative for long options. Accelerates in the final 30 days. This is daily theta." />
                  </p>
                  <p className="text-lg font-bold text-gray-800">{formatGreek(analysis.option_pricing.greeks.theta)}</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                    Vega
                    <Tooltip position="left" text="Change in option price per 1% change in IV. High vega = option sensitive to volatility changes. Long options benefit when IV increases (vega profit)." />
                  </p>
                  <p className="text-lg font-bold text-gray-800">{formatGreek(analysis.option_pricing.greeks.vega)}</p>
                </div>
                <div className="bg-indigo-50 p-4 rounded-lg">
                  <p className="text-xs text-gray-600 font-semibold mb-1 flex items-center gap-1">
                    Rho
                    <Tooltip position="left" text="Change in option price per 1% change in interest rates. Usually small impact unless option is long-dated. Calls have positive rho, puts have negative rho." />
                  </p>
                  <p className="text-lg font-bold text-gray-800">{formatGreek(analysis.option_pricing.greeks.rho)}</p>
                </div>
              </div>
            </div>

            {/* Scenarios */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <BarChart3 size={20} className="text-blue-600" />
                Price Scenarios at Expiration
                <Tooltip text="Three possible outcomes at expiration: Bull (optimistic), Base (most likely), and Bear (pessimistic) cases with their probabilities and resulting option values." />
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Bear Case */}
                <div className="border-2 border-red-200 rounded-lg p-5 bg-red-50">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-bold text-gray-800">Bear Case</h4>
                    <TrendingDown className="text-red-600" size={20} />
                  </div>
                  <p className="text-sm text-gray-600 mb-2">Probability: {formatPercent(analysis.scenarios.bear_case.probability)}</p>
                  <p className="text-2xl font-bold text-gray-800 mb-2">{formatCurrency(analysis.scenarios.bear_case.stock_price)}</p>
                  <div className="pt-3 border-t border-red-200">
                    <p className="text-xs text-gray-600">Intrinsic Value</p>
                    <p className={`text-lg font-bold ${analysis.scenarios.bear_case.in_the_money ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(analysis.scenarios.bear_case.intrinsic_value)}
                    </p>
                  </div>
                </div>

                {/* Base Case */}
                <div className="border-2 border-blue-300 rounded-lg p-5 bg-blue-50">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-bold text-gray-800">Base Case</h4>
                    <Activity className="text-blue-600" size={20} />
                  </div>
                  <p className="text-sm text-gray-600 mb-2">Probability: {formatPercent(analysis.scenarios.base_case.probability)}</p>
                  <p className="text-2xl font-bold text-gray-800 mb-2">{formatCurrency(analysis.scenarios.base_case.stock_price)}</p>
                  <div className="pt-3 border-t border-blue-200">
                    <p className="text-xs text-gray-600">Intrinsic Value</p>
                    <p className={`text-lg font-bold ${analysis.scenarios.base_case.in_the_money ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(analysis.scenarios.base_case.intrinsic_value)}
                    </p>
                  </div>
                </div>

                {/* Bull Case */}
                <div className="border-2 border-green-300 rounded-lg p-5 bg-green-50">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-bold text-gray-800">Bull Case</h4>
                    <TrendingUp className="text-green-600" size={20} />
                  </div>
                  <p className="text-sm text-gray-600 mb-2">Probability: {formatPercent(analysis.scenarios.bull_case.probability)}</p>
                  <p className="text-2xl font-bold text-gray-800 mb-2">{formatCurrency(analysis.scenarios.bull_case.stock_price)}</p>
                  <div className="pt-3 border-t border-green-200">
                    <p className="text-xs text-gray-600">Intrinsic Value</p>
                    <p className={`text-lg font-bold ${analysis.scenarios.bull_case.in_the_money ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(analysis.scenarios.bull_case.intrinsic_value)}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Volatility Regime */}
            <div className="bg-gradient-to-r from-gray-700 to-gray-800 text-white rounded-lg shadow-md p-6 mt-8">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold mb-2">Market Context</h3>
                  <p className="text-gray-300 text-sm">
                    Sector: {analysis.market.sector} • Risk-Free Rate: {formatPercent(analysis.market.risk_free_rate)}
                  </p>
                  <p className="text-gray-300 text-sm mt-1">
                    Volatility Regime: <span className="font-semibold text-white">{analysis.market.historical_volatility.regime.toUpperCase()}</span>
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-400">Analysis Timestamp</p>
                  <p className="text-sm">{new Date(analysis.timestamp).toLocaleString()}</p>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default OptionsAnalyzer;
