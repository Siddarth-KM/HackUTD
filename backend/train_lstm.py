"""
LSTM Price Prediction Model Training Script

This script trains a Long Short-Term Memory (LSTM) neural network to predict
next-day stock returns based on historical sequences of technical indicators.

Architecture:
- Input: 30-day sequences of 66 features
- LSTM layers with dropout for regularization
- Dense layers for final prediction
- Output: Next-day return prediction

Training approach:
- Time-series sequences grouped by ticker
- Chronological train/test split
- Early stopping to prevent overfitting
- Normalized features using saved scalers
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering utilities
from feature_engineering import normalize_features, FEATURE_GROUPS


def create_sequences(X, y, dates, tickers, sequence_length=30):
    """
    Create time-series sequences for LSTM training
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        dates: Array of dates for each sample
        tickers: Array of tickers for each sample
        sequence_length: Number of timesteps in each sequence
    
    Returns:
        X_sequences: (n_sequences, sequence_length, n_features)
        y_sequences: (n_sequences,) - target for the day after sequence
        sequence_dates: Dates corresponding to each sequence
        sequence_tickers: Tickers corresponding to each sequence
    """
    print(f"\nCreating sequences with length {sequence_length}...")
    
    X_sequences = []
    y_sequences = []
    sequence_dates = []
    sequence_tickers = []
    
    # Get unique tickers
    unique_tickers = np.unique(tickers)
    print(f"Processing {len(unique_tickers)} unique tickers: {', '.join(unique_tickers)}")
    
    # Create sequences for each ticker separately (to maintain chronological order)
    for ticker in unique_tickers:
        # Get indices for this ticker
        ticker_mask = tickers == ticker
        ticker_indices = np.where(ticker_mask)[0]
        
        # Sort by date within this ticker
        ticker_dates = dates[ticker_mask]
        ticker_X = X[ticker_mask]
        ticker_y = y[ticker_mask]
        
        # Sort chronologically
        sort_indices = np.argsort(ticker_dates)
        sorted_dates = ticker_dates[sort_indices]
        sorted_X = ticker_X[sort_indices]
        sorted_y = ticker_y[sort_indices]
        
        # Create sequences
        ticker_sequences = 0
        for i in range(len(sorted_X) - sequence_length):
            # Sequence of features (30 days)
            seq_X = sorted_X[i:i+sequence_length]
            
            # Target is the return AFTER the sequence
            seq_y = sorted_y[i+sequence_length]
            
            # Date of the prediction (day after sequence)
            seq_date = sorted_dates[i+sequence_length]
            
            X_sequences.append(seq_X)
            y_sequences.append(seq_y)
            sequence_dates.append(seq_date)
            sequence_tickers.append(ticker)
            ticker_sequences += 1
        
        print(f"  {ticker}: {ticker_sequences} sequences created from {len(sorted_X)} samples")
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    sequence_dates = np.array(sequence_dates)
    sequence_tickers = np.array(sequence_tickers)
    
    print(f"\n✓ Total sequences created: {len(X_sequences)}")
    print(f"  Sequence shape: {X_sequences.shape}")
    print(f"  Target shape: {y_sequences.shape}")
    
    return X_sequences, y_sequences, sequence_dates, sequence_tickers


def build_lstm_model(sequence_length, n_features):
    """
    Build LSTM model architecture
    
    Args:
        sequence_length: Number of timesteps (e.g., 30 days)
        n_features: Number of features per timestep (e.g., 66)
    
    Returns:
        Compiled Keras model
    """
    print(f"\n[Building LSTM Model]")
    print(f"  Input shape: ({sequence_length} timesteps, {n_features} features)")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(sequence_length, n_features)),
        
        # First LSTM layer - learns temporal patterns
        layers.LSTM(
            units=128,
            return_sequences=True,  # Pass sequences to next LSTM layer
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout=0.2,           # Dropout on inputs
            recurrent_dropout=0.2, # Dropout on recurrent connections
            name='lstm_1'
        ),
        
        # Batch normalization for stability
        layers.BatchNormalization(name='batch_norm_1'),
        
        # Second LSTM layer - learns higher-level patterns
        layers.LSTM(
            units=64,
            return_sequences=False,  # Output single vector
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout=0.2,
            recurrent_dropout=0.2,
            name='lstm_2'
        ),
        
        # Batch normalization
        layers.BatchNormalization(name='batch_norm_2'),
        
        # Dense layer 1 - feature extraction
        layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='dense_1'
        ),
        layers.Dropout(0.3, name='dropout_1'),
        
        # Dense layer 2 - refinement
        layers.Dense(
            units=16,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='dense_2'
        ),
        layers.Dropout(0.2, name='dropout_2'),
        
        # Output layer - single value (next-day return)
        layers.Dense(
            units=1,
            activation='linear',  # Regression task
            name='output'
        )
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',  # Huber loss is robust to outliers
        metrics=['mae', 'mse']
    )
    
    print("\n✓ Model architecture:")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train LSTM model with early stopping and checkpointing
    
    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
        epochs: Maximum epochs
        batch_size: Batch size for training
    
    Returns:
        Training history
    """
    print(f"\n[Training LSTM Model]")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    
    # Create callbacks
    callback_list = [
        # Early stopping - stop if validation loss doesn't improve
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Stop after 15 epochs without improvement
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,      # Reduce LR by half
            patience=7,       # After 7 epochs without improvement
            min_lr=1e-6,
            verbose=1
        ),
        
        # Model checkpoint - save best model
        callbacks.ModelCheckpoint(
            filepath='backend/models/lstm_price_predictor_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging (optional)
        callbacks.TensorBoard(
            log_dir=f'backend/logs/lstm_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=0
        )
    ]
    
    print("\n" + "="*70)
    print("Starting training... (this will take 30-40 minutes)")
    print("="*70 + "\n")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1  # Show progress bar
    )
    
    print("\n✓ Training complete!")
    
    return history


def evaluate_model(model, X_test, y_test, sequence_tickers):
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Test targets
        sequence_tickers: Tickers for each test sequence
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n[Evaluating Model Performance]")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (avoiding division by zero)
    mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))) * 100
    
    # Directional accuracy (did we predict the right direction?)
    direction_actual = np.sign(y_test)
    direction_pred = np.sign(y_pred)
    direction_accuracy = np.mean(direction_actual == direction_pred)
    
    print(f"\n  TEST SET METRICS:")
    print(f"    MAE:  {mae:.4f} (±{mae*100:.2f}% daily return error)")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R²:   {r2:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    Directional Accuracy: {direction_accuracy:.2%}")
    
    # Per-ticker performance
    print(f"\n  PER-TICKER PERFORMANCE:")
    unique_tickers = np.unique(sequence_tickers)
    for ticker in sorted(unique_tickers):
        ticker_mask = sequence_tickers == ticker
        ticker_mae = mean_absolute_error(y_test[ticker_mask], y_pred[ticker_mask])
        ticker_r2 = r2_score(y_test[ticker_mask], y_pred[ticker_mask])
        ticker_dir_acc = np.mean(
            np.sign(y_test[ticker_mask]) == np.sign(y_pred[ticker_mask])
        )
        print(f"    {ticker}: MAE={ticker_mae:.4f}, R²={ticker_r2:.4f}, Dir Acc={ticker_dir_acc:.2%}")
    
    # Sample predictions
    print(f"\n  SAMPLE PREDICTIONS (first 10 test samples):")
    print(f"  {'Ticker':<8} {'Actual Return':<15} {'Predicted':<15} {'Error':<15} {'Direction'}")
    print(f"  {'-'*75}")
    for i in range(min(10, len(y_test))):
        ticker = sequence_tickers[i]
        actual = y_test[i]
        pred = y_pred[i]
        error = pred - actual
        direction = '✓' if np.sign(actual) == np.sign(pred) else '✗'
        print(f"  {ticker:<8} {actual:>13.4f} {pred:>13.4f} {error:>13.4f}   {direction}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'predictions': y_pred,
        'actuals': y_test
    }


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("           LSTM PRICE PREDICTION - TRAINING")
    print("="*70)
    
    # ================================================================
    # [1] Load training data
    # ================================================================
    print("\n[1] Loading training data...")
    
    train_df = pd.read_csv('backend/data/train.csv')
    test_df = pd.read_csv('backend/data/test.csv')
    
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    
    # ================================================================
    # [2] Prepare features and targets
    # ================================================================
    print("\n[2] Preparing features and targets...")
    
    # Feature columns (66 features)
    feature_cols = [col for col in train_df.columns if col not in [
        'date', 'ticker', 'strike', 'expiration', 'option_type',
        'target_next_day_return', 'target_actual_iv', 
        'target_price_direction', 'target_iv_category'
    ]]
    
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Target variable: target_next_day_return")
    
    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_train = train_df['target_next_day_return'].values
    dates_train = pd.to_datetime(train_df['date']).values
    tickers_train = train_df['ticker'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target_next_day_return'].values
    dates_test = pd.to_datetime(test_df['date']).values
    tickers_test = test_df['ticker'].values
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train range: {y_train.min():.4f} to {y_train.max():.4f}")
    print(f"  Mean return: {y_train.mean():.4f}")
    
    # ================================================================
    # [3] Normalize features
    # ================================================================
    print("\n[3] Normalizing features...")
    
    # Load the scaler created during LightGBM training
    scaler_path = 'backend/scalers/feature_scalers.pkl'
    
    if not os.path.exists(scaler_path):
        print("  ERROR: Scaler not found! Run train_lightgbm.py first.")
        return
    
    scaler_dict = joblib.load(scaler_path)
    print(f"  Loaded scalers from: {scaler_path}")
    
    # Normalize using the same scalers as LightGBM
    train_feature_df = pd.DataFrame(X_train, columns=feature_cols)
    test_feature_df = pd.DataFrame(X_test, columns=feature_cols)
    
    X_train_norm = normalize_features(train_feature_df, scaler_dict, return_dataframe=False)
    X_test_norm = normalize_features(test_feature_df, scaler_dict, return_dataframe=False)
    
    print(f"  Normalized train shape: {X_train_norm.shape}")
    print(f"  Normalized test shape: {X_test_norm.shape}")
    
    # ================================================================
    # [4] Create sequences
    # ================================================================
    sequence_length = 30  # Use 30 days of history to predict next day
    
    X_train_seq, y_train_seq, train_seq_dates, train_seq_tickers = create_sequences(
        X_train_norm, y_train, dates_train, tickers_train, sequence_length
    )
    
    X_test_seq, y_test_seq, test_seq_dates, test_seq_tickers = create_sequences(
        X_test_norm, y_test, dates_test, tickers_test, sequence_length
    )
    
    print(f"\n  Training sequences: {len(X_train_seq):,}")
    print(f"  Test sequences: {len(X_test_seq):,}")
    
    # ================================================================
    # [5] Split training into train/validation
    # ================================================================
    print("\n[5] Creating train/validation split...")
    
    # Split by date for each ticker (to avoid ticker bias)
    # This ensures validation has all tickers represented
    val_split = 0.2
    
    X_train_final = []
    y_train_final = []
    X_val = []
    y_val = []
    
    # Get unique tickers in training sequences
    unique_tickers = np.unique(train_seq_tickers)
    
    for ticker in unique_tickers:
        # Get sequences for this ticker
        ticker_mask = train_seq_tickers == ticker
        ticker_X = X_train_seq[ticker_mask]
        ticker_y = y_train_seq[ticker_mask]
        ticker_dates = train_seq_dates[ticker_mask]
        
        # Sort by date (should already be sorted, but ensure)
        sort_idx = np.argsort(ticker_dates)
        ticker_X = ticker_X[sort_idx]
        ticker_y = ticker_y[sort_idx]
        
        # Split chronologically (last 20% for validation)
        n_val = int(len(ticker_X) * val_split)
        n_train = len(ticker_X) - n_val
        
        X_train_final.append(ticker_X[:n_train])
        y_train_final.append(ticker_y[:n_train])
        
        X_val.append(ticker_X[n_train:])
        y_val.append(ticker_y[n_train:])
    
    # Concatenate all tickers
    X_train_final = np.concatenate(X_train_final, axis=0)
    y_train_final = np.concatenate(y_train_final, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    
    print(f"  Final training sequences: {len(X_train_final):,}")
    print(f"  Validation sequences: {len(X_val):,}")
    print(f"  Test sequences: {len(X_test_seq):,}")
    print(f"  Tickers in validation: {len(unique_tickers)}")
    
    # ================================================================
    # [6] Build model
    # ================================================================
    n_features = X_train_seq.shape[2]
    model = build_lstm_model(sequence_length, n_features)
    
    # ================================================================
    # [7] Train model
    # ================================================================
    history = train_model(
        model, 
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # ================================================================
    # [8] Evaluate on test set
    # ================================================================
    metrics = evaluate_model(model, X_test_seq, y_test_seq, test_seq_tickers)
    
    # ================================================================
    # [9] Save model and metadata
    # ================================================================
    print("\n[9] Saving model and metadata...")
    
    # Save model
    model_path = 'backend/models/lstm_price_predictor.h5'
    model.save(model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'sequence_length': sequence_length,
        'n_features': n_features,
        'feature_columns': feature_cols,
        'train_samples': len(X_train_final),
        'val_samples': len(X_val),
        'test_samples': len(X_test_seq),
        'test_mae': float(metrics['mae']),
        'test_rmse': float(metrics['rmse']),
        'test_r2': float(metrics['r2']),
        'test_mape': float(metrics['mape']),
        'direction_accuracy': float(metrics['direction_accuracy']),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }
    
    metadata_path = 'backend/models/lstm_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"  ✓ Metadata saved: {metadata_path}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = 'backend/models/lstm_training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"  ✓ Training history saved: {history_path}")
    
    # ================================================================
    # [10] Summary
    # ================================================================
    print("\n" + "="*70)
    print("                    TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n  Model Performance:")
    print(f"    Test MAE: {metrics['mae']:.4f} (±{metrics['mae']*100:.2f}% daily return)")
    print(f"    Test R²:  {metrics['r2']:.4f}")
    print(f"    Test MAPE: {metrics['mape']:.2f}%")
    print(f"    Directional Accuracy: {metrics['direction_accuracy']:.2%}")
    
    print(f"\n  Files created:")
    print(f"    - {model_path}")
    print(f"    - backend/models/lstm_price_predictor_best.h5")
    print(f"    - {metadata_path}")
    print(f"    - {history_path}")
    
    print(f"\n  Model ready for predictions!")
    print(f"  Next step: Create /api/predict-price endpoint in Flask")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
