# model_handler.py
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
import torch
from model import xLSTM 

class ModelHandler:
    def __init__(self, model_path: str, scaler_path: str):
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.close_idx = 0
        self.seq_length = 60
        self.load_model()
    
    def load_model(self) -> None:
        """Load trained PyTorch model and preprocessing components from separate files"""
        try:
            # Load scaler and metadata
            with open(self.scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            
            self.scaler = scaler_data.get('scaler')
            self.feature_columns = scaler_data.get('features', [])
            self.close_idx = scaler_data.get('close_idx', 0)
            self.seq_length = scaler_data.get('seq_length', 60)
            
            # Create and load PyTorch model
            input_size = len(self.feature_columns)
            self.model = xLSTM(
                input_size=input_size,
                hidden_size=128,
                num_layers=3,
                output_size=input_size
            )
            
            # Load model weights
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            logging.info(f"Model loaded from {self.model_path}")
            logging.info(f"Scaler loaded from {self.scaler_path}")
            logging.info(f"Expecting {input_size} features, sequence length {self.seq_length}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
        
        def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
            """Prepare features from market data for prediction"""
            try:
                # Extract features based on your training setup
                features = market_data[self.feature_columns].copy()
                
                # Apply same preprocessing as training
                if self.scaler:
                    features_scaled = self.scaler.transform(features)
                    return features_scaled
                
                return features.values
                
            except Exception as e:
                logging.error(f"Feature preparation failed: {e}")
                return None
        
        def predict(self, market_data: pd.DataFrame) -> Optional[Dict[str, float]]:
            """Generate trading signal from market data"""
            if self.model is None:
                logging.error("Model not loaded")
                return None
            
            try:
                features = self.prepare_features(market_data)
                if features is None:
                    return None
                
                # Convert to tensor and predict
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features[-self.seq_length:]).unsqueeze(0)
                    prediction = self.model(input_tensor)
                
                # Convert to trading signal
                signal_value = float(prediction.squeeze().numpy())
                
                return {
                    'signal': signal_value,
                    'confidence': abs(signal_value),
                    'timestamp': pd.Timestamp.now()
                }
                
            except Exception as e:
                logging.error(f"Prediction failed: {e}")
                return None