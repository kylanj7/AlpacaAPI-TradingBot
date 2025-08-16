# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingConfig:
    # Alpaca API credentials (paper trading)
    API_KEY: str = 'PKEA698MGJA2UD1BUYC2'
    SECRET_KEY: str = 'zGIfvFFLfIrRUKwQemX0oOcHbDIE7mEGvjKlQKpx'
    BASE_URL: str = 'https://paper-api.alpaca.markets'  # Paper trading endpoint
    
    # Model configuration
    MODEL_PATH: str = r'C:\Users\Kylan\Desktop\PaperTrade\Model\best_model_20250816_142411.pth'
    SCALER_PATH: str = r'C:\Users\Kylan\Desktop\PaperTrade\Scaler\scaler.pkl'

    # Trading parameters
    MAX_POSITION_SIZE: float = 0.1  # Max 10% of portfolio per position
    STOP_LOSS: float = 0.05  # 5% stop loss
    TAKE_PROFIT: float = 0.10  # 10% take profit
    
    # Risk management
    MAX_DAILY_TRADES: int = 10
    MIN_ACCOUNT_BALANCE: float = 1000.0
    
    # Timing
    CHECK_INTERVAL: int = 60  # Check every 60 seconds
    MARKET_OPEN_BUFFER: int = 5  # Wait 5 minutes after market open