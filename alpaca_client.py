# alpaca_client.py
import alpaca_trade_api as tradeapi
import pandas as pd
from typing import Dict, List, Optional
import logging
from config import TradingConfig

class AlpacaClient:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.api = tradeapi.REST(
            config.API_KEY,
            config.SECRET_KEY,
            config.BASE_URL,
            api_version='v2'
        )
        self.account = None
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """Validate API connection and account access"""
        try:
            self.account = self.api.get_account()
            if self.account.trading_blocked:
                raise Exception("Trading is blocked on this account")
            logging.info(f"Connected to Alpaca. Account status: {self.account.status}")
        except Exception as e:
            logging.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def get_market_data(self, symbol: str, timeframe: str = '1Min', limit: int = 100) -> pd.DataFrame:
        """Fetch recent market data for a symbol"""
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                limit=limit
            ).df
            return bars
        except Exception as e:
            logging.error(f"Failed to fetch market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': int(account.day_trade_count)
            }
        except Exception as e:
            logging.error(f"Failed to get account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl)
            } for pos in positions]
        except Exception as e:
            logging.error(f"Failed to get positions: {e}")
            return []
    
    def place_order(self, symbol: str, qty: float, side: str, 
                   order_type: str = 'market', time_in_force: str = 'day') -> Optional[str]:
        """Place a trading order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            logging.info(f"Order placed: {side} {qty} {symbol}")
            return order.id
        except Exception as e:
            logging.error(f"Failed to place order: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logging.error(f"Failed to check market status: {e}")
            return False