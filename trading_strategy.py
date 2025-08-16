# trading_strategy.py
from typing import Dict, Optional, List
import logging
import pandas as pd
from model_handler import ModelHandler
from alpaca_client import AlpacaClient
from config import TradingConfig

class TradingStrategy:
    def __init__(self, config: TradingConfig, model_handler: ModelHandler, 
                 alpaca_client: AlpacaClient):
        self.config = config
        self.model = model_handler
        self.alpaca = alpaca_client
        self.daily_trades = 0
        self.last_trade_date = None
    
    def reset_daily_counters(self) -> None:
        """Reset daily trading counters if new day"""
        current_date = pd.Timestamp.now().date()
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
    
    def should_trade(self, signal: Dict) -> bool:
        """Determine if we should execute a trade based on signal and risk rules"""
        self.reset_daily_counters()
        
        # Check daily trade limit
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            logging.info("Daily trade limit reached")
            return False
        
        # Check account balance
        account_info = self.alpaca.get_account_info()
        if account_info.get('portfolio_value', 0) < self.config.MIN_ACCOUNT_BALANCE:
            logging.warning("Account balance too low")
            return False
        
        # Check market hours
        if not self.alpaca.is_market_open():
            logging.info("Market is closed")
            return False
        
        # Check signal strength (adjust thresholds based on your model)
        if signal.get('confidence', 0) < 0.6:  # Minimum 60% confidence
            logging.info("Signal confidence too low")
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calculate position size based on signal strength and risk management"""
        account_info = self.alpaca.get_account_info()
        portfolio_value = account_info.get('portfolio_value', 0)
        
        # Base position size as percentage of portfolio
        base_size = portfolio_value * self.config.MAX_POSITION_SIZE
        
        # Scale by signal strength
        adjusted_size = base_size * min(signal_strength, 1.0)
        
        # Get current price to convert to shares
        market_data = self.alpaca.get_market_data(symbol, limit=1)
        if market_data.empty:
            return 0
        
        current_price = market_data['close'].iloc[-1]
        shares = int(adjusted_size / current_price)
        
        return max(1, shares)  # Minimum 1 share
    
    def execute_signal(self, symbol: str, signal: Dict) -> bool:
        """Execute trading signal"""
        if not self.should_trade(signal):
            return False
        
        try:
            # Determine trade direction based on your model's output format
            # Adjust this logic based on how your model encodes signals
            if signal['signal'] > 0.5:  # Buy signal
                side = 'buy'
                qty = self.calculate_position_size(symbol, signal['confidence'])
            elif signal['signal'] < -0.5:  # Sell signal
                side = 'sell'
                # For sell signals, check if we have position to sell
                positions = self.alpaca.get_positions()
                position_qty = 0
                for pos in positions:
                    if pos['symbol'] == symbol:
                        position_qty = pos['qty']
                        break
                
                if position_qty <= 0:
                    logging.info(f"No position to sell for {symbol}")
                    return False
                
                qty = min(position_qty, self.calculate_position_size(symbol, signal['confidence']))
            else:
                logging.info("Neutral signal, no action taken")
                return False
            
            # Place the order
            order_id = self.alpaca.place_order(symbol, qty, side)
            
            if order_id:
                self.daily_trades += 1
                logging.info(f"Trade executed: {side} {qty} {symbol} (Order ID: {order_id})")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Failed to execute signal: {e}")
            return False
    
    def set_stop_loss_take_profit(self, symbol: str, entry_price: float, side: str) -> None:
        """Set stop loss and take profit orders (bracket orders)"""
        try:
            if side == 'buy':
                stop_price = entry_price * (1 - self.config.STOP_LOSS)
                take_profit_price = entry_price * (1 + self.config.TAKE_PROFIT)
            else:
                stop_price = entry_price * (1 + self.config.STOP_LOSS)
                take_profit_price = entry_price * (1 - self.config.TAKE_PROFIT)
            
            # Place stop loss
            self.alpaca.place_order(
                symbol, 1, 'sell' if side == 'buy' else 'buy',
                order_type='stop', stop_price=stop_price
            )
            
            # Place take profit
            self.alpaca.place_order(
                symbol, 1, 'sell' if side == 'buy' else 'buy',
                order_type='limit', limit_price=take_profit_price
            )
            
        except Exception as e:
            logging.error(f"Failed to set stop loss/take profit: {e}")