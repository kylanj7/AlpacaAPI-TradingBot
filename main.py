# main.py
import time
import logging
from typing import List
from config import TradingConfig
from model_handler import ModelHandler
from alpaca_client import AlpacaClient
from trading_strategy import TradingStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingBot:
    def __init__(self, symbols: List[str]):
        self.config = TradingConfig()
        self.symbols = symbols
        
        # Initialize components
        self.model_handler = ModelHandler(self.config.MODEL_PATH, self.config.SCALER_PATH)
        self.alpaca_client = AlpacaClient(self.config)
        self.strategy = TradingStrategy(self.config, self.model_handler, self.alpaca_client)
        
        logging.info("Trading bot initialized successfully")
    
    def run_single_iteration(self) -> None:
        """Run one iteration of the trading loop"""
        try:
            for symbol in self.symbols:
                # Get market data
                market_data = self.alpaca_client.get_market_data(symbol, limit=100)
                if market_data.empty:
                    logging.warning(f"No market data for {symbol}")
                    continue
                
                # Generate prediction
                signal = self.model_handler.predict(market_data)
                if signal is None:
                    logging.warning(f"No signal generated for {symbol}")
                    continue
                
                # Execute trade if signal is strong enough
                self.strategy.execute_signal(symbol, signal)
                
        except Exception as e:
            logging.error(f"Error in trading iteration: {e}")
    
    def run(self) -> None:
        """Main trading loop"""
        logging.info("Starting trading bot...")
        
        try:
            while True:
                if self.alpaca_client.is_market_open():
                    self.run_single_iteration()
                else:
                    logging.info("Market closed, waiting...")
                
                time.sleep(self.config.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logging.info("Trading bot stopped by user")
        except Exception as e:
            logging.error(f"Fatal error: {e}")
        finally:
            logging.info("Trading bot shutdown")

if __name__ == "__main__":
    # Define symbols to trade
    SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']  # Adjust based on your model
    
    # Create and run bot
    bot = TradingBot(SYMBOLS)
    bot.run()