# AlpacaAPI-TradingBot# AI-Powered Algorithmic Trading Bot

A sophisticated Python trading bot leveraging deep learning (xLSTM) for automated trading decisions via Alpaca Markets API. Features comprehensive risk management, real-time market analysis, and intelligent position sizing.

## 🚀 Key Features

**Trading Engine**
- Real-time automated execution through Alpaca API
- Custom xLSTM neural network with attention mechanisms  
- Multi-scale convolutions for pattern recognition
- Paper trading support for safe testing

**Risk Management**
- Dynamic position sizing based on signal confidence
- Configurable stop-loss and take-profit levels
- Daily trade limits and account balance monitoring
- Market hours validation

**Architecture**
- Modular design with separate components for model handling, trading strategy, and API communication
- Enhanced LSTM with layer normalization and residual connections
- Comprehensive logging and error handling
- Configurable parameters via centralized config

## 📋 Requirements

- Python 3.8+
- PyTorch for deep learning model
- Alpaca Trade API for market access
- See `requirements.txt` for complete dependencies

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure settings**
- Update `config.py` with your Alpaca API credentials
- Set model paths and trading parameters
- Adjust risk management settings

4. **Run the bot**
```bash
python main.py
```

## 📁 Project Structure

```
├── main.py              # Main entry point and trading loop
├── config.py            # Configuration settings
├── model.py             # xLSTM neural network architecture
├── model_handler.py     # Model loading and prediction logic
├── alpaca_client.py     # Alpaca API wrapper
├── trading_strategy.py  # Trading logic and risk management
└── requirements.txt     # Python dependencies
```

## ⚙️ Configuration

Key settings in `config.py`:
- `MAX_POSITION_SIZE`: Maximum portfolio percentage per trade (default: 10%)
- `STOP_LOSS`: Stop loss percentage (default: 5%)
- `TAKE_PROFIT`: Take profit percentage (default: 10%)
- `MAX_DAILY_TRADES`: Daily trade limit (default: 10)

## 🤖 Model Architecture

The xLSTM model features:
- Multi-layer LSTM with dropout and layer normalization
- Attention mechanism for focusing on relevant patterns
- Multi-scale convolutional layers (3, 5, 7 kernel sizes)
- Residual connections for improved gradient flow
- GELU activation functions for better performance

## ⚠️ Risk Disclaimer

This bot is for educational purposes. Always test with paper trading first. Trading involves risk of financial loss. Past performance doesn't guarantee future results.

## 📄 License

MIT License - see LICENSE file for details.
