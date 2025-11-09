# MetaDAsh - Modular Trading Dashboard

A professional, modular MetaTrader 5 trading analytics dashboard built with Dash and Plotly.

## 🚀 Features

- **Real-time MT5 Integration**: Direct connection to MetaTrader 5 terminal
- **Modular Architecture**: Clean separation of concerns with organized components
- **Professional Styling**: Custom CSS with modern, responsive design
- **Comprehensive Analytics**: 
  - Account overview with key metrics
  - Strategy performance analysis
  - PnL tracking and drawdown analysis
  - Trade-by-trade details
  - Win/loss streak analysis
  - Hourly performance patterns

## 📁 Project Structure

```
metadash_modular/
│
├── app.py                 # Main application entry point
├── callbacks.py           # All Dash callbacks
├── requirements.txt       # Python dependencies
│
├── assets/               
│   └── styles.css        # Custom CSS styling
│
├── layouts/              # Layout components
│   ├── __init__.py
│   ├── main_layout.py    # Main app layout
│   ├── header.py         # Header component
│   ├── sidebar.py        # Sidebar with controls
│   └── tabs.py           # Tab navigation
│
├── components/           # Tab content components
│   ├── __init__.py
│   ├── overview_tab.py   # Overview dashboard
│   ├── detailed_tab.py   # Detailed analysis
│   ├── pnl_tab.py        # PnL performance
│   ├── trades_tab.py     # Individual trades
│   └── raw_tab.py        # Data export
│
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── mt5_utils.py      # MT5 connection & data
│   └── metrics.py        # Metrics calculations
│
└── data/                 # Data storage (auto-created)
    └── historical_merged_deals.pkl
```

## 🛠️ Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
cd metadash_modular
pip install -r requirements.txt
```

3. **Ensure MetaTrader 5 is installed and running**

## 🚀 Running the Application

```bash
python app.py
```

The dashboard will be available at: `http://localhost:8050`

## 🎨 Customization

### Styling
- Edit `assets/styles.css` to customize colors, fonts, and layout
- The app uses CSS variables for easy theming:
  ```css
  :root {
    --primary-color: #0066cc;
    --secondary-color: #28a745;
    --danger-color: #dc3545;
    ...
  }
  ```

### Adding New Components
1. Create a new component in `components/`
2. Import it in `components/__init__.py`
3. Add corresponding callback in `callbacks.py`
4. Include in the layout via `layouts/`

### Modifying Metrics
- Edit `utils/metrics.py` to add new calculations
- Update relevant tab components to display new metrics

## 📊 Usage Guide

### 1. Initial Setup
- Launch the app
- Click "Connect to MT5" to establish connection
- Set date range and account size
- Click "Fetch Trading Data" to load trades

### 2. Overview Tab
- View account balance, equity, and margin
- Monitor daily performance trends
- Compare strategy performance metrics

### 3. Detailed Analysis Tab
- Select specific metrics to analyze
- View trade duration distributions
- Analyze hourly performance patterns

### 4. PnL Performance Tab
- Track equity curve over time
- Monitor drawdown levels
- Analyze win/loss streaks
- View individual trade performance

### 5. Trades Table Tab
- Filter trades by bot/strategy
- View detailed trade information
- Sort and search functionality

### 6. Raw Data Tab
- Preview raw data tables
- Export data as CSV files
- View data statistics

## 🔧 Configuration

### MT5 Connection
The app expects trades with specific comment patterns:
- Opens: Comments containing "meta"
- Closes: Comments containing "sl", "tp", or "Close"

Modify patterns in `utils/mt5_utils.py` if needed.

### Data Persistence
Historical data is automatically saved to `data/historical_merged_deals.pkl`

## 📈 Key Metrics Explained

- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Ratio of gross profit to gross loss
- **Win Rate**: Percentage of profitable trades
- **RRR**: Risk-Reward Ratio (average win/average loss)

## 🐛 Troubleshooting

### MT5 Connection Issues
- Ensure MT5 terminal is running
- Check that algo trading is enabled
- Verify Python MT5 package version compatibility

### No Data Retrieved
- Check date range contains trades
- Verify trade comment patterns match filters
- Ensure account has trading history

### Performance Issues
- Reduce date range for large datasets
- Clear browser cache
- Check system resources

## 🔄 Updates & Maintenance

### Adding New Features
1. Create feature branch
2. Add component/utility as needed
3. Update callbacks
4. Test thoroughly
5. Update documentation

### Regular Maintenance
- Clear old pickle files periodically
- Update dependencies: `pip install --upgrade -r requirements.txt`
- Monitor performance with large datasets

## 📝 License

This project is provided as-is for educational and trading analysis purposes.

## 🤝 Contributing

Contributions welcome! Please follow the modular structure and include documentation for new features.

## 📞 Support

For issues or questions, please refer to the documentation or create an issue in the project repository.
