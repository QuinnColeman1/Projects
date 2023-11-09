###############################################################################
# FINANCIAL DASHBOARD
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf



#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret




#==============================================================================
# Header
#==============================================================================

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """
    
    # Add dashboard title and description
    st.title("MY FINANCIAL DASHBOARD")
    col1, col2 = st.columns([1,5])
    col1.write("Data source:")
    col2.image('./img/yahoo_finance.png', width=100)
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    global ticker_list
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    # Add the selection boxes
    col1, col2, col3 = st.columns(3)  # Create 3 columns
    # Ticker name
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = col1.selectbox("Ticker", ticker_list)
    # Begin and end dates
    global start_date, end_date  # Set this variable as global, so all functions can read it
    start_date = col2.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col3.date_input("End date", datetime.today().date())
    

#==============================================================================
# Tab 1
#==============================================================================

def render_tab1():
    
    """
    This function render the Tab 1 - Company Profile of the dashboard.
    """

    def fetch_stock_data(ticker, period):
        stock_data = yf.download(ticker, period=period, interval="1d")
        return stock_data['Close']

    
        
    if ticker !="":
        ticker_obj = YFinance(ticker)
    
        # Display basic stock information
        st.subheader(f"Summary for {ticker}")
        st.write("Open:", ticker_obj.info["open"])
        st.write("Previous Close:", ticker_obj.info["previousClose"])
        st.write("Volume:", ticker_obj.info["volume"])
        st.write("Market Cap:", ticker_obj.info["marketCap"])
        st.write("52 Week Range:", ticker_obj.info["fiftyTwoWeekLow"], "-", ticker_obj.info["fiftyTwoWeekHigh"])
    
        # Display the chart with selectable duration
        time_options = {
            "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "3Y": "3y", "5Y": "5y", "MAX": "max"
        }
        selected_time = st.selectbox("Select Time Duration:", list(time_options.keys()))
    
        stock_data = fetch_stock_data(ticker, time_options[selected_time])
        st.line_chart(stock_data)
        
        
        
        
        
    
        # Display company profile and description
        st.subheader("Company Profile")
        st.write(ticker_obj.info["longBusinessSummary"])
    
        # Display major shareholders
        st.subheader("Major Shareholders")
        st.table(yf.Ticker(ticker).major_holders)
        
        st.write("Institutional Holders")
        st.table(yf.Ticker(ticker).institutional_holders)
            
#==============================================================================
# Tab 2
#==============================================================================

@st.cache_data
def fetch_stock_data(ticker, start_date=None, end_date=None, interval='1d'):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

def render_tab2():
    """
    This function renders the Tab 2 - Chart.
    """
    if ticker != '':
        # Date range selector
        st.subheader("Select Time Duration")
        duration = st.selectbox("", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"])
    
        today = pd.to_datetime("today").normalize()
        start_date = None
        end_date = today

        # Calculate start_date based on duration
        if duration == "1M":
            start_date = today - pd.DateOffset(months=1)
        elif duration == "3M":
            start_date = today - pd.DateOffset(months=3)
        elif duration == "6M":
            start_date = today - pd.DateOffset(months=6)
        elif duration == "YTD":
            start_date = pd.to_datetime(f"{today.year}-01-01")
        elif duration == "1Y":
            start_date = today - pd.DateOffset(years=1)
        elif duration == "3Y":
            start_date = today - pd.DateOffset(years=3)
        elif duration == "5Y":
            start_date = today - pd.DateOffset(years=5)
        # If duration is "MAX", start_date will remain None, which yfinance interprets as the max period available

        # Time interval selector
        st.subheader("Select Time Interval")
        interval = st.selectbox("", ["1d", "5d", "1wk", "1mo", "3mo"])
    
        # Plot type selector
        st.subheader("Select Plot Type")
        plot_type = st.radio("", ["Line", "Candle"])
    
        # Fetch the stock data
        df = fetch_stock_data(ticker, start_date, end_date, interval)
    
        if not df.empty:
            # Calculate moving average
            df['MA50'] = df['Close'].rolling(window=50).mean()
    
            # Create the plot
            fig = go.Figure()
    
            if plot_type == "Line":
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50'))
    
            elif plot_type == "Candle":
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50'))
    
            # Display the plot
            st.plotly_chart(fig)
    
#==============================================================================
# Tab 3
#==============================================================================

def render_tab3():
    
    """
    This function render the Tab 3 - Financials.
    """
    
    
    def fetch_financial_data(ticker, financial_type, frequency):
    # Fetch the Yahoo Finance Ticker object
        ticker_obj = yf.Ticker(ticker)
    
        # Mapping of user selection to yfinance function calls
        data_fetch_map = {
            'Income Statement': {
                'Annual': ticker_obj.financials,
                'Quarterly': ticker_obj.quarterly_financials
            },
            'Balance Sheet': {
                'Annual': ticker_obj.balancesheet,
                'Quarterly': ticker_obj.quarterly_balancesheet
            },
            'Cash Flow': {
                'Annual': ticker_obj.cashflow,
                'Quarterly': ticker_obj.quarterly_cashflow
            }
        }
    
        # Return the selected data
        return data_fetch_map[financial_type][frequency]

    # Streamlit UI
    st.title("Financial Statements Viewer")
    
        
    # Select financial statement type and frequency
    financial_type = st.selectbox("Choose the Financial Statement:", ["Income Statement", "Balance Sheet", "Cash Flow"])
    frequency = st.selectbox("Choose the Frequency:", ["Annual", "Quarterly"])
    
    if ticker:
        try:
            # Fetch data based on user input
            data = fetch_financial_data(ticker, financial_type, frequency)
    
            # Display the financial data in a table format
            st.write(data)
        except:
            st.write(f"Data not available for {ticker} or an error occurred.")

    
#==============================================================================
# Tab 4
#==============================================================================

def render_tab4():
    """
    This function render the Tab 4 - Monte Carlo simulation.
    """

    @st.cache_data
    def fetch_stock_data(ticker, period="1y"):
        return yf.Ticker(ticker).history(period=period)

    def simulate_stock_path(stock_price, daily_volatility, n_days=30, n_simulations=1000):
        simulated_df = pd.DataFrame()
        current_price = stock_price['Close'][-1]

        for r in range(n_simulations):
            stock_price_list = [current_price]
            for i in range(n_days):
                daily_return = np.random.normal(0, daily_volatility, 1)[0]
                future_price = stock_price_list[-1] * (1 + daily_return)
                stock_price_list.append(future_price)
            simulated_df['Sim' + str(r)] = stock_price_list[1:]

        return simulated_df

    def compute_daily_volatility(stock_price):
        daily_returns = stock_price['Close'].pct_change().dropna()
        return daily_returns.std()

       

    if ticker:
        stock_price = fetch_stock_data(ticker)
        daily_volatility = compute_daily_volatility(stock_price)

        st.write(f"Current closing price of {ticker}: ${stock_price['Close'][-1]:,.2f}")

        n_simulations = st.selectbox("Number of Simulations:", [200, 500, 1000])
        n_days = st.selectbox("Time Horizon (days from today):", [30, 60, 90])

        st.write(f"Running {n_simulations} simulations of the stock price for the next {n_days} days...")

        simulated_df = simulate_stock_path(stock_price, daily_volatility, n_days, n_simulations)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        for i in simulated_df.columns:
            plt.plot(simulated_df[i], alpha=0.5, linewidth=2)
            
        plt.xlabel('Days from Today')
        plt.ylabel('Simulated Stock Price')
        plt.title(f"Monte Carlo Simulation of {ticker} Stock Price over Next {n_days} Days")
        st.pyplot()

        # Value at Risk (VaR) at 95% confidence
        var_95 = np.percentile(simulated_df.iloc[-1], 5)
        st.write(f"Value at Risk (VaR) at 95% confidence level: ${stock_price['Close'][-1] - var_95:,.2f}")#==============================================================================


###############################################################################
# Tab 5
#==============================================================================

def render_tab5():
    """
    This function render the Tab 5 - Your own analysis.
    """
    
    selected_tickers = st.multiselect("Select Tickers for Analysis", ticker_list)  # default=ticker to retain previously selected tickers

    # If no ticker is selected, return to avoid errors
    if not selected_tickers:
        st.warning("Please select at least one ticker.")
        return

    @st.cache_data
    def get_data(ticker, start_date, end_date):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data

    def relativeret(df):
        rel = df.pct_change()
        rel = rel.fillna(0)
        cumret = rel
        return cumret


    # Use the get_data function to fetch the data for selected tickers
    df_data = get_data(selected_tickers, start_date, end_date)
    
    
    if len(selected_tickers) > 0:
        df = relativeret(df_data['Adj Close'])
        st.subheader('Returns of {}'.format(", ".join(selected_tickers)))  # Display all selected tickers separated by commas
        st.line_chart(df)
        
        
    # Calculate the average percentage change for the given time range
    avg_pct_changes = relativeret(df_data['Adj Close'])
    
    # Create an empty list to store results
    results = []
    
    # Iterate over each column (ticker) in the DataFrame
    for ticker in avg_pct_changes.columns:
        avg_change = avg_pct_changes[ticker].mean()
        results.append({
            "Ticker": ticker,
            "Average Percentage Change": avg_change*100})
        
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display DataFrame in Streamlit
    st.table(results_df)

 #==============================================================================
 # Main body
 #==============================================================================
   
# Render the header
render_header()

# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financial Statements", "Monte Carlo simulation", "Own Analysis" ])

with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################
