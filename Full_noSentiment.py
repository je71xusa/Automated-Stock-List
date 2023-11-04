import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# ---- Part 1: Scraping company list from multiple sources ----

def scrape_index_list(url, column_name):
    tables = pd.read_html(url, header=0)
    print(f"Total tables in {url}: {len(tables)}")  # debugging
    for i, table in enumerate(tables):
        print(f"Columns in the table #{i} from {url}: {table.columns}")  # debugging
        if column_name in table.columns:
            return table[column_name].tolist()
    return []  # Return an empty list if no table has the column

def main_getSymbols(urls):
    # In this function, we will scrape the symbols from multiple sources.
    # The sources are Wikipedia pages which list the symbols of the companies in certain stock market indices.
    symbols = []
    for url, column_name in urls.items():
        symbols.extend(scrape_index_list(url, column_name))
    symbols = list(set(symbols))  # Remove duplicate symbols
    print(f"Found {len(symbols)} unique symbols.")
    return symbols

# ---- Part 2: Getting metrics for each company using yfinance ----

def get_stock_info_yfinance(symbols):
    """Retrieve stock information and historical data for the symbols from Yahoo Finance."""
    stock_info = {}
    periods = ["7d", "1mo"]
    for symbol in symbols:
        print(symbols.index(symbol), "/", len(symbols))
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info:
                print(f"No info available for {symbol}")
                continue
            if 'currentPrice' in info:
                current_price = info['currentPrice']
                stock_info[symbol] = {"current_price": current_price}

                # Get historical data and calculate high and low values for each period
                for period in periods:
                    hist = stock.history(period=period)
                    stock_info[symbol][f"{period}_high"] = hist["High"].max()
                    stock_info[symbol][f"{period}_low"] = hist["Low"].min()
                    stock_info[symbol][f"is_in_low_10pct_{period}"] = current_price <= (hist["Low"].min() + 0.1 * (hist["High"].max() - hist["Low"].min()))
        except Exception as e:
            print(f"Error retrieving info for {symbol}: {e}")
    return stock_info


def filter_stocks(stock_info):
    filtered_stocks = {}
    for symbol, info in stock_info.items():
        try:
            if info['is_in_low_10pct_1mo'] or info['is_in_low_10pct_7d']:
                filtered_stocks[symbol] = info
        except KeyError as e:
            print(f"Error for {symbol}: {e}")
            print(f"Available keys: {info.keys()}")

    return filtered_stocks
def main_getMetrics(symbols):
    data = get_stock_info_yfinance(symbols)
    df = pd.DataFrame(data).T
    df.to_csv('all_stock_data.csv')
    filtered_data = filter_stocks(data)
    df_filtered = pd.DataFrame(filtered_data).T
    df_filtered.to_csv('filtered_stock_data.csv')

# ---- Part 3: Calculating the potential return ----

def calculate_potential_return(df, return_threshold):
    for period in ["7d", "1mo"]:
        df[f"{period}_return_potential"] = df.apply(
            lambda row: (row[f"{period}_high"] - row["current_price"]) / row["current_price"] * 100
            if row[f"is_in_low_10pct_{period}"] else 0, axis=1)
    return df

def filter_stocks_for_return(df, return_threshold):
    df = calculate_potential_return(df, return_threshold)
    df_filtered = df[(df["7d_return_potential"] >= return_threshold) |
                     (df["1mo_return_potential"] >= return_threshold)]
    return df_filtered

def main_calcMetrics():
    df = pd.read_csv('filtered_stock_data.csv', index_col=0)
    return_threshold = 10.0  # We want at least 10% return
    df_filtered = filter_stocks_for_return(df, return_threshold)
    df_filtered.to_csv('undervalued_stocks_with_return_potential.csv')

# ---- Visualization of data ----
def main_visualization():
    df = pd.read_csv('undervalued_stocks_with_return_potential.csv')
    df.set_index('Unnamed: 0', inplace=True)

    # Sort the DataFrame by '1mo_return_potential' and '7d_return_potential'
    df.sort_values(by='1mo_return_potential', ascending=False, inplace=True)
    df_7d_sorted = df.sort_values(by='7d_return_potential', ascending=False)

    n_stocks = len(df)
    cmap = get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n_stocks))

    fig, axs = plt.subplots(2, 1, figsize=(14, 30), sharex=True)

    df['1mo_return_potential'].plot(kind='bar', ax=axs[0], color=colors)
    axs[0].set_title('1-Month Return Potential')
    axs[0].set_ylabel('1-Month Potential Return (%)')

    df_7d_sorted['7d_return_potential'].plot(kind='bar', ax=axs[1], color=colors)
    axs[1].set_title('7-Day Return Potential')
    axs[1].set_ylabel('7-Day Potential Return (%)')

    plt.xlabel('Stocks')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('barplots_sorted.png')

if __name__ == "__main__":
    start_time = time.time()
    urls = {
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies': 'Symbol',
        'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average': 'Symbol',
        'https://en.wikipedia.org/wiki/NASDAQ-100': 'Ticker',
        'https://en.wikipedia.org/wiki/EURO_STOXX_50': 'Ticker'
    }
    symbols = main_getSymbols(urls)
    main_getMetrics(symbols)
    main_calcMetrics()
    main_visualization()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed run time: {elapsed_time/60} minutes")
