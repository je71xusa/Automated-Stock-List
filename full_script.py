import yfinance as yf
import pandas as pd
import numpy as np
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
    # In this function, we retrieve stock information and historical data for the symbols.
    # We are using Yahoo Finance for this.
    data = get_stock_info_yfinance(symbols)
    # Save all data to a CSV file
    df = pd.DataFrame(data).T
    df.to_csv('all_stock_data.csv')
    print("All data saved to all_stock_data.csv")

    # Filter and save selected stocks to a CSV file
    filtered_data = filter_stocks(data)
    df_filtered = pd.DataFrame(filtered_data).T
    df_filtered.to_csv('filtered_stock_data.csv')
    print("Filtered stock data saved to filtered_stock_data.csv")

# ---- Part 3: Calculating the potential return ----

def calculate_potential_return(df, return_threshold):
    """Calculate the potential return for the stocks and add the data to a new column."""
    for period in ["7d", "1mo"]:
        df[f"{period}_return_potential"] = df.apply(
            lambda row: (row[f"{period}_high"] - row["current_price"]) / row["current_price"] * 100
            if row[f"is_in_low_10pct_{period}"] else 0, axis=1)
    return df

def filter_stocks_for_return(df, return_threshold):
    """Filter the stocks that have the potential to give at least a return_threshold return."""
    df = calculate_potential_return(df, return_threshold)
    df_filtered = df[(df["7d_return_potential"] >= return_threshold) |
                     (df["1mo_return_potential"] >= return_threshold)]
    return df_filtered

def main_calcMetrics():
    # In this function, we calculate the potential return for each stock and filter the stocks based on the potential return.
    df = pd.read_csv('filtered_stock_data.csv', index_col=0)
    return_threshold = 10.0  # We want at least 5% return
    df_filtered = filter_stocks_for_return(df, return_threshold)
    df_filtered.to_csv('undervalued_stocks_with_return_potential.csv')
    print("Filtered data saved to undervalued_stocks_with_return_potential.csv")

# ---- Part 4: Fetching the news using NewsAPI and yfinance ----

def fetch_news(api_key, symbol, company_name):
    base_url = 'https://newsapi.org/v2/everything'
    parameters = {
        'q': f"{symbol} OR {company_name}",  # search for both symbol and company name
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': api_key
    }

    response = requests.get(base_url, params=parameters)

    if response.status_code != 200:
        print(f"Failed to fetch news for {symbol} ({company_name}), status code: {response.status_code}")
        return []

    news_data = response.json()
    return news_data['articles']

def get_company_name(symbol):
    ticker = yf.Ticker(symbol)
    return ticker.info['longName']

def main_newsFetch():
    # In this function, we fetch the latest news for each stock using NewsAPI.
    # We are also fetching the long name of the company using yfinance for the news query.
    df = pd.read_csv('undervalued_stocks_with_return_potential.csv')
    api_key = '70d8e15851ec48d18b3bcce727bb0e9b'  # Replace this with your NewsAPI key

    news_data = {}
    for index, row in df.iterrows():
        symbol = row[0]  # Get the symbol from the first column
        company_name = get_company_name(symbol)  # Fetch company name
        articles = fetch_news(api_key, symbol, company_name)
        news_data[symbol] = articles

    with open('stock_news_data.json', 'w') as f:
        json.dump(news_data, f)

# ---- Part 5: Performing sentiment analysis on the news titles ----

def get_sentiment_score(article, analyzer):
    return analyzer.polarity_scores(article['title'])['compound']

def calculate_average_sentiment(sentiments):
    if sentiments:
        return sum(sentiments) / len(sentiments)
    return 0  # Return neutral sentiment if no news articles

def main_sentiment():
    # In this function, we perform sentiment analysis on the news titles for each stock.
    # We then save the average sentiment to the dataframe and export it to a CSV file.
    df = pd.read_csv('undervalued_stocks_with_return_potential.csv')

    with open('stock_news_data.json', 'r') as f:
        news_data = json.load(f)

    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = {}

    for index, row in df.iterrows():
        symbol = row[0]  # Get the symbol from the first column
        articles = news_data.get(symbol, [])
        sentiment_scores = [get_sentiment_score(article, analyzer) for article in articles]
        sentiment_data[symbol] = calculate_average_sentiment(sentiment_scores)

    # Add sentiment data to the DataFrame and save to a new CSV file
    df['average_sentiment'] = df.iloc[:, 0].map(sentiment_data)
    df.to_csv('undervalued_stocks_with_sentiment.csv', index=False)

# ---- Visualization of data ----
def main_visualization():
    df = pd.read_csv('undervalued_stocks_with_sentiment.csv')
    df.rename(columns={'Unnamed: 0': 'Symbol'}, inplace=True)
    df.set_index('Symbol', inplace=True)  # Make sure 'Symbol' is the index of the DataFrame.
    # Making a grouped barplot that is ordered according to sentiment score
    # Order stocks by sentiment
    df.sort_values('average_sentiment', ascending=False, inplace=True)

    # Add sentiment score to the stock name for the plot
    df.index = df.index + ' (' + df['average_sentiment'].round(2).astype(str) + ')'
    # Set color palette
    n_stocks = len(df)
    cmap = get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n_stocks))

    fig, axs = plt.subplots(3, 1, figsize=(14, 50), sharex=True)

    # Plotting 1mo_return_potential
    df['1mo_return_potential'].plot(kind='bar', ax=axs[1], color=colors)
    axs[1].set_title('1-Month Return Potential Sorted by Sentiment Score')
    axs[1].set_ylabel('1-Month Potential Return (%)')

    # Plotting 7d_return_potential
    df['7d_return_potential'].plot(kind='bar', ax=axs[2], color=colors)
    axs[2].set_title('7-Day Return Potential Sorted by Sentiment Score')
    axs[2].set_ylabel('7-Day Potential Return (%)')

    plt.xlabel('Stocks')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('barplots_sorted_by_sentiment.png')

    # In this function, we create a heatmap for the calculated metrics.

    # Normalize data for better visualization
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)
    df.loc[:,:] = scaled_values

    # We will use the '7d_return_potential', '1mo_return_potential', and 'average_sentiment' columns
    metrics_df = df[['7d_return_potential', '1mo_return_potential', 'average_sentiment']]

    # Create a heatmap
    plt.figure(figsize=(14,14))
    sns.heatmap(metrics_df, annot=True, cmap='coolwarm')
    plt.title('Metrics Heatmap')
    plt.savefig('metrics_heatmap.png')  # Save the figure as a PNG image




# ---- The main function to rule them all ----

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
    main_newsFetch()
    main_sentiment()
    main_visualization()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed run time: {elapsed_time/60} minutes")
