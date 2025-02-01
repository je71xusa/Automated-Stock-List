import ftplib
import datetime
import time
import os
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
# Script #1 code
################################################################################

def step1_download_and_filter_ftp():
    """
    1) Connect and login to the FTP server at ftp.nasdaqtrader.com,
       navigate to Symboldirectory, download 'nasdaqlisted.txt' and 'otherlisted.txt'.
    2) Filter each file and create dated versions: e.g. 'nasdaqlisted_22012025.txt' => 'nasdaqlisted_22012025_filtered.txt'
    3) Then create 'TickerInput_<date>.csv' containing all unique tickers.
    """
    print("=== STEP 1: Downloading and filtering from FTP ===")
    ftp_server = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp_server.login()
    ftp_server.encoding = "utf-8"
    ftp_server.cwd("Symboldirectory")
    filenames = ["nasdaqlisted.txt", "otherlisted.txt"]
    today = datetime.date.today()
    date_str = today.strftime("%d%m%Y")
    filtered_files = []

    for filename in filenames:
        dated_filename = filename.replace(".txt", f"_{date_str}.txt")
        with open(dated_filename, "wb") as local_file:
            ftp_server.retrbinary(f"RETR {filename}", local_file.write)
        base_name, ext = os.path.splitext(dated_filename)
        filtered_filename = f"{base_name}_filtered{ext}"
        with open(dated_filename, "r", encoding="utf-8") as infile, \
             open(filtered_filename, "w", encoding="utf-8") as outfile:
            lines = infile.readlines()
            header = lines[0]
            outfile.write(header)
            header_cols = header.strip().split("|")
            etf_index = header_cols.index("ETF")
            test_index = header_cols.index("Test Issue")
            if filename == "nasdaqlisted.txt":
                fin_index = header_cols.index("Financial Status")
            for line in lines[1:]:
                row = line.strip().split("|")
                if row[etf_index] == "N" and row[test_index] == "N":
                    if filename == "nasdaqlisted.txt":
                        if row[fin_index] == "N":
                            outfile.write(line)
                    else:
                        outfile.write(line)
        filtered_files.append(filtered_filename)

    ftp_server.quit()

    unique_tickers = set()
    for ffile in filtered_files:
        with open(ffile, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                row = line.strip().split("|")
                ticker_symbol = row[0]
                unique_tickers.add(ticker_symbol)
    ticker_input_filename = f"TickerInput_{date_str}.csv"
    with open(ticker_input_filename, "w", encoding="utf-8") as outfile:
        outfile.write(",".join(sorted(unique_tickers)))
    print("Done. Created filtered files and combined ticker CSV:", ticker_input_filename)
    print("========================================\n")


################################################################################
# Script #4 code
################################################################################

def compute_RSI(data, window=14, column='Close'):
    """
    Computes RSI using a simple average gain/loss method.
    Returns a pandas Series of RSI values.
    """
    delta = data[column].diff()
    gains = delta.clip(lower=0)
    losses = -1 * delta.clip(upper=0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def round_or_none(val, decimals=2):
    """Round val to 'decimals' places if it's a real number, else None."""
    if val is None or pd.isna(val):
        return None
    try:
        return round(float(val), decimals)
    except:
        return None


def get_history_with_retries(ticker_obj, period="1y", max_retries=3, base_delay=2):
    """
    Attempts to get historical data from a yf.Ticker object with retries.

    Args:
        ticker_obj: A yfinance.Ticker object.
        period (str): Data period to request.
        max_retries (int): Maximum number of retry attempts.
        base_delay (int): Base delay in seconds for exponential backoff.

    Returns:
        pandas.DataFrame: The historical data.

    Raises:
        Exception: If maximum retries are reached without success.
    """
    for attempt in range(max_retries):
        try:
            hist = ticker_obj.history(period=period)
            if hist.empty:
                raise ValueError("Empty DataFrame returned")
            return hist
        except Exception as e:
            error_str = str(e)

            delay = base_delay * (2 ** attempt)
            print(f"Received {error_str} for {ticker_obj.ticker}. "
                  f"Sleeping for {delay} seconds before retrying (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)

    raise Exception(f"Max retries reached for ticker {ticker_obj.ticker}. Skipping.")


def step2_yfinance_data():
    """
    1) Reads TickerInput_<date>.csv
    2) For each ticker, downloads ~1 year of data from Yahoo Finance, extracts
       current price, 6M high/low, RSI(14), plus fundamentals (P/B, P/S, P/E, TargetPrice, Sector).
    3) Saves to RawData_<date>.csv
    """
    print("=== STEP 2: Downloading data from yfinance ===")
    today = datetime.date.today()
    date_str = today.strftime("%d%m%Y")
    ticker_input_filename = f"TickerInput_{date_str}.csv"
    if not os.path.exists(ticker_input_filename):
        print(f"Ticker input file '{ticker_input_filename}' not found. Skipping Step 2.")
        print("========================================\n")
        return

    with open(ticker_input_filename, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    tickers = line.split(",")

    records = []
    six_months_ago = pd.Timestamp.now(tz=None) - pd.Timedelta(days=180)

    for symbol in tickers:
        symbol = symbol.strip().upper()
        if not symbol:
            continue

        print(f"Processing ticker: {symbol}")
        try:
            ticker_obj = yf.Ticker(symbol)

            # Use the retry function instead of directly calling history()
            hist_1y = get_history_with_retries(ticker_obj, period="1y")

            # Adjust timezone if necessary
            if hist_1y.index.tz is not None:
                hist_1y.index = hist_1y.index.tz_convert(None)

            current_price = hist_1y["Close"].iloc[-1]

            hist_6m = hist_1y[hist_1y.index >= six_months_ago]
            if hist_6m.empty:
                print(f"  No 6-month data found for {symbol}. Skipping...")
                continue

            highest_6m = hist_6m["High"].max()
            lowest_6m = hist_6m["Low"].min()

            # Compute RSI
            hist_1y["RSI14"] = compute_RSI(hist_1y, window=14, column="Close")
            rsi_14 = hist_1y["RSI14"].iloc[-1]
            if pd.isna(rsi_14):
                print(f"  RSI(14) is NaN for {symbol}. Possibly not enough data. Skipping...")
                continue

            info = ticker_obj.info
            pb = info.get("priceToBook", None)
            ps = info.get("priceToSalesTrailing12Months", None)
            pe = info.get("trailingPE", None)
            target_price = info.get("targetMeanPrice", None)
            sector = info.get("sector", None)

            record = {
                "Ticker": symbol,
                "CurrentPrice": round_or_none(current_price, 2),
                "High_6M": round_or_none(highest_6m, 2),
                "Low_6M": round_or_none(lowest_6m, 2),
                "RSI14": round_or_none(rsi_14, 2),
                "P/B": round_or_none(pb, 2),
                "P/S": round_or_none(ps, 2),
                "P/E": round_or_none(pe, 2),
                "TargetPrice": round_or_none(target_price, 2),
                "Sector": sector
            }
            records.append(record)

        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            continue

    df = pd.DataFrame(records)
    output_filename = f"RawData_{date_str}.csv"
    if not df.empty:
        df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved data to '{output_filename}'.")
    else:
        print("\nNo valid tickers found. No CSV file was created.")

    print("========================================\n")


################################################################################
# Script #2 code
################################################################################

def step3_filter_raw_data():
    """
    1) Reads RawData_<date>.csv
    2) Filters on:
       - TargetPrice > CurrentPrice
       - P/B, P/S, P/E > -2
       - CurrentPrice > 10
       - RSI14 < 40
    3) Saves FilteredData_<date>.csv
    """
    print("=== STEP 3: Filtering RawData_<date>.csv ===")
    today = datetime.date.today()
    date_str = today.strftime("%d%m%Y")
    raw_csv = f"RawData_{date_str}.csv"
    if not os.path.exists(raw_csv):
        print(f"File '{raw_csv}' not found. Skipping Step 3.")
        print("========================================\n")
        return
    df = pd.read_csv(raw_csv)
    required_cols = ["Ticker", "CurrentPrice", "TargetPrice", "P/B", "P/S", "P/E", "RSI14"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in '{raw_csv}'. Cannot filter.")
            print("========================================\n")
            return
    condition = (
        (df["TargetPrice"] > df["CurrentPrice"]) &
        (df["P/B"] > -2) &
        (df["P/S"] > -2) &
        (df["P/E"] > -2) &
        (df["CurrentPrice"] > 10) &
        (df["RSI14"] < 40)
    )
    filtered_df = df[condition].copy()
    filtered_csv = f"FilteredData_{date_str}.csv"
    filtered_df.to_csv(filtered_csv, index=False)
    print(f"Filtered data saved to '{filtered_csv}'.")
    print(f"Number of rows after filtering: {len(filtered_df)}")
    print("========================================\n")


################################################################################
# Script #5 code
################################################################################

def step4_plot_histograms():
    """
    Reads RawData_<date>.csv, groups by 'Sector', and plots histograms for P/E, P/S, P/B.
    Saves plots under a 'Plots' subdirectory.
    """
    print("=== STEP 4: Creating histograms by sector from RawData_<date>.csv ===")
    today = datetime.date.today()
    date_str = today.strftime("%d%m%Y")
    csv_filename = f"RawData_{date_str}.csv"
    if not os.path.exists(csv_filename):
        print(f"File '{csv_filename}' not found. Skipping Step 4.")
        print("========================================\n")
        return
    df = pd.read_csv(csv_filename)
    metrics = ["P/E", "P/S", "P/B"]
    sector_column = "Sector"
    plot_dir = "Plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if sector_column not in df.columns:
        print(f"Error: '{sector_column}' column not found in CSV. Skipping histograms.")
        print("========================================\n")
        return
    grouped = df.groupby(sector_column, dropna=True)
    for sector_name, group_df in grouped:
        if pd.isna(sector_name) or str(sector_name).strip() == "":
            continue
        for metric in metrics:
            if metric not in group_df.columns:
                print(f"Warning: '{metric}' column not found. Skipping...")
                continue
            data_raw = group_df[metric]
            data_numeric = pd.to_numeric(data_raw, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if data_numeric.empty:
                print(f"No valid (finite) data for {metric} in sector '{sector_name}'. Skipping...")
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(data_numeric, bins=200, edgecolor="black", alpha=0.7)
            plt.title(f"{metric} Histogram - Sector: {sector_name}")
            plt.xlabel(metric)
            plt.ylabel("Frequency")
            safe_sector_name = str(sector_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
            safe_metric_name = metric.replace("/", "_").replace("\\", "_")
            output_file = os.path.join(plot_dir, f"{safe_sector_name}_{safe_metric_name}_hist.png")
            plt.savefig(output_file, dpi=100)
            plt.close()
            print(f"Saved histogram for {metric} in sector '{sector_name}' to '{output_file}'")
    print("========================================\n")


################################################################################
# Script #3 code (Processing AllMetricsRaw if present)
################################################################################

def step5_process_allMetricsRaw():
    """
    Looks for 'AllMetricsRaw_<date>.csv'
    Applies filtering rules on certain metrics and plots side-by-side heatmaps.
    Saves a filtered CSV 'AllMetricsRaw_<date>_FILTERED.csv'.
    """
    print("=== STEP 5: Processing AllMetricsRaw_<date>.csv (if present) ===")
    today = datetime.date.today()
    date_str = today.strftime("%d%m%Y")
    raw_csv_file = f"AllMetricsRaw_{date_str}.csv"
    if not os.path.exists(raw_csv_file):
        print(f"File '{raw_csv_file}' not found. Skipping Step 5.")
        print("========================================\n")
        return
    df_raw = pd.read_csv(raw_csv_file, index_col=0)
    if df_raw.empty:
        print("CSV is empty or invalid. Skipping Step 5.")
        print("========================================\n")
        return
    required_metrics = [
        "PossibleWinning",
        "MA50_vs_MA200",
        "DaysTo_MA50_MA200_Cross",
        "MACD_vs_Signal",
        "DaysTo_MACD_Signal_Cross"
    ]
    for rm in required_metrics:
        if rm not in df_raw.index:
            print(f"Required metric '{rm}' not found in CSV. Aborting Step 5.")
            print("========================================\n")
            return
    df_raw = df_raw.apply(pd.to_numeric, errors="coerce")
    valid_tickers = []
    for ticker in df_raw.columns:
        pw_val = df_raw.loc["PossibleWinning", ticker]
        if pd.isna(pw_val) or pw_val < 1.05:
            continue
        macd_vs_signal = df_raw.loc["MACD_vs_Signal", ticker]
        days_macd = df_raw.loc["DaysTo_MACD_Signal_Cross", ticker]
        macd_below_soon = (macd_vs_signal == -1) and (days_macd < 60)
        if macd_below_soon:
            valid_tickers.append(ticker)
            continue
        ma50_vs_ma200 = df_raw.loc["MA50_vs_MA200", ticker]
        days_ma = df_raw.loc["DaysTo_MA50_MA200_Cross", ticker]
        ma_condition = ((ma50_vs_ma200 == 1) or ((ma50_vs_ma200 == -1) and (days_ma < 60)))
        macd_condition = ((macd_vs_signal == 1) or ((macd_vs_signal == -1) and (days_macd < 60)))
        if ma_condition and macd_condition:
            valid_tickers.append(ticker)
    valid_tickers = list(set(valid_tickers))
    if not valid_tickers:
        print("No tickers pass the filtering conditions in AllMetricsRaw. Exiting Step 5.")
        print("========================================\n")
        return
    df_filtered = df_raw.loc[:, valid_tickers]
    pw_series = df_filtered.loc["PossibleWinning"].dropna()
    if not pw_series.empty:
        sorted_tickers = pw_series.sort_values(ascending=False).index.tolist()
    else:
        sorted_tickers = df_filtered.columns.tolist()
    metrics = df_filtered.index.tolist()
    n_metrics = len(metrics)
    metric_plot_config = {
        "RSI": {"cmap": "Reds_r", "vmin": 0, "vmax": 100},
        "PossibleWinning": {"cmap": "Reds"},
        "MA50_vs_MA200": {"cmap": "Reds", "vmin": -1, "vmax": 1},
        "DaysTo_MA50_MA200_Cross": {"cmap": "Reds_r"},
        "MACD_vs_Signal": {"cmap": "Reds", "vmin": -1, "vmax": 1},
        "DaysTo_MACD_Signal_Cross": {"cmap": "Reds_r"},
        "P/B": {"cmap": "Reds_r"},
        "P/S": {"cmap": "Reds_r"},
        "P/E": {"cmap": "Reds_r"},
        "Volume_Change_20d": {"cmap": "Reds"},
        "Short_Interest_Ratio": {"cmap": "Reds"},
        "Float_Short_Pct": {"cmap": "Reds"},
        "DividendYield_Pct": {"cmap": "Reds"},
        "DaysToExDiv": {"cmap": "Reds_r"}
    }
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_metrics,
        figsize=(2.0 * n_metrics, max(6, 0.4 * len(sorted_tickers)))
    )
    if n_metrics == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data_series = df_filtered.loc[metric].reindex(sorted_tickers)
        mini_df = pd.DataFrame(data_series, columns=[metric])
        cfg = metric_plot_config.get(metric, {})
        cmap = cfg.get("cmap", "Reds")
        vmin = cfg.get("vmin", None)
        vmax = cfg.get("vmax", None)
        sns.heatmap(
            mini_df,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            ax=ax,
            cbar=False
        )
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Ticker")
    plt.tight_layout()
    out_fig = f"FilteredSubplots_{date_str}.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Done. Filtered plot saved to '{out_fig}'.")
    filtered_csv_file = f"AllMetricsRaw_{date_str}_FILTERED.csv"
    df_filtered.to_csv(filtered_csv_file)
    print(f"Filtered data saved to '{filtered_csv_file}'.")
    print("========================================\n")


################################################################################
# Additional Helper Functions
################################################################################

def compute_rsi(series_close, window=14):
    """
    Compute RSI using a simple average gain/loss method.
    Returns a Pandas Series of RSI values.
    """
    delta = series_close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_macd_and_signal(series_close, fast=12, slow=26, signal=9):
    """
    Calculate MACD (fast EMA - slow EMA) and signal line (EMA of MACD).
    Returns two Pandas Series: (macd, macd_signal).
    """
    ema_fast = series_close.ewm(span=fast, adjust=False).mean()
    ema_slow = series_close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def days_to_cross_zero(current_diff, last_diffs):
    """
    Estimate how many days until 'current_diff' hits zero,
    based on the average daily change in 'last_diffs'.
    If the slope is pushing away from zero or is zero, return NaN.
    """
    if len(last_diffs) < 2:
        return np.nan
    diffs = np.diff(last_diffs)
    avg_slope = np.mean(diffs)
    if avg_slope == 0:
        return np.nan
    if (current_diff > 0 and avg_slope > 0) or (current_diff < 0 and avg_slope < 0):
        return np.nan
    return abs(current_diff) / abs(avg_slope)


################################################################################
# New Step: Compute AllMetricsRaw from FilteredData CSV
################################################################################

def step_compute_all_metrics_raw():
    """
    Computes additional technical metrics from FilteredData_<date>.csv
    and saves them to AllMetricsRaw_<date>.csv.
    """
    print("=== STEP X: Computing all metrics from FilteredData_<date>.csv ===")
    today = datetime.date.today()
    date_str = today.strftime("%d%m%Y")
    filtered_file = f"FilteredData_{date_str}.csv"
    if not os.path.exists(filtered_file):
        print(f"File '{filtered_file}' not found. Skipping step_compute_all_metrics_raw.")
        print("========================================\n")
        return
    df_filtered = pd.read_csv(filtered_file)
    if "Ticker" not in df_filtered.columns:
        print("No 'Ticker' column in filtered CSV. Skipping step_compute_all_metrics_raw.")
        return
    tickers = df_filtered["Ticker"].dropna().unique().tolist()
    if not tickers:
        print("No tickers found in filtered CSV. Skipping step_compute_all_metrics_raw.")
        return
    metric_list = [
        "PossibleWinning",
        "MA50_vs_MA200",
        "DaysTo_MA50_MA200_Cross",
        "MACD_vs_Signal",
        "DaysTo_MACD_Signal_Cross",
        "RSI",
        "P/B", "P/S", "P/E",
        "Volume_Change_10d",
        "Short_Interest_Ratio",
        "Float_Short_Pct",
        "DividendYield_Pct",
        "DaysToExDiv"
    ]
    master = {m: {} for m in metric_list}
    for sym in tickers:
        sym = sym.upper()
        print(f"Processing {sym}...")
        try:
            ticker_obj = yf.Ticker(sym)
            hist = ticker_obj.history(period="1y")
            if hist.empty:
                print(f"  No 1y data for {sym}, skipping.")
                continue
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            # Compute moving averages
            hist["MA50"] = hist["Close"].rolling(50).mean()
            hist["MA200"] = hist["Close"].rolling(200).mean()
            # Compute RSI using the new helper function
            hist["RSI14"] = compute_rsi(hist["Close"], 14)
            # Compute MACD and signal line
            macd, macd_signal = get_macd_and_signal(hist["Close"])
            hist["MACD"] = macd
            hist["MACD_Signal"] = macd_signal
            latest = hist.iloc[-1]
            current_price = float(latest["Close"])
            # Fundamentals
            info = ticker_obj.info
            target_price = info.get("targetMeanPrice", np.nan)
            pb = info.get("priceToBook", np.nan)
            ps = info.get("priceToSalesTrailing12Months", np.nan)
            pe = info.get("trailingPE", np.nan)
            short_ratio = info.get("shortRatio", np.nan)
            float_short_pct = info.get("sharesPercentSharesOut", np.nan)
            if not pd.isna(float_short_pct):
                float_short_pct *= 100.0
            else:
                float_short_pct = np.nan
            div_yield = info.get("dividendYield", 0.0)
            if not pd.isna(div_yield):
                div_yield *= 100.0
            ex_div_raw = info.get("exDividendDate", None)
            if ex_div_raw:
                try:
                    ex_div_date = pd.to_datetime(ex_div_raw, unit='s', errors='coerce')
                    if pd.isna(ex_div_date):
                        ex_div_date = pd.to_datetime(ex_div_raw, errors='coerce')
                except:
                    ex_div_date = np.nan
            else:
                ex_div_date = np.nan
            if not pd.isna(ex_div_date):
                days_to_ex_div = (ex_div_date.date() - datetime.date.today()).days
                if days_to_ex_div < 0:
                    days_to_ex_div = 300
            else:
                days_to_ex_div = np.nan
            # Compute PossibleWinning
            if (not pd.isna(target_price)) and target_price > 0 and current_price > 0:
                possible_winning = target_price / current_price
            else:
                possible_winning = np.nan
            # Compute MA50 vs MA200
            ma50 = latest.get("MA50", np.nan)
            ma200 = latest.get("MA200", np.nan)
            if pd.isna(ma50) or pd.isna(ma200):
                ma50_vs_ma200 = np.nan
            else:
                ma50_vs_ma200 = 1 if ma50 > ma200 else -1
            # DaysTo_MA50_MA200_Cross
            ma_diff = hist["MA50"] - hist["MA200"]
            if ma_diff.dropna().empty:
                days_to_ma_cross = np.nan
            else:
                current_madiff = ma_diff.iloc[-1]
                last_20_madiffs = ma_diff.dropna().iloc[-20:].values
                days_to_ma_cross = days_to_cross_zero(current_madiff, last_20_madiffs)
            # MACD vs Signal
            macd_val = latest.get("MACD", np.nan)
            macd_sig = latest.get("MACD_Signal", np.nan)
            if pd.isna(macd_val) or pd.isna(macd_sig):
                macd_vs_signal = np.nan
            else:
                macd_vs_signal = 1 if macd_val > macd_sig else -1
            # DaysTo_MACD_Signal_Cross
            macd_diff = hist["MACD"] - hist["MACD_Signal"]
            if macd_diff.dropna().empty:
                days_to_macd_cross = np.nan
            else:
                current_macd_diff = macd_diff.iloc[-1]
                last_20_macd_diffs = macd_diff.dropna().iloc[-20:].values
                days_to_macd_cross = days_to_cross_zero(current_macd_diff, last_20_macd_diffs)
            # RSI from latest row
            raw_rsi = latest.get("RSI14", np.nan)
            # Volume Change 10d
            hist["Volume_10dAgo"] = hist["Volume"].shift(10)
            vol_10d_ago = hist["Volume_10dAgo"].iloc[-1] if not hist["Volume_10dAgo"].dropna().empty else np.nan
            if pd.isna(vol_10d_ago) or vol_10d_ago == 0:
                vol_change_10d = np.nan
            else:
                vol_change_10d = latest["Volume"] / vol_10d_ago
            master["PossibleWinning"][sym]         = possible_winning
            master["MA50_vs_MA200"][sym]           = ma50_vs_ma200
            master["DaysTo_MA50_MA200_Cross"][sym] = days_to_ma_cross
            master["MACD_vs_Signal"][sym]          = macd_vs_signal
            master["DaysTo_MACD_Signal_Cross"][sym]= days_to_macd_cross
            master["RSI"][sym]                     = raw_rsi
            master["P/B"][sym]                     = pb
            master["P/S"][sym]                     = ps
            master["P/E"][sym]                     = pe
            master["Volume_Change_10d"][sym]       = vol_change_10d
            master["Short_Interest_Ratio"][sym]    = short_ratio
            master["Float_Short_Pct"][sym]         = float_short_pct
            master["DividendYield_Pct"][sym]       = div_yield
            master["DaysToExDiv"][sym]             = days_to_ex_div
        except Exception as exc:
            print(f"Error processing {sym}: {exc}")
            continue
    df_raw = pd.DataFrame(master).transpose()
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    df_raw.dropna(how='all', axis=1, inplace=True)
    raw_outfile = f"AllMetricsRaw_{date_str}.csv"
    df_raw.to_csv(raw_outfile)
    print(f"Saved raw metrics to '{raw_outfile}'.")
    print("========================================\n")


################################################################################
# Combine everything in one main() function
################################################################################

def main():
    # 1) Download from FTP + filter => TickerInput_<date>.csv
    #step1_download_and_filter_ftp()
    # 2) Use yfinance to create RawData_<date>.csv
    step2_yfinance_data()
    # 3) Filter the RawData_<date>.csv => FilteredData_<date>.csv
    step3_filter_raw_data()
    # 4) Create histograms (P/E, P/S, P/B) from RawData_<date>.csv => saved in 'Plots' dir
    step4_plot_histograms()
    # 5) NEW STEP: Compute AllMetricsRaw_<date>.csv from FilteredData_<date>.csv
    step_compute_all_metrics_raw()
    # 6) Process AllMetricsRaw_<date>.csv (if you have it) => side-by-side heatmap + CSV
    step5_process_allMetricsRaw()

if __name__ == "__main__":
    main()
