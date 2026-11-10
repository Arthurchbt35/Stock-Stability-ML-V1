# Aegis Project — Stock Market Stability Study (V1)

Aegis Project is a Machine Learning project built to predict stable stock investments ("compounders") over a 1-year horizon. It combines official company balance sheets from the SEC (10-K filings) with historical daily stock prices from Yahoo Finance.

## 1. How the Dataset is Built

1. **SEC Financial Data :** The script download data from sec.gov archives files from 2010 to today, convert them to Parquet format to reduce their wheight, and uses DuckDB to extract 6 accounting values for each annual reports.
- Assets, Liabilities, Revenues, OperatingIncome, NetIncome, and OperatingCash.

2. **YFinance :** After matching SEC CIK numbers with stock Tickers, the pipeline downloads daily closing prices and cleans empty rows, keeping ~7 million valid price rows.

3. **Clamar Ratio** : For each financial report, we look at the stock price over the following 365 days and compute a stability score:

    $$\text{Score} = \frac{\text{1-Year Return}}{\max(\vert{}\text{Max Drawdown}\vert{}, 0.01)}$$

