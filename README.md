# Aegis Project - Stock Market Stability Study (V1)

Aegis Project is a Machine Learning project built to predict stable stock investments ("compounders") over a 1-year horizon. It combines official company balance sheets from the SEC (10-K filings) with historical daily stock prices from Yahoo Finance.

## 1. How the Dataset is Built

1. **SEC Financial Data :** The script downloads data from sec.gov archive files from 2010 to today, converts them to Parquet format to reduce their weight, and uses DuckDB to extract 6 accounting values for each annual report.
- Assets, Liabilities, Revenues, OperatingIncome, NetIncome, and OperatingCash.

2. **YFinance :** After matching SEC CIK numbers with stock Tickers, the pipeline downloads daily closing prices and cleans empty rows, keeping ~7 million valid price rows.

3. **Calmar Ratio** : For each financial report, we look at the stock price over the following 365 days and compute a stability score:

    $$\text{Score} = \frac{\text{1-Year Return}}{\max(\vert{}\text{Max Drawdown}\vert{}, 0.01)}$$

## 2. Why did I choose a Binary Classification instead of a Linear Regression

When analyzing the 10 098 valid financial reports, the target score showed massive outliers :
- Median score : 0.09
- Mean score : 7.20 (Distorted by outliers)
- Min / Max score : -59.02 / 39 999.59
- 90% of stocks : score below 3.45

Standard regression models do not handle extreme outliers well because when a few stocks explode to 39,999, it completely distorts predictions for normal companies

**Binary Classification** problem  :
- 1 (Compounder - 30% of the data): Score $\ge 1.0$ (the annual return is higher than the worst crash of the year).
- 0 (Underperform - 70% of the data): Score $< 1.0$.

## 3. Models & Visuals V1

The data is split chronologically 80% training (before March 2023) 20% test (recent data).

**Baseline 1 :**

Started with a simple Decision Tree (max_depth=4) to visualize how the  algorithm splits the data :

Looking at the feature importance of a single tree, Revenues and OperatingCash do almost all the work, while Liabilities (debt) is at $0\%$:

**Baseline 2 :**

Using a Random Forest (n_estimators=100, max_depth=6) forces the model to look at all columns across different trees. Cash flow (OperatingCash) and operating profit (OperatingIncome) become the most important indicators, and debt (Liabilities) is finally taken into account:

**Test Set Results:**

*Overall Accuracy:* $\sim 61.34\%$

*Compounder Recall* (Class 1): $54\%$ (the model successfully detects $63\%$ of the good stocks in the test set, but still makes false positives with $35\%$ precision).

## 4. Next : V2 Improvements

Right now, the model only sees raw dollar amounts. Because it cannot divide columns by itself, it mostly filters big companies from small companies (for example, it doesn't know if $\$50\text{M}$ in debt is huge for a small business or tiny for a giant corporation).

1. **Calculations Between Values (Financial Ratios):**
Create new columns by calculating ratios between the existing values to give them real financial meaning regardless of company size:

    - *Profit Margins:*  $\frac{\text{OperatingIncome}}{\text{Revenues}}$ and $\frac{\text{NetIncome}}{\text{Revenues}}$
    - *Debt Ratios:*  $\frac{\text{Liabilities}}{\text{Assets}}$ and $\frac{\text{Liabilities}}{\text{OperatingCash}}$
    - *Return on Assets (ROA):*  $\frac{\text{NetIncome}}{\text{Assets}}$
    - *Year-over-Year Growth:* Compare year $N$ with year $N-1$ to see if revenues and cash are accelerating.

2. **Neural Network with PyTorch:**
Build a custom Neural Network architecture in *PyTorch* trained on these new financial ratios to capture more complex combinations than basic decision trees.

## Files in this Repository

Finances.ipynb: Full pipeline (SEC scraping, DuckDB extraction, Yahoo Finance download, Calmar computation, and ML models).

dataset_aegis_final_ml.parquet: Final processed dataset ($10 281$ scored filings) so the Machine Learning section can be run directly.

.png: Graphical results of the machine learning models.
