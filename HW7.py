#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:34:02 2025

@author: allanmccarthy
"""



import pandas as pd
import numpy as np
import statsmodels.api as sm
from cvxopt.solvers import qp
from cvxopt import matrix
import altair as alt

# Suppress cvxopt output for cleaner results
from cvxopt import solvers
solvers.options['show_progress'] = False


import warnings
warnings.filterwarnings("ignore")





ADF = pd.read_csv('/Users/allanmccarthy/Desktop/Alanis/Additional_factors.csv')
CCM = pd.read_parquet('/Users/allanmccarthy/Desktop/Alanis/CRSP_DATA/CCM.parquet')
FF3 = pd.read_parquet('/Users/allanmccarthy/Desktop/Alanis/ThreeFactors.parquet')

# ============================================================================
# QUESTION 2: Smart Beta Portfolio with Momentum Tilt (2005-2024)
# ============================================================================

print("\n\n" + "=" * 70)
print("Starting Question 2: Momentum Long-Short Portfolio (2005-2024)")
print("=" * 70)

# ============================================================================
# Step 2(a): Data Preparation and FF4 Model Setup
# ============================================================================

print("\n2(a). Preparing data and setting up FF4 model...")

# FF3 already has Mom, so we can use it directly as FF4
print(f"FF3 columns: {FF3.columns.tolist()}")
print(f"FF3 date range: {FF3['mdate'].min()} to {FF3['mdate'].max()}")

# Make sure datadate is datetime in CCM
if not pd.api.types.is_datetime64_any_dtype(CCM['datadate']):
    CCM['datadate'] = pd.to_datetime(CCM['datadate'])

# Create market cap, lag, and mdate for merging
CCM_q2 = (
    CCM.sort_values(by=['permno', 'datadate'])
    .assign(
        mktCap=lambda df: df['prc'].abs() * df['shrout'],
        mktCap_lag=lambda df: df.groupby('permno')['mktCap'].shift(1),
        mdate=lambda df: df['datadate'].dt.to_period('M')
    )
)

print(f"CCM_q2 now has mdate column")

# Prepare CRSP data with FF4 factors (FF3 + Mom)
mret_q2 = (
    pd.merge(CCM_q2, FF3, on='mdate', how='inner')
    .assign(ret_rf=lambda df: df['ret'] - df['RF'])
)

# Rank by market cap each month
mret_q2['mkt_rank'] = mret_q2.groupby('datadate')['mktCap_lag'].rank(ascending=False, method='first')

print(f"Data range: {mret_q2['datadate'].min()} to {mret_q2['datadate'].max()}")
print(f"Number of unique stocks: {mret_q2['permno'].nunique()}")


def myFF4(DF):
    """Estimate FF4 factor model for a given stock"""
    try:
        reg_model = sm.OLS(
            endog=DF['ret_rf'],
            exog=sm.add_constant(DF[['Mkt-RF', 'SMB', 'HML', 'Mom']])
        ).fit()
        
        estimates = pd.Series({
            'alpha': reg_model.params[0],
            'beta_mkt': reg_model.params[1],
            'beta_smb': reg_model.params[2],
            'beta_hml': reg_model.params[3],
            'beta_mom': reg_model.params[4],
            'alpha_se': reg_model.bse[0],
            'r2': reg_model.rsquared,
            'nobs': reg_model.nobs
        })
        return estimates
    except:
        return pd.Series({
            'alpha': np.nan, 'beta_mkt': np.nan, 'beta_smb': np.nan, 
            'beta_hml': np.nan, 'beta_mom': np.nan, 'alpha_se': np.nan,
            'r2': np.nan, 'nobs': np.nan
        })

# ============================================================================
# Step 2(b) & 2(c): Construct Long-Short Portfolio (2005-2024)
# ============================================================================

print("\n2(b)-(c). Running out-of-sample Long-Short portfolio construction...")

# Storage for out-of-sample returns
oos_returns_q2 = []

# Loop through years 2005 to 2024
for year in range(2005, 2025):
    print(f"\n  Processing year {year}...")
    
    # Select top 500 stocks based on Dec (year-1) market cap
    dec_data = mret_q2[
        (mret_q2['datadate'].dt.year == year - 1) &
        (mret_q2['datadate'].dt.month == 12)
    ]
    
    if len(dec_data) == 0:
        print(f"  No December {year-1} data found, skipping...")
        continue
    
    last_dec_date = dec_data['datadate'].max()
    
    # NEW CODE (CORRECT)
    top500_stocks = (
    dec_data[dec_data['datadate'] == last_dec_date]
    .drop_duplicates(subset='permno')  # Add this line!
    .nsmallest(500, 'mkt_rank')['permno']
    .tolist()
)
    
    if len(top500_stocks) < 100:
        print(f"  Not enough stocks for {year}, skipping...")
        continue
    
    print(f"    Selected {len(top500_stocks)} stocks")
    
    # Get estimation data (up to 60 months prior to January year)
    estimation_end = pd.Timestamp(f'{year}-01-01')
    estimation_start = estimation_end - pd.DateOffset(months=60)
    
    estimation_data = mret_q2[
        (mret_q2['permno'].isin(top500_stocks)) &
        (mret_q2['datadate'] >= estimation_start) &
        (mret_q2['datadate'] < estimation_end)
    ].copy()
    
    if len(estimation_data) == 0:
        print(f"  No estimation data for {year}, skipping...")
        continue
    
    # Keep only stocks with at least 36 observations
    obs_count = estimation_data.groupby('permno')['datadate'].count()
    valid_stocks = obs_count[obs_count >= 36].index.tolist()
    
    if len(valid_stocks) < 100:
        print(f"  Not enough stocks with sufficient data for {year}, skipping...")
        continue
    
    estimation_data = estimation_data[estimation_data['permno'].isin(valid_stocks)]
    
    print(f"    {len(valid_stocks)} stocks with sufficient data")
    
    # Estimate FF4 model
    FF4_reg = (
        estimation_data
        .groupby('permno', group_keys=False)
        .apply(myFF4)
        .reset_index()
    )
    
    FF4_reg = FF4_reg.dropna(subset=['beta_mom'])
    
    if len(FF4_reg) < 100:
        print(f"  Not enough successful regressions for {year}, skipping...")
        continue
    
    print(f"    {len(FF4_reg)} successful regressions")
    
    # Get market cap for value weighting
    mktcap_data = (
        dec_data[dec_data['datadate'] == last_dec_date]
        [['permno', 'mktCap_lag']]
        .rename(columns={'mktCap_lag': 'mktcap'})
    )
    
    FF4_reg = pd.merge(FF4_reg, mktcap_data, on='permno', how='inner')
    
    # Rank by momentum beta
    FF4_reg = FF4_reg.sort_values('beta_mom', ascending=False)
    
    # Top 50 for Long, Bottom 50 for Short
    long_stocks = FF4_reg.head(50).copy()
    short_stocks = FF4_reg.tail(50).copy()
    
    # Value-weight within each portfolio
    long_stocks['weight'] = long_stocks['mktcap'] / long_stocks['mktcap'].sum()
    short_stocks['weight'] = short_stocks['mktcap'] / short_stocks['mktcap'].sum()
    
    # Long portfolio gets +1 weight, Short gets -1 weight (equal dollar long/short)
    long_stocks['portfolio_weight'] = long_stocks['weight'] * 1.0
    short_stocks['portfolio_weight'] = short_stocks['weight'] * -1.0
    
    print(f"    Long: Top 50 stocks, avg beta_mom = {long_stocks['beta_mom'].mean():.4f}")
    print(f"    Short: Bottom 50 stocks, avg beta_mom = {short_stocks['beta_mom'].mean():.4f}")
    
    # Combine portfolios
    portfolio_stocks = pd.concat([
        long_stocks[['permno', 'portfolio_weight']],
        short_stocks[['permno', 'portfolio_weight']]
    ])
    
    # Calculate out-of-sample returns for year t
    oos_data = mret_q2[
        (mret_q2['permno'].isin(portfolio_stocks['permno'])) &
        (mret_q2['datadate'].dt.year == year)
    ].copy()
    
    if len(oos_data) == 0:
        print(f"  No out-of-sample data for {year}, skipping...")
        continue
    
    oos_data = pd.merge(oos_data, portfolio_stocks, on='permno', how='inner')
    
    monthly_returns = (
        oos_data
        .assign(weighted_ret=lambda df: df['portfolio_weight'] * df['ret'])
        .groupby('datadate')
        .agg(
            portfolio_return=('weighted_ret', 'sum'),
            sp500_return=('sprtrn', 'first')
        )
        .reset_index()
    )
    
    oos_returns_q2.append(monthly_returns)
    print(f"    Completed {year}: {len(monthly_returns)} months of returns")


# ============================================================================
# Step 2(d): Plot Performance and Drawdowns
# ============================================================================

if len(oos_returns_q2) == 0:
    print("\nERROR: No out-of-sample returns were generated for Question 2!")
else:
    oos_df_q2 = pd.concat(oos_returns_q2, ignore_index=True)
    oos_df_q2.rename(columns={'datadate': 'date'}, inplace=True)
    
    print(f"\n2(d). Creating performance and drawdown plots...")
    print(f"   Total months: {len(oos_df_q2)}")
    print(f"   Date range: {oos_df_q2['date'].min()} to {oos_df_q2['date'].max()}")
    
    # Convert percentage returns to decimal
    oos_df_q2['portfolio_return_decimal'] = oos_df_q2['portfolio_return'] / 100
    oos_df_q2['sp500_return_decimal'] = oos_df_q2['sp500_return'] / 100
    
    # Calculate cumulative returns
    oos_df_q2['portfolio_cumret'] = (1 + oos_df_q2['portfolio_return_decimal']).cumprod()
    oos_df_q2['sp500_cumret'] = (1 + oos_df_q2['sp500_return_decimal']).cumprod()
    
    # Reshape for plotting
    cumret_plot_data = pd.melt(
        oos_df_q2,
        id_vars='date',
        value_vars=['portfolio_cumret', 'sp500_cumret'],
        var_name='strategy',
        value_name='cumulative_return'
    )
    
    cumret_plot_data['strategy'] = cumret_plot_data['strategy'].replace({
        'portfolio_cumret': 'Long-Short Momentum',
        'sp500_cumret': 'S&P 500'
    })
    
    # Create cumulative returns chart
    cumret_chart_q2 = (
        alt.Chart(cumret_plot_data,
                  title=f'Value of $1 Invested at Start of {oos_df_q2["date"].min().strftime("%Y")} - Q2')
        .mark_line()
        .encode(
            alt.X('date:T').title(None).axis(format='%Y'),
            alt.Y('cumulative_return:Q')
                .title('Value ($)')
                .scale(zero=False)
                .axis(format='$.2f'),
            alt.Color('strategy:N').title(None)
        )
        .properties(width=700, height=400)
    )
    
    cumret_chart_q2.save('q2_cumulative_returns.html')
    print("   Saved to 'q2_cumulative_returns.html'")
    
    # Calculate drawdowns
    oos_df_q2['portfolio_peak'] = oos_df_q2['portfolio_cumret'].cummax()
    oos_df_q2['sp500_peak'] = oos_df_q2['sp500_cumret'].cummax()
    oos_df_q2['portfolio_drawdown'] = (oos_df_q2['portfolio_cumret'] / oos_df_q2['portfolio_peak'] - 1) * 100
    oos_df_q2['sp500_drawdown'] = (oos_df_q2['sp500_cumret'] / oos_df_q2['sp500_peak'] - 1) * 100
    
    # Reshape for plotting
    drawdown_plot_data = pd.melt(
        oos_df_q2,
        id_vars='date',
        value_vars=['portfolio_drawdown', 'sp500_drawdown'],
        var_name='strategy',
        value_name='drawdown'
    )
    
    drawdown_plot_data['strategy'] = drawdown_plot_data['strategy'].replace({
        'portfolio_drawdown': 'Long-Short Momentum',
        'sp500_drawdown': 'S&P 500'
    })
    
    # Create drawdown chart
    drawdown_chart_q2 = (
        alt.Chart(drawdown_plot_data,
                  title='Drawdowns - Q2')
        .mark_line()
        .encode(
            alt.X('date:T').title(None).axis(format='%Y'),
            alt.Y('drawdown:Q')
                .title('Drawdown (%)')
                .scale(zero=False),
            alt.Color('strategy:N').title(None)
        )
        .properties(width=700, height=400)
    )
    
    drawdown_chart_q2.save('q2_drawdowns.html')
    print("   Saved to 'q2_drawdowns.html'")
    
    
    # ============================================================================
    # Step 2(e): Calculate Performance Statistics
    # ============================================================================
    
    print("\n2(e). Calculating performance statistics...")
    
    # Portfolio statistics
    avg_monthly_return = oos_df_q2['portfolio_return'].mean()
    std_dev = oos_df_q2['portfolio_return'].std()
    var_1pct = oos_df_q2['portfolio_return'].quantile(0.01)
    
    # Sharpe ratio (annual risk-free rate = 3%, so monthly = 0.25%)
    monthly_rf = 0.25
    excess_returns = oos_df_q2['portfolio_return'] - monthly_rf
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    
    # S&P 500 statistics
    sp500_avg_return = oos_df_q2['sp500_return'].mean()
    sp500_std = oos_df_q2['sp500_return'].std()
    sp500_var_1pct = oos_df_q2['sp500_return'].quantile(0.01)
    sp500_excess = oos_df_q2['sp500_return'] - monthly_rf
    sp500_sharpe = sp500_excess.mean() / sp500_excess.std() * np.sqrt(12)
    
    print("\n" + "=" * 70)
    print("PERFORMANCE STATISTICS - QUESTION 2")
    print("=" * 70)
    print("\nLong-Short Momentum Portfolio:")
    print(f"  Average Monthly Return: {avg_monthly_return:.4f}%")
    print(f"  Standard Deviation:     {std_dev:.4f}%")
    print(f"  1% VaR:                 {var_1pct:.4f}%")
    print(f"  Sharpe Ratio (Annual):  {sharpe_ratio:.4f}")
    
    print("\nS&P 500:")
    print(f"  Average Monthly Return: {sp500_avg_return:.4f}%")
    print(f"  Standard Deviation:     {sp500_std:.4f}%")
    print(f"  1% VaR:                 {sp500_var_1pct:.4f}%")
    print(f"  Sharpe Ratio (Annual):  {sp500_sharpe:.4f}")
    print("=" * 70)
    
    
    # ============================================================================
    # Step 2(f): Compute FF4 Model on Out-of-Sample Portfolio
    # ============================================================================
    
    print("\n2(f). Computing FF4 model on out-of-sample portfolio...")
    
    # Create mdate column in oos_df_q2 for merging
    oos_df_q2['mdate'] = oos_df_q2['date'].dt.to_period('M')
    
    # Merge portfolio returns with factors
    portfolio_analysis = pd.merge(
        oos_df_q2[['date', 'mdate', 'portfolio_return']],
        FF3[['mdate', 'Mkt-RF', 'SMB', 'HML', 'Mom', 'RF']],
        on='mdate',
        how='inner'
    )
    
    # Calculate excess returns
    portfolio_analysis['portfolio_excess'] = portfolio_analysis['portfolio_return'] - portfolio_analysis['RF']
    
    # Run FF4 regression on portfolio
    try:
        ff4_portfolio_model = sm.OLS(
            endog=portfolio_analysis['portfolio_excess'],
            exog=sm.add_constant(portfolio_analysis[['Mkt-RF', 'SMB', 'HML', 'Mom']])
        ).fit()
        
        print("\nFF4 Regression Results on Out-of-Sample Portfolio:")
        print(f"  Alpha:       {ff4_portfolio_model.params[0]:.6f} (t-stat: {ff4_portfolio_model.tvalues[0]:.4f})")
        print(f"  Beta (MKT):  {ff4_portfolio_model.params[1]:.6f} (t-stat: {ff4_portfolio_model.tvalues[1]:.4f})")
        print(f"  Beta (SMB):  {ff4_portfolio_model.params[2]:.6f} (t-stat: {ff4_portfolio_model.tvalues[2]:.4f})")
        print(f"  Beta (HML):  {ff4_portfolio_model.params[3]:.6f} (t-stat: {ff4_portfolio_model.tvalues[3]:.4f})")
        print(f"  Beta (MOM):  {ff4_portfolio_model.params[4]:.6f} (t-stat: {ff4_portfolio_model.tvalues[4]:.4f})")
        print(f"  R-squared:   {ff4_portfolio_model.rsquared:.6f}")
        print(f"\nActual MOM exposure: {ff4_portfolio_model.params[4]:.6f}")
        
    except Exception as e:
        print(f"  ERROR in FF4 regression: {e}")
    
    print("\n" + "=" * 70)
    print("QUESTION 2 COMPLETE")
    print("=" * 70)
    
    
    
    
    














    
    
# ============================================================================
# QUESTION 3: Momentum Portfolio Variations
# ============================================================================

print("\n\n" + "=" * 70)
print("Starting Question 3: Long-Only Momentum Portfolios (2005-2024)")
print("=" * 70)

# ============================================================================
# Question 3.1: Long-Only Portfolio (Value-Weighted)
# ============================================================================

print("\n3.1. Long-Only Portfolio (Value-Weighted)...")

# Storage for out-of-sample returns
oos_returns_q31 = []

# Loop through years 2005 to 2024
for year in range(2005, 2025):
    print(f"\n  Processing year {year}...")
    
    # Select top 500 stocks based on Dec (year-1) market cap
    dec_data = mret_q2[
        (mret_q2['datadate'].dt.year == year - 1) &
        (mret_q2['datadate'].dt.month == 12)
    ]
    
    if len(dec_data) == 0:
        print(f"  No December {year-1} data found, skipping...")
        continue
    
    last_dec_date = dec_data['datadate'].max()
    
    # NEW CODE (CORRECT)
    top500_stocks = (
    dec_data[dec_data['datadate'] == last_dec_date]
    .drop_duplicates(subset='permno')  # Add this line!
    .nsmallest(500, 'mkt_rank')['permno']
    .tolist()
)
    
    if len(top500_stocks) < 100:
        print(f"  Not enough stocks for {year}, skipping...")
        continue
    
    # Get estimation data
    estimation_end = pd.Timestamp(f'{year}-01-01')
    estimation_start = estimation_end - pd.DateOffset(months=60)
    
    estimation_data = mret_q2[
        (mret_q2['permno'].isin(top500_stocks)) &
        (mret_q2['datadate'] >= estimation_start) &
        (mret_q2['datadate'] < estimation_end)
    ].copy()
    
    if len(estimation_data) == 0:
        continue
    
    # Keep only stocks with at least 36 observations
    obs_count = estimation_data.groupby('permno')['datadate'].count()
    valid_stocks = obs_count[obs_count >= 36].index.tolist()
    
    if len(valid_stocks) < 50:
        continue
    
    estimation_data = estimation_data[estimation_data['permno'].isin(valid_stocks)]
    
    # Estimate FF4 model
    FF4_reg = (
        estimation_data
        .groupby('permno', group_keys=False)
        .apply(myFF4)
        .reset_index()
    )
    
    FF4_reg = FF4_reg.dropna(subset=['beta_mom'])
    
    if len(FF4_reg) < 50:
        continue
    
    # Get market cap for value weighting
    mktcap_data = (
        dec_data[dec_data['datadate'] == last_dec_date]
        [['permno', 'mktCap_lag']]
        .rename(columns={'mktCap_lag': 'mktcap'})
    )
    
    FF4_reg = pd.merge(FF4_reg, mktcap_data, on='permno', how='inner')
    
    # Rank by momentum beta and select top 50
    FF4_reg = FF4_reg.sort_values('beta_mom', ascending=False)
    long_stocks = FF4_reg.head(50).copy()
    
    # Value-weight the portfolio
    long_stocks['weight'] = long_stocks['mktcap'] / long_stocks['mktcap'].sum()
    
    print(f"    Long-Only: Top 50 stocks, avg beta_mom = {long_stocks['beta_mom'].mean():.4f}")
    
    portfolio_stocks = long_stocks[['permno', 'weight']]
    
    # Calculate out-of-sample returns for year t
    oos_data = mret_q2[
        (mret_q2['permno'].isin(portfolio_stocks['permno'])) &
        (mret_q2['datadate'].dt.year == year)
    ].copy()
    
    if len(oos_data) == 0:
        continue
    
    oos_data = pd.merge(oos_data, portfolio_stocks, on='permno', how='inner')
    
    monthly_returns = (
        oos_data
        .assign(weighted_ret=lambda df: df['weight'] * df['ret'])
        .groupby('datadate')
        .agg(
            portfolio_return=('weighted_ret', 'sum'),
            sp500_return=('sprtrn', 'first')
        )
        .reset_index()
    )
    
    oos_returns_q31.append(monthly_returns)
    print(f"    Completed {year}: {len(monthly_returns)} months")


# ============================================================================
# Question 3.2: Long-Only Portfolio (Momentum Beta-Weighted)
# ============================================================================

print("\n3.2. Long-Only Portfolio (Momentum Beta-Weighted)...")

# Storage for out-of-sample returns
oos_returns_q32 = []

# Loop through years 2005 to 2024
for year in range(2005, 2025):
    print(f"\n  Processing year {year}...")
    
    # Select top 500 stocks based on Dec (year-1) market cap
    dec_data = mret_q2[
        (mret_q2['datadate'].dt.year == year - 1) &
        (mret_q2['datadate'].dt.month == 12)
    ]
    
    if len(dec_data) == 0:
        print(f"  No December {year-1} data found, skipping...")
        continue
    
    last_dec_date = dec_data['datadate'].max()
    
    # NEW CODE (CORRECT)
    top500_stocks = (
    dec_data[dec_data['datadate'] == last_dec_date]
    .drop_duplicates(subset='permno')  # Add this line!
    .nsmallest(500, 'mkt_rank')['permno']
    .tolist()
)
    
    if len(top500_stocks) < 100:
        continue
    
    # Get estimation data
    estimation_end = pd.Timestamp(f'{year}-01-01')
    estimation_start = estimation_end - pd.DateOffset(months=60)
    
    estimation_data = mret_q2[
        (mret_q2['permno'].isin(top500_stocks)) &
        (mret_q2['datadate'] >= estimation_start) &
        (mret_q2['datadate'] < estimation_end)
    ].copy()
    
    if len(estimation_data) == 0:
        continue
    
    # Keep only stocks with at least 36 observations
    obs_count = estimation_data.groupby('permno')['datadate'].count()
    valid_stocks = obs_count[obs_count >= 36].index.tolist()
    
    if len(valid_stocks) < 50:
        continue
    
    estimation_data = estimation_data[estimation_data['permno'].isin(valid_stocks)]
    
    # Estimate FF4 model
    FF4_reg = (
        estimation_data
        .groupby('permno', group_keys=False)
        .apply(myFF4)
        .reset_index()
    )
    
    FF4_reg = FF4_reg.dropna(subset=['beta_mom'])
    
    if len(FF4_reg) < 50:
        continue
    
    # Rank by momentum beta and select top 50
    FF4_reg = FF4_reg.sort_values('beta_mom', ascending=False)
    long_stocks = FF4_reg.head(50).copy()
    
    # Weight by momentum beta (normalize to sum to 1)
    long_stocks['weight'] = long_stocks['beta_mom'] / long_stocks['beta_mom'].sum()
    
    print(f"    Beta-Weighted: Top 50 stocks, avg beta_mom = {long_stocks['beta_mom'].mean():.4f}")
    print(f"    Weight range: {long_stocks['weight'].min():.4f} to {long_stocks['weight'].max():.4f}")
    
    portfolio_stocks = long_stocks[['permno', 'weight']]
    
    # Calculate out-of-sample returns for year t
    oos_data = mret_q2[
        (mret_q2['permno'].isin(portfolio_stocks['permno'])) &
        (mret_q2['datadate'].dt.year == year)
    ].copy()
    
    if len(oos_data) == 0:
        continue
    
    oos_data = pd.merge(oos_data, portfolio_stocks, on='permno', how='inner')
    
    monthly_returns = (
        oos_data
        .assign(weighted_ret=lambda df: df['weight'] * df['ret'])
        .groupby('datadate')
        .agg(
            portfolio_return=('weighted_ret', 'sum'),
            sp500_return=('sprtrn', 'first')
        )
        .reset_index()
    )
    
    oos_returns_q32.append(monthly_returns)
    print(f"    Completed {year}: {len(monthly_returns)} months")


# ============================================================================
# Combine Results and Create Comparison Plots
# ============================================================================

if len(oos_returns_q31) > 0 and len(oos_returns_q32) > 0:
    oos_df_q31 = pd.concat(oos_returns_q31, ignore_index=True)
    oos_df_q31.rename(columns={'datadate': 'date'}, inplace=True)
    
    oos_df_q32 = pd.concat(oos_returns_q32, ignore_index=True)
    oos_df_q32.rename(columns={'datadate': 'date'}, inplace=True)
    
    print(f"\nQuestion 3 data prepared:")
    print(f"  Q3.1 months: {len(oos_df_q31)}")
    print(f"  Q3.2 months: {len(oos_df_q32)}")
    
    # Convert percentage returns to decimal
    oos_df_q31['portfolio_return_decimal'] = oos_df_q31['portfolio_return'] / 100
    oos_df_q31['sp500_return_decimal'] = oos_df_q31['sp500_return'] / 100
    
    oos_df_q32['portfolio_return_decimal'] = oos_df_q32['portfolio_return'] / 100
    oos_df_q32['sp500_return_decimal'] = oos_df_q32['sp500_return'] / 100
    
    # Calculate cumulative returns
    oos_df_q31['portfolio_cumret'] = (1 + oos_df_q31['portfolio_return_decimal']).cumprod()
    oos_df_q31['sp500_cumret'] = (1 + oos_df_q31['sp500_return_decimal']).cumprod()
    
    oos_df_q32['portfolio_cumret'] = (1 + oos_df_q32['portfolio_return_decimal']).cumprod()
    
    # Prepare data for plotting (all three strategies)
    cumret_plot_data_q3 = pd.concat([
        oos_df_q2[['date', 'portfolio_cumret']].assign(strategy='Long-Short (Q2)'),
        oos_df_q31[['date', 'portfolio_cumret']].assign(strategy='Long-Only VW (Q3.1)'),
        oos_df_q32[['date', 'portfolio_cumret']].assign(strategy='Long-Only Beta-Weighted (Q3.2)'),
        oos_df_q31[['date', 'sp500_cumret']].rename(columns={'sp500_cumret': 'portfolio_cumret'}).assign(strategy='S&P 500')
    ], ignore_index=True)
    
    cumret_plot_data_q3.rename(columns={'portfolio_cumret': 'cumulative_return'}, inplace=True)
    
    # Create cumulative returns comparison chart
    cumret_chart_q3 = (
        alt.Chart(cumret_plot_data_q3,
                  title='Value of $1 Invested - All Strategies Comparison')
        .mark_line()
        .encode(
            alt.X('date:T').title(None).axis(format='%Y'),
            alt.Y('cumulative_return:Q')
                .title('Value ($)')
                .scale(zero=False)
                .axis(format='$.2f'),
            alt.Color('strategy:N').title(None)
        )
        .properties(width=700, height=400)
    )
    
    cumret_chart_q3.save('q3_cumulative_returns_comparison.html')
    print("\n   Saved to 'q3_cumulative_returns_comparison.html'")
    
    # Calculate drawdowns for Q3 strategies
    oos_df_q31['portfolio_peak'] = oos_df_q31['portfolio_cumret'].cummax()
    oos_df_q31['portfolio_drawdown'] = (oos_df_q31['portfolio_cumret'] / oos_df_q31['portfolio_peak'] - 1) * 100
    
    oos_df_q32['portfolio_peak'] = oos_df_q32['portfolio_cumret'].cummax()
    oos_df_q32['portfolio_drawdown'] = (oos_df_q32['portfolio_cumret'] / oos_df_q32['portfolio_peak'] - 1) * 100
    
    # Prepare drawdown data for plotting
    drawdown_plot_data_q3 = pd.concat([
        oos_df_q2[['date', 'portfolio_drawdown']].assign(strategy='Long-Short (Q2)'),
        oos_df_q31[['date', 'portfolio_drawdown']].assign(strategy='Long-Only VW (Q3.1)'),
        oos_df_q32[['date', 'portfolio_drawdown']].assign(strategy='Long-Only Beta-Weighted (Q3.2)'),
        oos_df_q31[['date', 'sp500_drawdown']].assign(strategy='S&P 500')
    ], ignore_index=True)
    
    drawdown_plot_data_q3.rename(columns={'portfolio_drawdown': 'drawdown'}, inplace=True)
    
    # Create drawdown comparison chart
    drawdown_chart_q3 = (
        alt.Chart(drawdown_plot_data_q3,
                  title='Drawdowns - All Strategies Comparison')
        .mark_line()
        .encode(
            alt.X('date:T').title(None).axis(format='%Y'),
            alt.Y('drawdown:Q')
                .title('Drawdown (%)')
                .scale(zero=False),
            alt.Color('strategy:N').title(None)
        )
        .properties(width=700, height=400)
    )
    
    drawdown_chart_q3.save('q3_drawdowns_comparison.html')
    print("   Saved to 'q3_drawdowns_comparison.html'")
    
    
    # ============================================================================
    # Calculate Performance Statistics for All Strategies
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("PERFORMANCE STATISTICS - ALL STRATEGIES")
    print("=" * 70)
    
    def calc_stats(df, name):
        """Calculate performance statistics"""
        monthly_rf = 0.25
        avg_ret = df['portfolio_return'].mean()
        std = df['portfolio_return'].std()
        var_1 = df['portfolio_return'].quantile(0.01)
        excess = df['portfolio_return'] - monthly_rf
        sharpe = excess.mean() / excess.std() * np.sqrt(12)
        
        print(f"\n{name}:")
        print(f"  Average Monthly Return: {avg_ret:.4f}%")
        print(f"  Standard Deviation:     {std:.4f}%")
        print(f"  1% VaR:                 {var_1:.4f}%")
        print(f"  Sharpe Ratio (Annual):  {sharpe:.4f}")
        print(f"  Final Value:            ${df['portfolio_cumret'].iloc[-1]:.2f}")
    
    calc_stats(oos_df_q2, "Q2: Long-Short Momentum")
    calc_stats(oos_df_q31, "Q3.1: Long-Only (Value-Weighted)")
    calc_stats(oos_df_q32, "Q3.2: Long-Only (Beta-Weighted)")
    
    # S&P 500 stats
    monthly_rf = 0.25
    sp_avg = oos_df_q31['sp500_return'].mean()
    sp_std = oos_df_q31['sp500_return'].std()
    sp_var = oos_df_q31['sp500_return'].quantile(0.01)
    sp_excess = oos_df_q31['sp500_return'] - monthly_rf
    sp_sharpe = sp_excess.mean() / sp_excess.std() * np.sqrt(12)
    
    print(f"\nS&P 500:")
    print(f"  Average Monthly Return: {sp_avg:.4f}%")
    print(f"  Standard Deviation:     {sp_std:.4f}%")
    print(f"  1% VaR:                 {sp_var:.4f}%")
    print(f"  Sharpe Ratio (Annual):  {sp_sharpe:.4f}")
    print(f"  Final Value:            ${oos_df_q31['sp500_cumret'].iloc[-1]:.2f}")
    
    print("\n" + "=" * 70)
    print("QUESTIONS 2 & 3 COMPLETE")
    print("=" * 70)
    
else:
    print("\nERROR: Failed to generate returns for Question 3!")
    











# ============================================================================
# QUESTION 4: Portfolio Risk Report (2020-2024)
# ============================================================================

print("\n\n" + "=" * 70)
print("Starting Question 4: Portfolio Risk Report (2020-2024)")
print("=" * 70)

# ============================================================================
# Step 4(a): Define Portfolio
# ============================================================================

print("\n4(a). Setting up portfolio...")

# Portfolio composition
portfolio_stocks = pd.DataFrame({
    'company': ['Walmart', 'McDonalds', 'NextEra Energy', 'MetLife', 'Apple'],
    'ticker': ['WMT', 'MCD', 'NEE', 'MET', 'AAPL'],
    'permno': [55976, 43449, 24205, 87842, 14593],
    'weight': [0.30, 0.20, 0.20, 0.15, 0.15]
})

print("\nPortfolio Composition:")
print(portfolio_stocks.to_string(index=False))
print(f"\nTotal weight: {portfolio_stocks['weight'].sum():.2f}")


# ============================================================================
# Step 4(b): Extract Stock Returns (2020-2024)
# ============================================================================

print("\n4(b). Extracting stock returns for 2020-2024...")

# Filter CCM data for our stocks and time period
portfolio_data = CCM[
    (CCM['permno'].isin(portfolio_stocks['permno'])) &
    (CCM['datadate'].dt.year >= 2020) &
    (CCM['datadate'].dt.year <= 2024)
].copy()

print(f"Initial observations: {len(portfolio_data)}")

# Create year-month column for grouping
portfolio_data['year_month'] = portfolio_data['datadate'].dt.to_period('M')

# Keep only the last observation per stock per month (most recent datadate in each month)
portfolio_data = (
    portfolio_data
    .sort_values(['permno', 'datadate'])
    .groupby(['permno', 'year_month'])
    .last()
    .reset_index()
)

print(f"After deduplication: {len(portfolio_data)} observations")
print(f"Unique stocks: {portfolio_data['permno'].nunique()}")
print(f"Date range: {portfolio_data['datadate'].min()} to {portfolio_data['datadate'].max()}")

# Check data availability for each stock
for permno in portfolio_stocks['permno']:
    stock_data = portfolio_data[portfolio_data['permno'] == permno]
    company = portfolio_stocks[portfolio_stocks['permno'] == permno]['company'].values[0]
    print(f"  {company} (permno {permno}): {len(stock_data)} months")

# ============================================================================
# Step 4(c): Calculate Covariance Matrix
# ============================================================================

print("\n4(c). Calculating covariance matrix...")

# Pivot data to wide format (dates x stocks)
returns_wide = pd.pivot(
    portfolio_data,
    columns='permno',
    index='datadate',
    values='ret'
)

# Drop any rows with missing data
returns_wide = returns_wide.dropna()

print(f"Clean returns matrix: {returns_wide.shape[0]} months x {returns_wide.shape[1]} stocks")

# Sort columns by permno to ensure consistent ordering
returns_wide = returns_wide[sorted(returns_wide.columns)]

# Calculate covariance matrix (monthly returns)
cov_matrix = returns_wide.cov()

print("\nCovariance Matrix:")
print(cov_matrix)

# Calculate correlation matrix for reference
corr_matrix = returns_wide.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)


# ============================================================================
# Step 4(d): Calculate Risk Contributions
# ============================================================================

print("\n4(d). Calculating risk contributions...")

# Get weights in the same order as covariance matrix columns
weights_ordered = []
for permno in sorted(portfolio_stocks['permno']):
    w = portfolio_stocks[portfolio_stocks['permno'] == permno]['weight'].values[0]
    weights_ordered.append(w)

W = np.array(weights_ordered)

print(f"Weights (ordered by permno): {W}")
print(f"Sum of weights: {W.sum():.4f}")

# Convert covariance matrix to numpy array
Sigma = cov_matrix.values

# Portfolio variance
port_variance = W.T @ Sigma @ W
port_volatility = np.sqrt(port_variance)

print(f"\nPortfolio Volatility (monthly): {port_volatility:.6f} ({port_volatility*100:.4f}%)")
print(f"Portfolio Volatility (annualized): {port_volatility*np.sqrt(12):.6f} ({port_volatility*np.sqrt(12)*100:.4f}%)")

# Marginal contribution to risk (MCR)
MCR = Sigma @ W / port_volatility

# Risk contribution (RC) = Weight * MCR
RC = W * MCR

# Percent risk contribution
RC_percent = RC / RC.sum()

print("\nRisk Contributions:")
for i, permno in enumerate(sorted(portfolio_stocks['permno'])):
    company = portfolio_stocks[portfolio_stocks['permno'] == permno]['company'].values[0]
    print(f"  {company}: {RC_percent[i]*100:.2f}%")


# ============================================================================
# Step 4(e): Create Risk Report Table
# ============================================================================

print("\n4(e). Creating risk report table...")

# Build the risk report dataframe
risk_report = portfolio_stocks.copy()
risk_report = risk_report.sort_values('permno').reset_index(drop=True)

# Add portfolio metrics
risk_report['Weight'] = W
risk_report['Risk_Contribution'] = RC
risk_report['RC_Percent'] = RC_percent * 100

# Calculate individual stock volatilities
stock_vols = []
for permno in sorted(portfolio_stocks['permno']):
    stock_ret = returns_wide[permno]
    stock_vol = stock_ret.std()
    stock_vols.append(stock_vol)

risk_report['Stock_Volatility'] = stock_vols
risk_report['Ann_Stock_Volatility'] = risk_report['Stock_Volatility'] * np.sqrt(12)

print("\n" + "=" * 70)
print("PORTFOLIO RISK REPORT")
print("=" * 70)
print(f"\nPortfolio Period: January 2020 - December 2024")
print(f"Number of Stocks: {len(risk_report)}")
print(f"Portfolio Volatility (Annual): {port_volatility*np.sqrt(12)*100:.2f}%")
print("\n")

# Format and display the risk report
risk_report_display = risk_report.copy()
risk_report_display['Weight'] = risk_report_display['Weight'].apply(lambda x: f"{x*100:.1f}%")
risk_report_display['RC_Percent'] = risk_report_display['RC_Percent'].apply(lambda x: f"{x:.2f}%")
risk_report_display['Stock_Volatility'] = risk_report_display['Stock_Volatility'].apply(lambda x: f"{x*100:.2f}%")
risk_report_display['Ann_Stock_Volatility'] = risk_report_display['Ann_Stock_Volatility'].apply(lambda x: f"{x*100:.2f}%")

print(risk_report_display[['company', 'ticker', 'permno', 'Weight', 'RC_Percent', 'Ann_Stock_Volatility']].to_string(index=False))


# ============================================================================
# Step 4(f): Calculate Portfolio Performance
# ============================================================================

print("\n\n4(f). Calculating portfolio performance...")

# Merge weights with return data
portfolio_data_weighted = pd.merge(
    portfolio_data,
    portfolio_stocks[['permno', 'weight']],
    on='permno',
    how='inner'
)

# Calculate weighted returns
portfolio_returns = (
    portfolio_data_weighted
    .assign(weighted_ret=lambda df: df['weight'] * df['ret'])
    .groupby('datadate')
    .agg(
        portfolio_return=('weighted_ret', 'sum'),
        sp500_return=('sprtrn', 'first')
    )
    .reset_index()
)

print(f"Portfolio returns calculated: {len(portfolio_returns)} months")

# Calculate performance statistics
avg_monthly_return = portfolio_returns['portfolio_return'].mean()
portfolio_std = portfolio_returns['portfolio_return'].std()
portfolio_var_5 = portfolio_returns['portfolio_return'].quantile(0.05)

# Sharpe ratio (assume 3% annual risk-free rate = 0.25% monthly)
monthly_rf = 0.25
excess_returns = portfolio_returns['portfolio_return'] - monthly_rf
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)

# S&P 500 stats for comparison
sp500_avg = portfolio_returns['sp500_return'].mean()
sp500_std = portfolio_returns['sp500_return'].std()
sp500_excess = portfolio_returns['sp500_return'] - monthly_rf
sp500_sharpe = sp500_excess.mean() / sp500_excess.std() * np.sqrt(12)

print("\n" + "=" * 70)
print("PORTFOLIO PERFORMANCE STATISTICS")
print("=" * 70)
print("\nYour Portfolio:")
print(f"  Average Monthly Return:     {avg_monthly_return:.4f}%")
print(f"  Average Annual Return:      {avg_monthly_return*12:.2f}%")
print(f"  Monthly Volatility:         {portfolio_std:.4f}%")
print(f"  Annual Volatility:          {portfolio_std*np.sqrt(12):.2f}%")
print(f"  5% VaR (monthly):           {portfolio_var_5:.4f}%")
print(f"  Sharpe Ratio (annual):      {sharpe_ratio:.4f}")

print("\nS&P 500 (for comparison):")
print(f"  Average Monthly Return:     {sp500_avg:.4f}%")
print(f"  Average Annual Return:      {sp500_avg*12:.2f}%")
print(f"  Monthly Volatility:         {sp500_std:.4f}%")
print(f"  Annual Volatility:          {sp500_std*np.sqrt(12):.2f}%")
print(f"  Sharpe Ratio (annual):      {sp500_sharpe:.4f}")


# ============================================================================
# Step 4(g): Plot Cumulative Returns
# ============================================================================

print("\n4(g). Creating cumulative returns plot...")

# Calculate cumulative returns
portfolio_returns['portfolio_return_decimal'] = portfolio_returns['portfolio_return'] / 100
portfolio_returns['sp500_return_decimal'] = portfolio_returns['sp500_return'] / 100

portfolio_returns['portfolio_cumret'] = (1 + portfolio_returns['portfolio_return_decimal']).cumprod()
portfolio_returns['sp500_cumret'] = (1 + portfolio_returns['sp500_return_decimal']).cumprod()

# Reshape for plotting
cumret_plot_data = pd.melt(
    portfolio_returns,
    id_vars='datadate',
    value_vars=['portfolio_cumret', 'sp500_cumret'],
    var_name='strategy',
    value_name='cumulative_return'
)

cumret_plot_data['strategy'] = cumret_plot_data['strategy'].replace({
    'portfolio_cumret': 'Your Portfolio',
    'sp500_cumret': 'S&P 500'
})

# Create cumulative returns chart
cumret_chart_q4 = (
    alt.Chart(cumret_plot_data,
              title='Value of $1 Invested (Jan 2020 - Dec 2024) - Q4')
    .mark_line()
    .encode(
        alt.X('datadate:T').title(None).axis(format='%Y'),
        alt.Y('cumulative_return:Q')
            .title('Value ($)')
            .scale(zero=False)
            .axis(format='$.2f'),
        alt.Color('strategy:N').title(None)
    )
    .properties(width=700, height=400)
)

cumret_chart_q4.save('q4_cumulative_returns.html')
print("   Saved to 'q4_cumulative_returns.html'")

# Calculate drawdowns
portfolio_returns['portfolio_peak'] = portfolio_returns['portfolio_cumret'].cummax()
portfolio_returns['sp500_peak'] = portfolio_returns['sp500_cumret'].cummax()
portfolio_returns['portfolio_drawdown'] = (portfolio_returns['portfolio_cumret'] / portfolio_returns['portfolio_peak'] - 1) * 100
portfolio_returns['sp500_drawdown'] = (portfolio_returns['sp500_cumret'] / portfolio_returns['sp500_peak'] - 1) * 100

# Reshape for plotting
drawdown_plot_data = pd.melt(
    portfolio_returns,
    id_vars='datadate',
    value_vars=['portfolio_drawdown', 'sp500_drawdown'],
    var_name='strategy',
    value_name='drawdown'
)

drawdown_plot_data['strategy'] = drawdown_plot_data['strategy'].replace({
    'portfolio_drawdown': 'Your Portfolio',
    'sp500_drawdown': 'S&P 500'
})

# Create drawdown chart
drawdown_chart_q4 = (
    alt.Chart(drawdown_plot_data,
              title='Drawdowns (Jan 2020 - Dec 2024) - Q4')
    .mark_line()
    .encode(
        alt.X('datadate:T').title(None).axis(format='%Y'),
        alt.Y('drawdown:Q')
            .title('Drawdown (%)')
            .scale(zero=False),
        alt.Color('strategy:N').title(None)
    )
    .properties(width=700, height=400)
)

drawdown_chart_q4.save('q4_drawdowns.html')
print("   Saved to 'q4_drawdowns.html'")

print("\n" + "=" * 70)
print(f"Final Portfolio Value: ${portfolio_returns['portfolio_cumret'].iloc[-1]:.2f}")
print(f"Final S&P 500 Value:   ${portfolio_returns['sp500_cumret'].iloc[-1]:.2f}")
print("=" * 70)

print("\n" + "=" * 70)
print("QUESTION 4 COMPLETE")
print("=" * 70)









