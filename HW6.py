#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 18:27:08 2025

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





ADF = pd.read_csv('/Users/allanmccarthy/Desktop/Additional_factors.csv')
CCM = pd.read_parquet('/Users/allanmccarthy/Desktop/Alanis/CRSP_DATA/CCM.parquet')
FF3 = pd.read_parquet('/Users/allanmccarthy/Desktop/Alanis/ThreeFactors.parquet')






# ============================================================================
# QUESTION 1: FF3 Portfolio Optimization (1990-2024)
# ============================================================================

print("Starting Question 1: FF3 Portfolio Optimization (1990-2024)")
print("=" * 70)

# ============================================================================
# Step 1: Data Preparation
# ============================================================================

print("\n1. Preparing data...")

# Make sure datadate is datetime
if not pd.api.types.is_datetime64_any_dtype(CCM['datadate']):
    CCM['datadate'] = pd.to_datetime(CCM['datadate'])

# Create market cap, lag, and mdate for merging
CCM = (
    CCM.sort_values(by=['permno', 'datadate'])
    .assign(
        mktCap=lambda df: df['prc'].abs() * df['shrout'],
        mktCap_lag=lambda df: df.groupby('permno')['mktCap'].shift(1),
        mdate=lambda df: df['datadate'].dt.to_period('M')
    )
)

# Merge CRSP and Fama-French factors by mdate
mret = (
    pd.merge(CCM, FF3, on='mdate', how='inner')
    .assign(ret_rf=lambda df: df['ret'] - df['RF'])
)

# Rank by market cap each month
mret['mkt_rank'] = mret.groupby('datadate')['mktCap_lag'].rank(ascending=False, method='first')

print(f"Data range: {mret['datadate'].min()} to {mret['datadate'].max()}")
print(f"Number of unique stocks: {mret['permno'].nunique()}")

# ============================================================================
# Step 2: Out-of-Sample Portfolio Construction (1990-2024)
# ============================================================================

print("\n2. Running out-of-sample portfolio optimization (1990-2024)...")

def myFF3(DF):
    """Estimate FF3 factor model for a given stock"""
    try:
        reg_model = sm.OLS(
            endog=DF['ret_rf'],
            exog=sm.add_constant(DF[['Mkt-RF', 'SMB', 'HML']])
        ).fit()
        
        estimates = pd.Series({
            'alpha': reg_model.params[0],
            'beta': reg_model.params[1],
            'smb': reg_model.params[2],
            'hml': reg_model.params[3],
            'alpha_se': reg_model.bse[0],
            'beta_se': reg_model.bse[1],
            'smb_se': reg_model.bse[2],
            'hml_se': reg_model.bse[3],
            'r2': reg_model.rsquared,
            'sigma': reg_model.mse_resid
        })
        return estimates
    except:
        return pd.Series({
            'alpha': np.nan, 'beta': np.nan, 'smb': np.nan, 'hml': np.nan,
            'alpha_se': np.nan, 'beta_se': np.nan, 'smb_se': np.nan, 'hml_se': np.nan,
            'r2': np.nan, 'sigma': np.nan
        })


def optimize_portfolio(returns_cov, expected_returns, constraints_type='basic'):
    """
    Optimize portfolio using Markowitz mean-variance framework
    """
    N_assets = len(expected_returns)
    
    Omega = np.array(returns_cov)
    R = np.array(expected_returns).flatten()
    
    if np.any(np.isnan(Omega)) or np.any(np.isnan(R)):
        return None, None
    
    # Add regularization for numerical stability
    Omega = Omega + np.eye(N_assets) * 1e-8
    
    finv = np.ones(N_assets)
    
    # Cap extreme expected returns
    r_min = max(R.min(), -50)
    r_max = min(R.max(), 50)
    
    if r_min >= r_max:
        r_max = r_min + 0.01
    
    r_obj = np.linspace(start=r_min, stop=0.95*r_max, num=30)
    
    Port_stats = []
    
    for i, target_return in enumerate(r_obj):
        G_mat = np.concatenate(
            (np.asmatrix(-1 * R),
             -1 * np.diag(np.ones(N_assets))),
            axis=0
        )
        
        h_Mat = np.concatenate(
            (-1 * np.array([target_return]),
             np.zeros(N_assets))
        )
        
        if constraints_type == 'capped':
            cap_constraint = np.diag(np.ones(N_assets))
            G_mat = np.concatenate((G_mat, cap_constraint), axis=0)
            h_Mat = np.concatenate((h_Mat, 0.17 * np.ones(N_assets)))
        
        q_Mat = np.zeros(N_assets)
        
        try:
            Opt_P = qp(
                P=matrix(Omega, (N_assets, N_assets)),
                q=matrix(q_Mat),
                G=matrix(G_mat),
                h=matrix(h_Mat),
                A=matrix(finv).T,
                b=matrix(np.ones(1))
            )
            
            if Opt_P['status'] == 'optimal':
                weights = np.array(Opt_P['x']).flatten()
                
                port_vol = np.sqrt(weights @ Omega @ weights)
                port_ret = weights @ R
                port_sharpe = port_ret / port_vol if port_vol > 0 else 0
                
                Port_stats.append({
                    'Volatility': port_vol,
                    'Return': port_ret,
                    'Sharpe': port_sharpe,
                    'weights': weights
                })
        except:
            continue
    
    if len(Port_stats) == 0:
        return None, None
    
    Port_df = pd.DataFrame(Port_stats)
    max_sharpe_idx = Port_df['Sharpe'].idxmax()
    optimal_weights = Port_df.loc[max_sharpe_idx, 'weights']
    
    min_var_idx = Port_df['Volatility'].idxmin()
    min_var_weights = Port_df.loc[min_var_idx, 'weights']
    
    return optimal_weights, min_var_weights


# Storage for out-of-sample returns
oos_returns = []

# Loop through years 1990 to 2024
for year in range(1990, 2025):
    print(f"\n  Processing year {year}...")
    
    # Select top 150 stocks based on Dec (year-1) market cap
    dec_data = mret[
        (mret['datadate'].dt.year == year - 1) &
        (mret['datadate'].dt.month == 12)
    ]
    
    if len(dec_data) == 0:
        print(f"  No December {year-1} data found, skipping...")
        continue
    
    last_dec_date = dec_data['datadate'].max()
    
    top150_stocks = (
        dec_data[dec_data['datadate'] == last_dec_date]
        .nsmallest(150, 'mkt_rank')['permno']
        .unique()
    )
    
    if len(top150_stocks) == 0:
        print(f"  No stocks found for {year}, skipping...")
        continue
    
    print(f"    Selected {len(top150_stocks)} stocks")
    
    # Get estimation data (60 months prior to January year)
    estimation_end = pd.Timestamp(f'{year}-01-01')
    estimation_start = estimation_end - pd.DateOffset(months=60)
    
    estimation_data = mret[
        (mret['permno'].isin(top150_stocks)) &
        (mret['datadate'] >= estimation_start) &
        (mret['datadate'] < estimation_end)
    ].copy()
    
    if len(estimation_data) == 0:
        print(f"  No estimation data for {year}, skipping...")
        continue
    
    # Keep only stocks with at least 36 observations
    obs_count = estimation_data.groupby('permno')['datadate'].count()
    valid_stocks = obs_count[obs_count >= 36].index.tolist()
    
    if len(valid_stocks) == 0:
        print(f"  No stocks with sufficient data for {year}, skipping...")
        continue
    
    estimation_data = estimation_data[estimation_data['permno'].isin(valid_stocks)]
    
    print(f"    {len(valid_stocks)} stocks with sufficient data")
    
    # Estimate FF3 model
    FF3_reg = (
        estimation_data
        .groupby('permno', group_keys=False)
        .apply(myFF3)
        .reset_index()
    )
    
    FF3_reg = FF3_reg.dropna()
    
    if len(FF3_reg) == 0:
        print(f"  All regressions failed for {year}, skipping...")
        continue
    
    print(f"    {len(FF3_reg)} successful regressions")
    
    valid_stocks = FF3_reg['permno'].tolist()
    
    # Calculate covariance matrix and expected returns
    Factors = (
        estimation_data[estimation_data['permno'].isin(valid_stocks)]
        .filter(items=['datadate', 'Mkt-RF', 'SMB', 'HML'])
        .drop_duplicates(subset='datadate')
        .drop(columns='datadate')
    )
    
    Omega_K = Factors.cov().values
    b = FF3_reg[['beta', 'smb', 'hml']].values
    S = np.diag(FF3_reg['sigma'].values)
    
    Omega = b @ Omega_K @ b.T + S
    
    factor_means = Factors.mean().values
    R = b @ factor_means
    
    # Optimize portfolio
    optimal_weights, _ = optimize_portfolio(Omega, R, constraints_type='basic')
    
    if optimal_weights is None:
        print(f"  Optimization failed for {year}, skipping...")
        continue
    
    print(f"    Weights sum: {optimal_weights.sum():.4f}, Min: {optimal_weights.min():.4f}, Max: {optimal_weights.max():.4f}")
    
    weights_df = pd.DataFrame({
        'permno': valid_stocks,
        'weight': optimal_weights
    })
    
    # Calculate out-of-sample returns for year t
    oos_data = mret[
        (mret['permno'].isin(valid_stocks)) &
        (mret['datadate'].dt.year == year)
    ].copy()
    
    if len(oos_data) == 0:
        print(f"  No out-of-sample data for {year}, skipping...")
        continue
    
    oos_data = pd.merge(oos_data, weights_df, on='permno', how='inner')
    
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
    
    oos_returns.append(monthly_returns)
    print(f"    Completed {year}: {len(monthly_returns)} months of returns")

# Combine all out-of-sample returns
if len(oos_returns) == 0:
    print("\nERROR: No out-of-sample returns were generated!")
else:
    oos_returns_df = pd.concat(oos_returns, ignore_index=True)
    oos_returns_df.rename(columns={'datadate': 'date'}, inplace=True)
    
    print(f"\n3. Out-of-sample analysis complete.")
    print(f"   Total months: {len(oos_returns_df)}")
    print(f"   Date range: {oos_returns_df['date'].min()} to {oos_returns_df['date'].max()}")
    
    # ============================================================================
    # Step 3: Calculate Performance Statistics
    # ============================================================================
    
    print("\n4. Calculating performance statistics...")
    
    # CRITICAL FIX: Convert percentage returns to decimal
    oos_returns_df['portfolio_return_decimal'] = oos_returns_df['portfolio_return'] / 100
    oos_returns_df['sp500_return_decimal'] = oos_returns_df['sp500_return'] / 100
    
    # Calculate statistics (using percentage form for display)
    avg_monthly_return = oos_returns_df['portfolio_return'].mean()
    std_dev = oos_returns_df['portfolio_return'].std()
    var_1pct = oos_returns_df['portfolio_return'].quantile(0.01)
    
    # Sharpe ratio (annual risk-free rate = 3%, so monthly = 0.25%)
    monthly_rf = 0.25  # in percentage
    excess_returns = oos_returns_df['portfolio_return'] - monthly_rf
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    
    # S&P 500 statistics
    sp500_avg_return = oos_returns_df['sp500_return'].mean()
    sp500_std = oos_returns_df['sp500_return'].std()
    sp500_var_1pct = oos_returns_df['sp500_return'].quantile(0.01)
    sp500_excess = oos_returns_df['sp500_return'] - monthly_rf
    sp500_sharpe = sp500_excess.mean() / sp500_excess.std() * np.sqrt(12)
    
    print("\n" + "=" * 70)
    print("PERFORMANCE STATISTICS (1990-2024)")
    print("=" * 70)
    print("\nOptimal Portfolio (Max Sharpe Ratio):")
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
    # Step 4: Plot Cumulative Returns
    # ============================================================================
    
    print("\n5. Creating cumulative returns plot...")
    
    plot_data = oos_returns_df.copy()
    plot_data = plot_data.dropna(subset=['portfolio_return', 'sp500_return'])
    
    # CRITICAL FIX: Use decimal returns for compounding
    plot_data['portfolio_cumret'] = (1 + plot_data['portfolio_return_decimal']).cumprod()
    plot_data['sp500_cumret'] = (1 + plot_data['sp500_return_decimal']).cumprod()
    
    # Reshape for plotting
    cumret_plot_data = pd.melt(
        plot_data,
        id_vars='date',
        value_vars=['portfolio_cumret', 'sp500_cumret'],
        var_name='strategy',
        value_name='cumulative_return'
    )
    
    cumret_plot_data['strategy'] = cumret_plot_data['strategy'].replace({
        'portfolio_cumret': 'Optimal Portfolio',
        'sp500_cumret': 'S&P 500'
    })
    
    # Create cumulative returns chart
    cumret_chart = (
        alt.Chart(cumret_plot_data,
                  title=f'Value of $1 Invested at Start of {plot_data["date"].min().strftime("%Y")}')
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
    
    cumret_chart.save('cumulative_returns_q1.html')
    print("   Saved to 'cumulative_returns_q1.html'")
    
    # ============================================================================
    # Step 5: Plot Drawdowns
    # ============================================================================
    
    print("\n6. Creating drawdown plot...")
    
    # Calculate drawdowns
    plot_data['portfolio_peak'] = plot_data['portfolio_cumret'].cummax()
    plot_data['sp500_peak'] = plot_data['sp500_cumret'].cummax()
    plot_data['portfolio_drawdown'] = (plot_data['portfolio_cumret'] / plot_data['portfolio_peak'] - 1) * 100
    plot_data['sp500_drawdown'] = (plot_data['sp500_cumret'] / plot_data['sp500_peak'] - 1) * 100
    
    # Reshape for plotting
    drawdown_plot_data = pd.melt(
        plot_data,
        id_vars='date',
        value_vars=['portfolio_drawdown', 'sp500_drawdown'],
        var_name='strategy',
        value_name='drawdown'
    )
    
    drawdown_plot_data['strategy'] = drawdown_plot_data['strategy'].replace({
        'portfolio_drawdown': 'Optimal Portfolio',
        'sp500_drawdown': 'S&P 500'
    })
    
    # Create drawdown chart
    drawdown_chart = (
        alt.Chart(drawdown_plot_data,
                  title='Drawdowns')
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
    
    drawdown_chart.save('drawdowns_q1.html')
    print("   Saved to 'drawdowns_q1.html'")
    
    print("\n" + "=" * 70)
    print("QUESTION 1 COMPLETE")
    print("=" * 70)
    
    print(f"\nFinal Portfolio Value: ${plot_data['portfolio_cumret'].iloc[-1]:.2f}")
    print(f"Final S&P 500 Value:   ${plot_data['sp500_cumret'].iloc[-1]:.2f}")
    
    
    
    
    
    
jckjsdnckjsnd     
    
    
    
# ============================================================================
# QUESTION 5: FF3 + Mom Portfolio Optimization with 17% Weight Cap
# ============================================================================

print("\n\n" + "=" * 70)
print("Starting Question 5: FF3 + Mom Portfolio Optimization (1990-2024)")
print("=" * 70)

# ============================================================================
# Step 1: Data Preparation for Question 5
# ============================================================================

print("\n1. Preparing data for Question 5 (FF3 + Mom)...")

# Check if Mom is in FF3 dataframe
if 'Mom' not in FF3.columns:
    print("ERROR: Mom not found in FF3 dataframe. Available columns:", FF3.columns.tolist())
    print("Skipping Question 5...")
else:
    # Re-prepare data with Mom included
    CCM_q5 = CCM.copy()
    
    # Merge CRSP and Fama-French factors (including Mom) by mdate
    mret_q5 = (
        pd.merge(CCM_q5, FF3, on='mdate', how='inner')
        .assign(ret_rf=lambda df: df['ret'] - df['RF'])
    )
    
    # Rank by market cap each month
    mret_q5['mkt_rank'] = mret_q5.groupby('datadate')['mktCap_lag'].rank(ascending=False, method='first')
    
    print(f"Data range: {mret_q5['datadate'].min()} to {mret_q5['datadate'].max()}")
    print(f"Number of unique stocks: {mret_q5['permno'].nunique()}")
    print(f"Factors available: Mkt-RF, SMB, HML, Mom")
    
    # ============================================================================
    # Step 2: Define FF4 Model Function (FF3 + Mom)
    # ============================================================================
    
    def myFF4(DF):
        """Estimate FF3 + Mom factor model for a given stock"""
        try:
            reg_model = sm.OLS(
                endog=DF['ret_rf'],
                exog=sm.add_constant(DF[['Mkt-RF', 'SMB', 'HML', 'Mom']])
            ).fit()
            
            estimates = pd.Series({
                'alpha': reg_model.params[0],
                'beta': reg_model.params[1],
                'smb': reg_model.params[2],
                'hml': reg_model.params[3],
                'mom': reg_model.params[4],
                'alpha_se': reg_model.bse[0],
                'beta_se': reg_model.bse[1],
                'smb_se': reg_model.bse[2],
                'hml_se': reg_model.bse[3],
                'mom_se': reg_model.bse[4],
                'r2': reg_model.rsquared,
                'sigma': reg_model.mse_resid
            })
            return estimates
        except:
            return pd.Series({
                'alpha': np.nan, 'beta': np.nan, 'smb': np.nan, 'hml': np.nan, 'mom': np.nan,
                'alpha_se': np.nan, 'beta_se': np.nan, 'smb_se': np.nan, 'hml_se': np.nan, 'mom_se': np.nan,
                'r2': np.nan, 'sigma': np.nan
            })
    
    
    def optimize_portfolio_capped(returns_cov, expected_returns, portfolio_type='sharpe'):
        """
        Optimize portfolio using Markowitz mean-variance framework with 17% weight cap
        
        portfolio_type: 'sharpe' for max Sharpe ratio, 'minvar' for minimum variance
        """
        N_assets = len(expected_returns)
        
        Omega = np.array(returns_cov)
        R = np.array(expected_returns).flatten()
        
        if np.any(np.isnan(Omega)) or np.any(np.isnan(R)):
            return None
        
        # Add regularization for numerical stability
        Omega = Omega + np.eye(N_assets) * 1e-8
        
        finv = np.ones(N_assets)
        
        # Cap extreme expected returns
        r_min = max(R.min(), -50)
        r_max = min(R.max(), 50)
        
        if r_min >= r_max:
            r_max = r_min + 0.01
        
        r_obj = np.linspace(start=r_min, stop=0.95*r_max, num=30)
        
        Port_stats = []
        
        for i, target_return in enumerate(r_obj):
            # G matrix: return constraint + non-negative weights + 17% cap
            G_mat = np.concatenate(
                (np.asmatrix(-1 * R),                    # return >= target
                 -1 * np.diag(np.ones(N_assets)),        # weights >= 0
                 np.diag(np.ones(N_assets))),            # weights <= 0.17
                axis=0
            )
            
            # h vector: corresponding constraint values
            h_Mat = np.concatenate(
                (-1 * np.array([target_return]),         # -target_return
                 np.zeros(N_assets),                      # 0 for non-negative
                 0.17 * np.ones(N_assets))                # 0.17 for cap
            )
            
            q_Mat = np.zeros(N_assets)
            
            try:
                Opt_P = qp(
                    P=matrix(Omega, (N_assets, N_assets)),
                    q=matrix(q_Mat),
                    G=matrix(G_mat),
                    h=matrix(h_Mat),
                    A=matrix(finv).T,
                    b=matrix(np.ones(1))
                )
                
                if Opt_P['status'] == 'optimal':
                    weights = np.array(Opt_P['x']).flatten()
                    
                    port_vol = np.sqrt(weights @ Omega @ weights)
                    port_ret = weights @ R
                    port_sharpe = port_ret / port_vol if port_vol > 0 else 0
                    
                    Port_stats.append({
                        'Volatility': port_vol,
                        'Return': port_ret,
                        'Sharpe': port_sharpe,
                        'weights': weights
                    })
            except:
                continue
        
        if len(Port_stats) == 0:
            return None
        
        Port_df = pd.DataFrame(Port_stats)
        
        if portfolio_type == 'sharpe':
            # Return max Sharpe ratio portfolio
            max_sharpe_idx = Port_df['Sharpe'].idxmax()
            return Port_df.loc[max_sharpe_idx, 'weights']
        else:  # minvar
            # Return minimum variance portfolio
            min_var_idx = Port_df['Volatility'].idxmin()
            return Port_df.loc[min_var_idx, 'weights']
    
    
    # ============================================================================
    # Step 3: Out-of-Sample Portfolio Construction (1990-2024)
    # ============================================================================
    
    print("\n2. Running out-of-sample portfolio optimization (1990-2024)...")
    print("   Finding both Optimal (Max Sharpe) and Minimum Variance portfolios...")
    
    # Storage for out-of-sample returns
    oos_returns_optimal = []
    oos_returns_minvar = []
    
    # Loop through years 1990 to 2024
    for year in range(1990, 2025):
        print(f"\n  Processing year {year}...")
        
        # Select top 150 stocks based on Dec (year-1) market cap
        dec_data = mret_q5[
            (mret_q5['datadate'].dt.year == year - 1) &
            (mret_q5['datadate'].dt.month == 12)
        ]
        
        if len(dec_data) == 0:
            print(f"  No December {year-1} data found, skipping...")
            continue
        
        last_dec_date = dec_data['datadate'].max()
        
        top150_stocks = (
            dec_data[dec_data['datadate'] == last_dec_date]
            .nsmallest(150, 'mkt_rank')['permno']
            .unique()
        )
        
        if len(top150_stocks) == 0:
            print(f"  No stocks found for {year}, skipping...")
            continue
        
        print(f"    Selected {len(top150_stocks)} stocks")
        
        # Get estimation data (60 months prior to January year)
        estimation_end = pd.Timestamp(f'{year}-01-01')
        estimation_start = estimation_end - pd.DateOffset(months=60)
        
        estimation_data = mret_q5[
            (mret_q5['permno'].isin(top150_stocks)) &
            (mret_q5['datadate'] >= estimation_start) &
            (mret_q5['datadate'] < estimation_end)
        ].copy()
        
        if len(estimation_data) == 0:
            print(f"  No estimation data for {year}, skipping...")
            continue
        
        # Keep only stocks with at least 36 observations
        obs_count = estimation_data.groupby('permno')['datadate'].count()
        valid_stocks = obs_count[obs_count >= 36].index.tolist()
        
        if len(valid_stocks) == 0:
            print(f"  No stocks with sufficient data for {year}, skipping...")
            continue
        
        estimation_data = estimation_data[estimation_data['permno'].isin(valid_stocks)]
        
        print(f"    {len(valid_stocks)} stocks with sufficient data")
        
        # Estimate FF4 model (FF3 + Mom)
        FF4_reg = (
            estimation_data
            .groupby('permno', group_keys=False)
            .apply(myFF4)
            .reset_index()
        )
        
        FF4_reg = FF4_reg.dropna()
        
        if len(FF4_reg) == 0:
            print(f"  All regressions failed for {year}, skipping...")
            continue
        
        print(f"    {len(FF4_reg)} successful regressions")
        
        valid_stocks = FF4_reg['permno'].tolist()
        
        # Calculate covariance matrix and expected returns
        Factors = (
            estimation_data[estimation_data['permno'].isin(valid_stocks)]
            .filter(items=['datadate', 'Mkt-RF', 'SMB', 'HML', 'Mom'])
            .drop_duplicates(subset='datadate')
            .drop(columns='datadate')
        )
        
        Omega_K = Factors.cov().values
        b = FF4_reg[['beta', 'smb', 'hml', 'mom']].values
        S = np.diag(FF4_reg['sigma'].values)
        
        Omega = b @ Omega_K @ b.T + S
        
        factor_means = Factors.mean().values
        R = b @ factor_means
        
        # Optimize portfolios (both optimal and minimum variance)
        optimal_weights = optimize_portfolio_capped(Omega, R, portfolio_type='sharpe')
        minvar_weights = optimize_portfolio_capped(Omega, R, portfolio_type='minvar')
        
        if optimal_weights is None or minvar_weights is None:
            print(f"  Optimization failed for {year}, skipping...")
            continue
        
        print(f"    Optimal weights - Sum: {optimal_weights.sum():.4f}, Min: {optimal_weights.min():.4f}, Max: {optimal_weights.max():.4f}")
        print(f"    MinVar weights - Sum: {minvar_weights.sum():.4f}, Min: {minvar_weights.min():.4f}, Max: {minvar_weights.max():.4f}")
        
        weights_optimal_df = pd.DataFrame({
            'permno': valid_stocks,
            'weight': optimal_weights
        })
        
        weights_minvar_df = pd.DataFrame({
            'permno': valid_stocks,
            'weight': minvar_weights
        })
        
        # Calculate out-of-sample returns for year t
        oos_data = mret_q5[
            (mret_q5['permno'].isin(valid_stocks)) &
            (mret_q5['datadate'].dt.year == year)
        ].copy()
        
        if len(oos_data) == 0:
            print(f"  No out-of-sample data for {year}, skipping...")
            continue
        
        # Optimal portfolio returns
        oos_data_optimal = pd.merge(oos_data, weights_optimal_df, on='permno', how='inner')
        monthly_returns_optimal = (
            oos_data_optimal
            .assign(weighted_ret=lambda df: df['weight'] * df['ret'])
            .groupby('datadate')
            .agg(
                portfolio_return=('weighted_ret', 'sum'),
                sp500_return=('sprtrn', 'first')
            )
            .reset_index()
        )
        oos_returns_optimal.append(monthly_returns_optimal)
        
        # Minimum variance portfolio returns
        oos_data_minvar = pd.merge(oos_data, weights_minvar_df, on='permno', how='inner')
        monthly_returns_minvar = (
            oos_data_minvar
            .assign(weighted_ret=lambda df: df['weight'] * df['ret'])
            .groupby('datadate')
            .agg(
                portfolio_return=('weighted_ret', 'sum'),
                sp500_return=('sprtrn', 'first')
            )
            .reset_index()
        )
        oos_returns_minvar.append(monthly_returns_minvar)
        
        print(f"    Completed {year}: {len(monthly_returns_optimal)} months of returns")
    
    # ============================================================================
    # Step 4: Combine Results and Calculate Statistics
    # ============================================================================
    
    if len(oos_returns_optimal) == 0 or len(oos_returns_minvar) == 0:
        print("\nERROR: No out-of-sample returns were generated for Question 5!")
    else:
        # Combine all out-of-sample returns
        oos_optimal_df = pd.concat(oos_returns_optimal, ignore_index=True)
        oos_minvar_df = pd.concat(oos_returns_minvar, ignore_index=True)
        
        oos_optimal_df.rename(columns={'datadate': 'date'}, inplace=True)
        oos_minvar_df.rename(columns={'datadate': 'date'}, inplace=True)
        
        print(f"\n3. Out-of-sample analysis complete.")
        print(f"   Total months: {len(oos_optimal_df)}")
        print(f"   Date range: {oos_optimal_df['date'].min()} to {oos_optimal_df['date'].max()}")
        
        # Convert percentage returns to decimal
        oos_optimal_df['portfolio_return_decimal'] = oos_optimal_df['portfolio_return'] / 100
        oos_minvar_df['portfolio_return_decimal'] = oos_minvar_df['portfolio_return'] / 100
        oos_optimal_df['sp500_return_decimal'] = oos_optimal_df['sp500_return'] / 100
        oos_minvar_df['sp500_return_decimal'] = oos_minvar_df['sp500_return'] / 100
        
        # ============================================================================
        # Step 5: Calculate Performance Statistics
        # ============================================================================
        
        print("\n4. Calculating performance statistics...")
        
        def calculate_statistics(df, portfolio_name):
            """Calculate performance statistics for a portfolio"""
            avg_monthly_return = df['portfolio_return'].mean()
            std_dev = df['portfolio_return'].std()
            var_1pct = df['portfolio_return'].quantile(0.01)
            
            # Sharpe ratio (annual risk-free rate = 3%, so monthly = 0.25%)
            monthly_rf = 0.25  # in percentage
            excess_returns = df['portfolio_return'] - monthly_rf
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
            
            return {
                'name': portfolio_name,
                'avg_return': avg_monthly_return,
                'std_dev': std_dev,
                'var_1pct': var_1pct,
                'sharpe': sharpe_ratio
            }
        
        # Calculate statistics for both portfolios
        optimal_stats = calculate_statistics(oos_optimal_df, 'Optimal Portfolio (Max Sharpe)')
        minvar_stats = calculate_statistics(oos_minvar_df, 'Minimum Variance Portfolio')
        
        # S&P 500 statistics (use optimal df, same S&P data)
        sp500_avg_return = oos_optimal_df['sp500_return'].mean()
        sp500_std = oos_optimal_df['sp500_return'].std()
        sp500_var_1pct = oos_optimal_df['sp500_return'].quantile(0.01)
        monthly_rf = 0.25
        sp500_excess = oos_optimal_df['sp500_return'] - monthly_rf
        sp500_sharpe = sp500_excess.mean() / sp500_excess.std() * np.sqrt(12)
        
        print("\n" + "=" * 70)
        print("PERFORMANCE STATISTICS - QUESTION 5 (FF3 + Mom, 17% Cap)")
        print("=" * 70)
        
        for stats in [optimal_stats, minvar_stats]:
            print(f"\n{stats['name']}:")
            print(f"  Average Monthly Return: {stats['avg_return']:.4f}%")
            print(f"  Standard Deviation:     {stats['std_dev']:.4f}%")
            print(f"  1% VaR:                 {stats['var_1pct']:.4f}%")
            print(f"  Sharpe Ratio (Annual):  {stats['sharpe']:.4f}")
        
        print("\nS&P 500:")
        print(f"  Average Monthly Return: {sp500_avg_return:.4f}%")
        print(f"  Standard Deviation:     {sp500_std:.4f}%")
        print(f"  1% VaR:                 {sp500_var_1pct:.4f}%")
        print(f"  Sharpe Ratio (Annual):  {sp500_sharpe:.4f}")
        print("=" * 70)
        
        # ============================================================================
        # Step 6: Plot Cumulative Returns
        # ============================================================================
        
        print("\n5. Creating cumulative returns plot...")
        
        # Calculate cumulative returns
        oos_optimal_df['portfolio_cumret'] = (1 + oos_optimal_df['portfolio_return_decimal']).cumprod()
        oos_minvar_df['portfolio_cumret'] = (1 + oos_minvar_df['portfolio_return_decimal']).cumprod()
        oos_optimal_df['sp500_cumret'] = (1 + oos_optimal_df['sp500_return_decimal']).cumprod()
        
        # Prepare data for plotting
        cumret_plot_data = pd.concat([
            oos_optimal_df[['date', 'portfolio_cumret']].assign(strategy='Optimal Portfolio'),
            oos_minvar_df[['date', 'portfolio_cumret']].assign(strategy='Minimum Variance'),
            oos_optimal_df[['date', 'sp500_cumret']].rename(columns={'sp500_cumret': 'portfolio_cumret'}).assign(strategy='S&P 500')
        ], ignore_index=True)
        
        cumret_plot_data.rename(columns={'portfolio_cumret': 'cumulative_return'}, inplace=True)
        
        # Create cumulative returns chart
        cumret_chart_q5 = (
            alt.Chart(cumret_plot_data,
                      title=f'Value of $1 Invested at Start of {oos_optimal_df["date"].min().strftime("%Y")} (FF3+Mom, 17% Cap)')
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
        
        cumret_chart_q5.save('cumulative_returns_q5.html')
        print("   Saved to 'cumulative_returns_q5.html'")
        
        # ============================================================================
        # Step 7: Plot Drawdowns
        # ============================================================================
        
        print("\n6. Creating drawdown plot...")
        
        # Calculate drawdowns
        oos_optimal_df['portfolio_peak'] = oos_optimal_df['portfolio_cumret'].cummax()
        oos_minvar_df['portfolio_peak'] = oos_minvar_df['portfolio_cumret'].cummax()
        oos_optimal_df['sp500_peak'] = oos_optimal_df['sp500_cumret'].cummax()
        
        oos_optimal_df['portfolio_drawdown'] = (oos_optimal_df['portfolio_cumret'] / oos_optimal_df['portfolio_peak'] - 1) * 100
        oos_minvar_df['portfolio_drawdown'] = (oos_minvar_df['portfolio_cumret'] / oos_minvar_df['portfolio_peak'] - 1) * 100
        oos_optimal_df['sp500_drawdown'] = (oos_optimal_df['sp500_cumret'] / oos_optimal_df['sp500_peak'] - 1) * 100
        
        # Prepare data for plotting
        drawdown_plot_data = pd.concat([
            oos_optimal_df[['date', 'portfolio_drawdown']].assign(strategy='Optimal Portfolio'),
            oos_minvar_df[['date', 'portfolio_drawdown']].assign(strategy='Minimum Variance'),
            oos_optimal_df[['date', 'sp500_drawdown']].rename(columns={'sp500_drawdown': 'portfolio_drawdown'}).assign(strategy='S&P 500')
        ], ignore_index=True)
        
        drawdown_plot_data.rename(columns={'portfolio_drawdown': 'drawdown'}, inplace=True)
        
        # Create drawdown chart
        drawdown_chart_q5 = (
            alt.Chart(drawdown_plot_data,
                      title='Drawdowns (FF3+Mom, 17% Cap)')
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
        
        drawdown_chart_q5.save('drawdowns_q5.html')
        print("   Saved to 'drawdowns_q5.html'")
        
        print("\n" + "=" * 70)
        print("QUESTION 5 COMPLETE")
        print("=" * 70)
        
        print(f"\nFinal Values:")
        print(f"  Optimal Portfolio:      ${oos_optimal_df['portfolio_cumret'].iloc[-1]:.2f}")
        print(f"  Minimum Variance:       ${oos_minvar_df['portfolio_cumret'].iloc[-1]:.2f}")
        print(f"  S&P 500:                ${oos_optimal_df['sp500_cumret'].iloc[-1]:.2f}")
        
        print("\n" + "=" * 70)
        print("ALL QUESTIONS COMPLETE!")
        print("=" * 70)
    
    
    
    
    
    
    
    
    





















# ============================================================================
# QUESTION 5: FF3 + Additional Factor Portfolio Optimization with 17% Weight Cap
# ============================================================================

print("\n\n" + "=" * 70)
print("Starting Question 5: FF3 + Additional Factor Portfolio Optimization (1990-2024)")
print("=" * 70)

# ============================================================================
# Step 1: Load Additional Factors and Prepare Data
# ============================================================================

print("\n1. Loading additional factors from ADF...")

# Load the ADF file (user should have this loaded already)
# ADF should contain: mdate, bab, rmw, cma, mom, date

# Check what factors are available in ADF
print(f"ADF columns: {ADF.columns.tolist()}")
print(f"ADF shape: {ADF.shape}")

# Convert date column to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(ADF['date']):
    ADF['date'] = pd.to_datetime(ADF['date'])

# Create mdate for merging if not already in Period format
if not isinstance(ADF['mdate'].iloc[0], pd.Period):
    ADF['mdate'] = ADF['date'].dt.to_period('M')

print(f"ADF date range: {ADF['date'].min()} to {ADF['date'].max()}")

# ============================================================================
# Step 2: Choose Additional Factor (using MOM as example)
# ============================================================================

# Let's use MOM (momentum) as the additional factor
# You can change this to 'bab', 'rmw', or 'cma' if you prefer
ADDITIONAL_FACTOR = 'mom'

print(f"\n2. Using additional factor: {ADDITIONAL_FACTOR.upper()}")

# ============================================================================
# Step 3: Merge FF3 with Additional Factor
# ============================================================================

print("\n3. Merging FF3 factors with additional factor...")

# Merge FF3 with ADF to get the additional factor
FF3_plus = pd.merge(
    FF3, 
    ADF[['mdate', ADDITIONAL_FACTOR]], 
    on='mdate', 
    how='inner'
)

print(f"Combined factors available: {[col for col in FF3_plus.columns if col not in ['mdate', 'RF']]}")
print(f"Combined data range: {FF3_plus['mdate'].min()} to {FF3_plus['mdate'].max()}")

# ============================================================================
# Step 4: Prepare CRSP Data with New Factors
# ============================================================================

print("\n4. Preparing CRSP data with FF3 + additional factor...")

CCM_q5 = CCM.copy()

# Merge CRSP with FF3 + additional factor
mret_q5 = (
    pd.merge(CCM_q5, FF3_plus, on='mdate', how='inner')
    .assign(ret_rf=lambda df: df['ret'] - df['RF'])
)

# Rank by market cap each month
mret_q5['mkt_rank'] = mret_q5.groupby('datadate')['mktCap_lag'].rank(ascending=False, method='first')

print(f"Data range: {mret_q5['datadate'].min()} to {mret_q5['datadate'].max()}")
print(f"Number of unique stocks: {mret_q5['permno'].nunique()}")

# ============================================================================
# Step 5: Define FF4 Model Function (FF3 + Additional Factor)
# ============================================================================

def myFF4(DF):
    """Estimate FF3 + additional factor model for a given stock"""
    try:
        reg_model = sm.OLS(
            endog=DF['ret_rf'],
            exog=sm.add_constant(DF[['Mkt-RF', 'SMB', 'HML', ADDITIONAL_FACTOR]])
        ).fit()
        
        estimates = pd.Series({
            'alpha': reg_model.params[0],
            'beta': reg_model.params[1],
            'smb': reg_model.params[2],
            'hml': reg_model.params[3],
            'add_factor': reg_model.params[4],
            'alpha_se': reg_model.bse[0],
            'beta_se': reg_model.bse[1],
            'smb_se': reg_model.bse[2],
            'hml_se': reg_model.bse[3],
            'add_factor_se': reg_model.bse[4],
            'r2': reg_model.rsquared,
            'sigma': reg_model.mse_resid
        })
        return estimates
    except:
        return pd.Series({
            'alpha': np.nan, 'beta': np.nan, 'smb': np.nan, 'hml': np.nan, 'add_factor': np.nan,
            'alpha_se': np.nan, 'beta_se': np.nan, 'smb_se': np.nan, 'hml_se': np.nan, 'add_factor_se': np.nan,
            'r2': np.nan, 'sigma': np.nan
        })


def optimize_portfolio_capped(returns_cov, expected_returns, portfolio_type='sharpe'):
    """
    Optimize portfolio using Markowitz mean-variance framework with 17% weight cap
    
    portfolio_type: 'sharpe' for max Sharpe ratio, 'minvar' for minimum variance
    """
    N_assets = len(expected_returns)
    
    Omega = np.array(returns_cov)
    R = np.array(expected_returns).flatten()
    
    if np.any(np.isnan(Omega)) or np.any(np.isnan(R)):
        return None
    
    # Add regularization for numerical stability
    Omega = Omega + np.eye(N_assets) * 1e-8
    
    finv = np.ones(N_assets)
    
    # Cap extreme expected returns
    r_min = max(R.min(), -50)
    r_max = min(R.max(), 50)
    
    if r_min >= r_max:
        r_max = r_min + 0.01
    
    r_obj = np.linspace(start=r_min, stop=0.95*r_max, num=30)
    
    Port_stats = []
    
    for i, target_return in enumerate(r_obj):
        # G matrix: return constraint + non-negative weights + 17% cap
        G_mat = np.concatenate(
            (np.asmatrix(-1 * R),                    # return >= target
             -1 * np.diag(np.ones(N_assets)),        # weights >= 0
             np.diag(np.ones(N_assets))),            # weights <= 0.17
            axis=0
        )
        
        # h vector: corresponding constraint values
        h_Mat = np.concatenate(
            (-1 * np.array([target_return]),         # -target_return
             np.zeros(N_assets),                      # 0 for non-negative
             0.17 * np.ones(N_assets))                # 0.17 for cap
        )
        
        q_Mat = np.zeros(N_assets)
        
        try:
            Opt_P = qp(
                P=matrix(Omega, (N_assets, N_assets)),
                q=matrix(q_Mat),
                G=matrix(G_mat),
                h=matrix(h_Mat),
                A=matrix(finv).T,
                b=matrix(np.ones(1))
            )
            
            if Opt_P['status'] == 'optimal':
                weights = np.array(Opt_P['x']).flatten()
                
                port_vol = np.sqrt(weights @ Omega @ weights)
                port_ret = weights @ R
                port_sharpe = port_ret / port_vol if port_vol > 0 else 0
                
                Port_stats.append({
                    'Volatility': port_vol,
                    'Return': port_ret,
                    'Sharpe': port_sharpe,
                    'weights': weights
                })
        except:
            continue
    
    if len(Port_stats) == 0:
        return None
    
    Port_df = pd.DataFrame(Port_stats)
    
    if portfolio_type == 'sharpe':
        # Return max Sharpe ratio portfolio
        max_sharpe_idx = Port_df['Sharpe'].idxmax()
        return Port_df.loc[max_sharpe_idx, 'weights']
    else:  # minvar
        # Return minimum variance portfolio
        min_var_idx = Port_df['Volatility'].idxmin()
        return Port_df.loc[min_var_idx, 'weights']


# ============================================================================
# Step 6: Out-of-Sample Portfolio Construction (1990-2024)
# ============================================================================

print("\n5. Running out-of-sample portfolio optimization (1990-2024)...")
print("   Finding both Optimal (Max Sharpe) and Minimum Variance portfolios...")

# Storage for out-of-sample returns
oos_returns_optimal = []
oos_returns_minvar = []

# Loop through years 1990 to 2024
for year in range(1990, 2025):
    print(f"\n  Processing year {year}...")
    
    # Select top 150 stocks based on Dec (year-1) market cap
    dec_data = mret_q5[
        (mret_q5['datadate'].dt.year == year - 1) &
        (mret_q5['datadate'].dt.month == 12)
    ]
    
    if len(dec_data) == 0:
        print(f"  No December {year-1} data found, skipping...")
        continue
    
    last_dec_date = dec_data['datadate'].max()
    
    top150_stocks = (
        dec_data[dec_data['datadate'] == last_dec_date]
        .nsmallest(150, 'mkt_rank')['permno']
        .unique()
    )
    
    if len(top150_stocks) == 0:
        print(f"  No stocks found for {year}, skipping...")
        continue
    
    print(f"    Selected {len(top150_stocks)} stocks")
    
    # Get estimation data (60 months prior to January year)
    estimation_end = pd.Timestamp(f'{year}-01-01')
    estimation_start = estimation_end - pd.DateOffset(months=60)
    
    estimation_data = mret_q5[
        (mret_q5['permno'].isin(top150_stocks)) &
        (mret_q5['datadate'] >= estimation_start) &
        (mret_q5['datadate'] < estimation_end)
    ].copy()
    
    if len(estimation_data) == 0:
        print(f"  No estimation data for {year}, skipping...")
        continue
    
    # Keep only stocks with at least 36 observations
    obs_count = estimation_data.groupby('permno')['datadate'].count()
    valid_stocks = obs_count[obs_count >= 36].index.tolist()
    
    if len(valid_stocks) == 0:
        print(f"  No stocks with sufficient data for {year}, skipping...")
        continue
    
    estimation_data = estimation_data[estimation_data['permno'].isin(valid_stocks)]
    
    print(f"    {len(valid_stocks)} stocks with sufficient data")
    
    # Estimate FF4 model (FF3 + additional factor)
    FF4_reg = (
        estimation_data
        .groupby('permno', group_keys=False)
        .apply(myFF4)
        .reset_index()
    )
    
    FF4_reg = FF4_reg.dropna()
    
    if len(FF4_reg) == 0:
        print(f"  All regressions failed for {year}, skipping...")
        continue
    
    print(f"    {len(FF4_reg)} successful regressions")
    
    valid_stocks = FF4_reg['permno'].tolist()
    
    # Calculate covariance matrix and expected returns
    Factors = (
        estimation_data[estimation_data['permno'].isin(valid_stocks)]
        .filter(items=['datadate', 'Mkt-RF', 'SMB', 'HML', ADDITIONAL_FACTOR])
        .drop_duplicates(subset='datadate')
        .drop(columns='datadate')
    )
    
    Omega_K = Factors.cov().values
    b = FF4_reg[['beta', 'smb', 'hml', 'add_factor']].values
    S = np.diag(FF4_reg['sigma'].values)
    
    Omega = b @ Omega_K @ b.T + S
    
    factor_means = Factors.mean().values
    R = b @ factor_means
    
    # Optimize portfolios (both optimal and minimum variance)
    optimal_weights = optimize_portfolio_capped(Omega, R, portfolio_type='sharpe')
    minvar_weights = optimize_portfolio_capped(Omega, R, portfolio_type='minvar')
    
    if optimal_weights is None or minvar_weights is None:
        print(f"  Optimization failed for {year}, skipping...")
        continue
    
    print(f"    Optimal weights - Sum: {optimal_weights.sum():.4f}, Min: {optimal_weights.min():.4f}, Max: {optimal_weights.max():.4f}")
    print(f"    MinVar weights - Sum: {minvar_weights.sum():.4f}, Min: {minvar_weights.min():.4f}, Max: {minvar_weights.max():.4f}")
    
    weights_optimal_df = pd.DataFrame({
        'permno': valid_stocks,
        'weight': optimal_weights
    })
    
    weights_minvar_df = pd.DataFrame({
        'permno': valid_stocks,
        'weight': minvar_weights
    })
    
    # Calculate out-of-sample returns for year t
    oos_data = mret_q5[
        (mret_q5['permno'].isin(valid_stocks)) &
        (mret_q5['datadate'].dt.year == year)
    ].copy()
    
    if len(oos_data) == 0:
        print(f"  No out-of-sample data for {year}, skipping...")
        continue
    
    # Optimal portfolio returns
    oos_data_optimal = pd.merge(oos_data, weights_optimal_df, on='permno', how='inner')
    monthly_returns_optimal = (
        oos_data_optimal
        .assign(weighted_ret=lambda df: df['weight'] * df['ret'])
        .groupby('datadate')
        .agg(
            portfolio_return=('weighted_ret', 'sum'),
            sp500_return=('sprtrn', 'first')
        )
        .reset_index()
    )
    oos_returns_optimal.append(monthly_returns_optimal)
    
    # Minimum variance portfolio returns
    oos_data_minvar = pd.merge(oos_data, weights_minvar_df, on='permno', how='inner')
    monthly_returns_minvar = (
        oos_data_minvar
        .assign(weighted_ret=lambda df: df['weight'] * df['ret'])
        .groupby('datadate')
        .agg(
            portfolio_return=('weighted_ret', 'sum'),
            sp500_return=('sprtrn', 'first')
        )
        .reset_index()
    )
    oos_returns_minvar.append(monthly_returns_minvar)
    
    print(f"    Completed {year}: {len(monthly_returns_optimal)} months of returns")

# ============================================================================
# Step 7: Combine Results and Calculate Statistics
# ============================================================================

if len(oos_returns_optimal) == 0 or len(oos_returns_minvar) == 0:
    print("\nERROR: No out-of-sample returns were generated for Question 5!")
else:
    # Combine all out-of-sample returns
    oos_optimal_df = pd.concat(oos_returns_optimal, ignore_index=True)
    oos_minvar_df = pd.concat(oos_returns_minvar, ignore_index=True)
    
    oos_optimal_df.rename(columns={'datadate': 'date'}, inplace=True)
    oos_minvar_df.rename(columns={'datadate': 'date'}, inplace=True)
    
    print(f"\n6. Out-of-sample analysis complete.")
    print(f"   Total months: {len(oos_optimal_df)}")
    print(f"   Date range: {oos_optimal_df['date'].min()} to {oos_optimal_df['date'].max()}")
    
    # Convert percentage returns to decimal
    oos_optimal_df['portfolio_return_decimal'] = oos_optimal_df['portfolio_return'] / 100
    oos_minvar_df['portfolio_return_decimal'] = oos_minvar_df['portfolio_return'] / 100
    oos_optimal_df['sp500_return_decimal'] = oos_optimal_df['sp500_return'] / 100
    oos_minvar_df['sp500_return_decimal'] = oos_minvar_df['sp500_return'] / 100
    
    # ============================================================================
    # Step 8: Calculate Performance Statistics
    # ============================================================================
    
    print("\n7. Calculating performance statistics...")
    
    def calculate_statistics(df, portfolio_name):
        """Calculate performance statistics for a portfolio"""
        avg_monthly_return = df['portfolio_return'].mean()
        std_dev = df['portfolio_return'].std()
        var_1pct = df['portfolio_return'].quantile(0.01)
        
        # Sharpe ratio (annual risk-free rate = 3%, so monthly = 0.25%)
        monthly_rf = 0.25  # in percentage
        excess_returns = df['portfolio_return'] - monthly_rf
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        
        return {
            'name': portfolio_name,
            'avg_return': avg_monthly_return,
            'std_dev': std_dev,
            'var_1pct': var_1pct,
            'sharpe': sharpe_ratio
        }
    
    # Calculate statistics for both portfolios
    optimal_stats = calculate_statistics(oos_optimal_df, 'Optimal Portfolio (Max Sharpe)')
    minvar_stats = calculate_statistics(oos_minvar_df, 'Minimum Variance Portfolio')
    
    # S&P 500 statistics (use optimal df, same S&P data)
    sp500_avg_return = oos_optimal_df['sp500_return'].mean()
    sp500_std = oos_optimal_df['sp500_return'].std()
    sp500_var_1pct = oos_optimal_df['sp500_return'].quantile(0.01)
    monthly_rf = 0.25
    sp500_excess = oos_optimal_df['sp500_return'] - monthly_rf
    sp500_sharpe = sp500_excess.mean() / sp500_excess.std() * np.sqrt(12)
    
    print("\n" + "=" * 70)
    print(f"PERFORMANCE STATISTICS - QUESTION 5 (FF3 + {ADDITIONAL_FACTOR.upper()}, 17% Cap)")
    print("=" * 70)
    
    for stats in [optimal_stats, minvar_stats]:
        print(f"\n{stats['name']}:")
        print(f"  Average Monthly Return: {stats['avg_return']:.4f}%")
        print(f"  Standard Deviation:     {stats['std_dev']:.4f}%")
        print(f"  1% VaR:                 {stats['var_1pct']:.4f}%")
        print(f"  Sharpe Ratio (Annual):  {stats['sharpe']:.4f}")
    
    print("\nS&P 500:")
    print(f"  Average Monthly Return: {sp500_avg_return:.4f}%")
    print(f"  Standard Deviation:     {sp500_std:.4f}%")
    print(f"  1% VaR:                 {sp500_var_1pct:.4f}%")
    print(f"  Sharpe Ratio (Annual):  {sp500_sharpe:.4f}")
    print("=" * 70)
    
    # ============================================================================
    # Step 9: Plot Cumulative Returns
    # ============================================================================
    
    print("\n8. Creating cumulative returns plot...")
    
    # Calculate cumulative returns
    oos_optimal_df['portfolio_cumret'] = (1 + oos_optimal_df['portfolio_return_decimal']).cumprod()
    oos_minvar_df['portfolio_cumret'] = (1 + oos_minvar_df['portfolio_return_decimal']).cumprod()
    oos_optimal_df['sp500_cumret'] = (1 + oos_optimal_df['sp500_return_decimal']).cumprod()
    
    # Prepare data for plotting
    cumret_plot_data = pd.concat([
        oos_optimal_df[['date', 'portfolio_cumret']].assign(strategy='Optimal Portfolio'),
        oos_minvar_df[['date', 'portfolio_cumret']].assign(strategy='Minimum Variance'),
        oos_optimal_df[['date', 'sp500_cumret']].rename(columns={'sp500_cumret': 'portfolio_cumret'}).assign(strategy='S&P 500')
    ], ignore_index=True)
    
    cumret_plot_data.rename(columns={'portfolio_cumret': 'cumulative_return'}, inplace=True)
    
    # Create cumulative returns chart
    cumret_chart_q5 = (
        alt.Chart(cumret_plot_data,
                  title=f'Value of $1 Invested at Start of {oos_optimal_df["date"].min().strftime("%Y")} (FF3+{ADDITIONAL_FACTOR.upper()}, 17% Cap)')
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
    
    cumret_chart_q5.save('cumulative_returns_q5.html')
    print("   Saved to 'cumulative_returns_q5.html'")
    
    # ============================================================================
    # Step 10: Plot Drawdowns
    # ============================================================================
    
    print("\n9. Creating drawdown plot...")
    
    # Calculate drawdowns
    oos_optimal_df['portfolio_peak'] = oos_optimal_df['portfolio_cumret'].cummax()
    oos_minvar_df['portfolio_peak'] = oos_minvar_df['portfolio_cumret'].cummax()
    oos_optimal_df['sp500_peak'] = oos_optimal_df['sp500_cumret'].cummax()
    
    oos_optimal_df['portfolio_drawdown'] = (oos_optimal_df['portfolio_cumret'] / oos_optimal_df['portfolio_peak'] - 1) * 100
    oos_minvar_df['portfolio_drawdown'] = (oos_minvar_df['portfolio_cumret'] / oos_minvar_df['portfolio_peak'] - 1) * 100
    oos_optimal_df['sp500_drawdown'] = (oos_optimal_df['sp500_cumret'] / oos_optimal_df['sp500_peak'] - 1) * 100
    
    # Prepare data for plotting
    drawdown_plot_data = pd.concat([
        oos_optimal_df[['date', 'portfolio_drawdown']].assign(strategy='Optimal Portfolio'),
        oos_minvar_df[['date', 'portfolio_drawdown']].assign(strategy='Minimum Variance'),
        oos_optimal_df[['date', 'sp500_drawdown']].rename(columns={'sp500_drawdown': 'portfolio_drawdown'}).assign(strategy='S&P 500')
    ], ignore_index=True)
    
    drawdown_plot_data.rename(columns={'portfolio_drawdown': 'drawdown'}, inplace=True)
    
    # Create drawdown chart
    drawdown_chart_q5 = (
        alt.Chart(drawdown_plot_data,
                  title=f'Drawdowns (FF3+{ADDITIONAL_FACTOR.upper()}, 17% Cap)')
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
    
    drawdown_chart_q5.save('drawdowns_q5.html')
    print("   Saved to 'drawdowns_q5.html'")
    
    print("\n" + "=" * 70)
    print("QUESTION 5 COMPLETE")
    print("=" * 70)
    
    print(f"\nFinal Values:")
    print(f"  Optimal Portfolio:      ${oos_optimal_df['portfolio_cumret'].iloc[-1]:.2f}")
    print(f"  Minimum Variance:       ${oos_minvar_df['portfolio_cumret'].iloc[-1]:.2f}")
    print(f"  S&P 500:                ${oos_optimal_df['sp500_cumret'].iloc[-1]:.2f}")
    
    print("\n" + "=" * 70)
    print("ALL QUESTIONS COMPLETE!")
    print("=" * 70)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    