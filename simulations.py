"""
Monte Carlo Simulation for Quantified STF Fund Investor Class (QSTFX)


We implemented Monte Carlo simulations to estimate portfolio growth over 10 and 15 year horizons using historical QSTFX data,
divided into Pre-Pandemic (2016–2020) and Post-Pandemic (2020–2024) subsets. Each subset's distributions, summary statistics,
and financial metrics were analyzed and visualized with interactive plotly charts.

Functions:
1. load_data()
2. monte_carlo()
3. summary_stats()
4. plot_timehorizons()
5. plot_summary_stats()

Dependencies:
1. NumPy
2. Pandas
3. Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def load_data():
    """
    1. The load_data() function is implemented to extract the 'Date', 'Adj Close', and 'Return' columns from the QSTFX dataset.
    2. Preprocessing steps:
        - Convert the 'Date' attribute to datetime format and set it as the index for splitting the data into two different time_ranges.
        - Drop rows with NaN values on 'Adj Close' and 'Daily Return' attributes.
    3. Return qstfx_df dataframe.
    """

    data_qstfx = 'QSTFX.csv'
    qstfx_df = pd.read_csv(data_qstfx)

    qstfx_df['Date'] = pd.to_datetime(qstfx_df['Date'])
    qstfx_df.set_index('Date', inplace=True)

    qstfx_df['Adj Close'] = pd.to_numeric(qstfx_df['Adj Close'])
    qstfx_df.dropna(subset=['Adj Close'], inplace=True)

    qstfx_df['Daily Return'] = qstfx_df['Return']
    qstfx_df.dropna(subset=['Daily Return'], inplace=True)

    return qstfx_df
def monte_carlo(initial_investment, drift, std_return, years, simulations):
    """
    1. Convert the time horizons (10 and 15 years) into 252 trading days.
    2. np.random.seed(42) to provide consistency in the results every time this code runs.
    3. random_paths generates random values drawn from a normal distribution :
        - Mean(drift) represents the expected return adjusting for volatility.
        - std_return represents the variability in the returns.
        - Shape (days , simulations) one simulation path over each time horizon.
    4. simulated_investment calculates the cummulative sum of daily returns for each simulation.  The cummulative is exponentiate
    to model the Geometric Brownian Motion.
    """

    days = int(years * 252)

    np.random.seed(42)
    random_paths = np.random.normal(drift, std_return, (days, simulations))

    simulated_investment = initial_investment * np.exp(np.cumsum(random_paths, axis=0))

    return simulated_investment
def summary_stats(time_horizons, initial_investment, subset_data, simulations, time_range):
    """
    1. summary_statistics dictionary stores the summary_stats and finance metrics for each time horizon.
    2. calculate the summary stats (mean and std_dev) of daily returns.
    3. calculate the drift: adjusted mean return incorporating volatility and the sharpe ratio: measures risk-adjusted return.
    4. print the results on a formatted table.
    """

    summary_statistics = {}

    mean_return = subset_data['Daily Return'].mean()
    std_return = subset_data['Daily Return'].std()

    print(f"\nSubset: {time_range}")
    print(f"Mean Daily Return: {mean_return:.6f}")
    print(f"Standard Deviation of Daily Return: {std_return:.6f}")

    drift = mean_return - 0.5 * (std_return ** 2)
    sharpe_ratio = (mean_return * 252 - 0.02) / (std_return * np.sqrt(252))

    header_row = f"{'Years':^6} {'Mean Final Price':^18} {'Median Final Price':^20} {'5th Percentile':^16} {'95th Percentile':^16} {'Mean Profit':^16}"
    print(header_row)
    print("-" * len(header_row))

    for years in time_horizons:

        simulated_investment = monte_carlo(initial_investment, drift, std_return, years, simulations)

        final_investment = simulated_investment[-1, :]
        profit = final_investment - initial_investment
        mean_investment = np.mean(final_investment)
        median_investment = np.median(final_investment)
        lower_end = np.percentile(final_investment, 5)
        higher_end = np.percentile(final_investment, 95)

        mean_profit = np.mean(profit)

        summary_statistics[years] = {
            'Simulated_Investment': simulated_investment,
            'Final_Investment': final_investment,
            'Mean Final Price': mean_investment,
            'Median Final Price': median_investment,
            '5th Percentile': lower_end,
            '95th Percentile': higher_end,
            'Mean Profit': mean_profit,
            'Sharpe Ratio': sharpe_ratio,
            'Drift': drift
        }

        data_row = f"{years:^6} {mean_investment:^18.2f} {median_investment:^20.2f} {lower_end:^16.2f} {higher_end:^16.2f} {mean_profit:^16.2f}"
        print(data_row)

    print(f"\nSharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Drift: {drift:.6f}")
    print()

    return summary_statistics
def plot_timehorizons(summary_statistics, time_horizons):
    """
    Plot the Monte Carlo simulation results for portfolio growth over 10 and 15 time horizons.
    """

    for years in time_horizons:

        simulated_investment = summary_statistics[years]['Simulated_Investment']

        fig = go.Figure()

        for i in range(min(100, simulated_investment.shape[1])):

            fig.add_trace(go.Scatter(
                x=np.arange(simulated_investment.shape[0]),
                y=simulated_investment[:, i],
                mode='lines',
                line=dict(width=1),
                name=f"Simulation {i + 1}",
                showlegend=i < 5
            ))

        fig.update_layout(
            title=f"Monte Carlo Simulations of Portfolio Value for {years} Years ({simulated_investment.shape[1]} Simulations)"
                  f"<br><sup>Quantified STF Fund Investor Class (QSTFX)</sup>",
            xaxis_title=f"Simulated Trading Days over {years} Years",
            yaxis_title="Portfolio Value (in Dollars)",
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.show()
def plot_summary_stats(summary_statistics):
    """
    Histograms : Plot the distribution of final portfolio values for 10 and 15 time horizons.
    """

    fig = go.Figure()

    for years, stats in summary_statistics.items():

        final_investment = stats['Final_Investment']

        fig.add_trace(go.Histogram(
            x=final_investment,
            name=f"{years} Years",
            opacity=0.6,
            nbinsx=50,
            texttemplate='%{y}',
            textposition='outside',
            textfont=dict(color='black')
        ))

    fig.update_layout(
        title=f"Distribution of Final Portfolio Value for All Time Horizons (Simulations: {simulations}) "
              f"<br><sup>Quantified STF Fund Investor Class (QSTFX)</sup>",
        xaxis_title="Final Portfolio Value (in Dollars)",
        yaxis_title="Frequency Count",
        barmode='overlay',
        template="plotly_white",
        showlegend=True
    )
    fig.update_traces()

    fig.show()

## Call the load_data() function
qstfx_df = load_data()

## Run 10000 simulatios for 10 and 15 time horizons
simulations = 10000

## Assigns the most recent adjusted closing price from the data set to the initial_investment
initial_investment = qstfx_df['Adj Close'].iloc[-1]

## Subset's dictionary divide the QSTFX dataset into two time ranges :
subset = { "Pre Pandemic 2016 - 2020": qstfx_df['2016-01-01':'2020-12-31'],
           "Post Pandemic 2020 - 2024": qstfx_df['2020-01-01':'2024-12-31'] }

## Iterate over the time_ranges , assigns the 10 and 15 time horizons for each distribution, and compute the summary_statistics  :
## Plots : Portfolio evolution for each horizon.
## Histograms of final portfolio values, comparing outcomes between time horizons and time periods.
for time_range, subset_data in subset.items():
    if time_range == "Pre Pandemic 2016 - 2020":
        time_horizons = [10,15]
    elif time_range == "Post Pandemic 2020 - 2024":
        time_horizons = [10,15]
    summary_statistics = summary_stats(time_horizons, initial_investment, subset_data, simulations, time_range)
    #plot_timehorizons(summary_statistics, time_horizons)
    #plot_summary_stats(summary_statistics)