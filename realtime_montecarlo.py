"""
Monte Carlo Simulation for Quantified STF Fund Investor Class (QSTFX)


Real time Monte Carlo (Computational cost).

Main functions:
1. load_data()
2. monte_carlo()
3. realtime_montecarlo()
4. live_app()

Dependencies:
1. NumPy
2. Pandas
3. Plotly
4. Dash
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

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

    qstfx_df['Daily Return'] = pd.to_numeric(qstfx_df['Return'], errors='coerce')
    qstfx_df.dropna(subset=['Daily Return'], inplace=True)

    return qstfx_df
def monte_carlo(initial_investment, drift, std_return, days, simulations):
    """
    1. np.random.seed(42) to provide consistency in the results every time this code runs.
    2. random_paths generates random values drawn from a normal distribution.
    3. calculate cumulative (np.cumsum) returns and converts them to simulated_investment value.
    """

    np.random.seed(42)
    random_paths = np.random.normal(0, std_return, (days, simulations))

    returns = np.cumsum(drift + random_paths, axis=0)

    simulated_investment = initial_investment * np.exp(returns)

    return simulated_investment
def realtime_montecarlo(n_intervals, qstfx_df, result, ctr_time_horizons, ctr_n_simulations):
    """
    1. Runs a real-time Monte Carlo simulation and updates results.
        - qstfx_df: DataFrame containing preprocessed data from the load_data() function.
        - time_horizons: Simulates portfolio growth for time periods of 10 to 15 years.
        - n_simulations: Performs simulations starting with 1000, incrementing to 5000 and 10000.
        - Calculates summary statistics: Computes the mean and standard deviation of daily returns.
        - ctr_time_horizons: Counter for iterating over different time horizons (10 and 15 years) .
        - ctr_n_simulations: Counter for iterating over different number of simulations.
    """

    time_horizons = [10, 15]
    n_simulations = [1000, 5000, 10000]

    mean_return = qstfx_df['Daily Return'].mean()
    std_return = qstfx_df['Daily Return'].std()

    drift = mean_return - 0.5 * (std_return ** 2)

    initial_investment = qstfx_df['Adj Close'].iloc[-1]

    years = time_horizons[ctr_time_horizons]
    days = years * 252

    simulations = n_simulations[ctr_n_simulations]

    start_time = time.time()

    monte_carlo(initial_investment, drift, std_return, days, simulations)

    execution_time = time.time() - start_time

    result = pd.concat([
        result,
        pd.DataFrame({
            'Years': [years],
            'Simulations': [simulations],
            'Execution Time': [execution_time]
        })], ignore_index=True)

    ctr_n_simulations += 1
    if ctr_n_simulations >= len(n_simulations):
        ctr_n_simulations = 0
        ctr_time_horizons += 1
        if ctr_time_horizons >= len(time_horizons):
            ctr_time_horizons = 0

    fig = go.Figure()
    for years in result['Years'].unique():
        subset = result[result['Years'] == years]
        fig.add_trace(go.Scatter(
            x=subset['Simulations'],
            y=subset['Execution Time'],
            mode='lines+markers',
            name=f'{years} Years'))

    fig.update_layout(
        xaxis_title='Simulations (Log Scale)',
        yaxis_title='Execution Time (Seconds)',
        xaxis_type='log',
        template='plotly_white')

    return fig, execution_time, result, ctr_time_horizons, ctr_n_simulations

### Dash App Initialization :
### 1 . Customizations : We applied consistent styling to the application using Google Fonts (Nunito).
### 2 . Component Definitions : html.div as principal container for the layout  and ddc.Interval to trigger periodic updates

app = Dash(__name__)
app.layout = html.Div([
    html.Link(
        href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600&display=swap",
        rel="stylesheet"
    ),
    html.H1("Real-Time Monte Carlo Simulation (Computational cost)", style={
    'font-family': 'Nunito, sans-serif', 'font-size': '30px', 'color': '#2E4053', 'text-align': 'center' }
    ),
    html.Div(id='execution_time', style={
    'font-family': 'Nunito, sans-serif', 'font-size': '20px', 'color': '#34495E', 'text-align': 'center', 'margin-bottom': '20px'
    }),
    dcc.Graph(id='execution_time_graph'),
    dcc.Interval(
        id='n_intervals', interval=1 * 1000, n_intervals=0
    ),
    html.Div(id='explanation', style={
    'font-family': 'Nunito, sans-serif', 'font-size': '18px', 'color': '#5D6D7E', 'text-align': 'center', 'margin': '10px 50px'
    }),
    html.Div(id='timecomplexity', style={
        'font-family': 'Nunito, sans-serif', 'font-size': '20px', 'color': '#5D6D7E', 'text-align': 'center', 'margin': '30px 50px', 'font-weight': 'bold'
    }),
    html.Div(id='ournames', style={
    'font-family': 'Nunito, sans-serif', 'font-size': '15px', 'color': '#5D6D7E', 'text-align': 'left', 'margin': '50px 50px', 'font-weight': 'bold'
    }) ])

## Call the load_data() function
qstfx_df = load_data()

## Result is a container to collect and store the output of Monte Carlo simulations as they are executed.
result = pd.DataFrame(columns=['Years', 'Simulations', 'Execution Time (seconds)', 'Mean Final Price',
                                'Median Final Price', '5th Percentile', '95th Percentile'])

## Counters to keep track of iterations over time_horizons and n_simulations.
ctr_time_horizons, ctr_n_simulations = 0, 0
@app.callback(
    [Output('execution_time_graph', 'figure'),
            Output('execution_time', 'children'),
            Output('explanation', 'children'),
            Output('timecomplexity', 'children'),
            Output('ournames', 'children')],
            Input('n_intervals', 'n_intervals'))

def live_app(n_intervals):
    """
    live_app() function dynamically updates the application components at regular intervals.
    It uses the realtime_montecarlo function to compute new simulation data and updates
    """

    global result, ctr_time_horizons, ctr_n_simulations

    fig, execution_time, result, ctr_time_horizons, ctr_n_simulations = (
        realtime_montecarlo(n_intervals, qstfx_df, result, ctr_time_horizons, ctr_n_simulations ))

    simulations = [1000, 5000, 10000][ctr_n_simulations]

    execution_time_text = f"Last Execution Time: {execution_time:.4f} seconds | Simulations: {simulations}"

    explanation = (
        "This graph illustrates the trade-offs between computational cost and simulation accuracy in the Monte Carlo algorithm. "
        "Higher simulation counts and longer time horizons demand significantly more computational resources.")

    timecomplexity = (
        "Time complexity : O(h * n * d) ) "
        " n = the number of simulations , h = time_horizons and d = days" )

    ournames = (
        "Melanie Loaiza BU ID: U78379196 , "
        "Carlos Vargas BU ID: U42396592 ")

    return fig, execution_time_text , explanation , timecomplexity, ournames

app.run_server(debug=True)





