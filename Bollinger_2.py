# Define the ticker list
import yfinance as yf
import pandas as pd
from matplotlib import pyplot
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import dateutil.parser
import pandas_ta as ta
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from datetime import date
import json

tickers_list = ["^HSI"]

# Fetch the data
data = yf.download(tickers_list, progress=False)

# data.ta.macd(close="Close", fast=12, slow=26, signal=9, append=True)
# data.ta.rsi(close="Close", length=14, append=True)

data.ta.bbands(close="Close", length=20, std=2, append=True)

# View our data
pd.set_option("display.max_columns", None)
data = data.loc[data.index[-1] - timedelta(days=365):]
#print(data.to_json(orient="index").replace(" ", ""))
str_bbands = data.to_json(orient="index").replace(" ", "")
dict_bbands = json.loads(str_bbands)
# format of the json:
# index: open: high: low: close: asjclose: colume: bbl: bbm: bbu: bbb: bbp:

sold_before = False
bought_before = False
for i in dict_bbands:
    if(dict_bbands[i]["BBP_20_2.0"] <= 0):
        if(bought_before == False):
            dict_bbands[i]['status'] = 'BUY'
            bought_before = True
            sold_before = False
        else:
            dict_bbands[i]['status'] = 'KEEP'
    elif(dict_bbands[i]["BBP_20_2.0"] >= 1):
        if(sold_before == False):
            dict_bbands[i]['status'] = 'SELL'
            sold_before = True
            bought_before = False
        else:
            dict_bbands[i]['status'] = 'KEEP'
    else:
        dict_bbands[i]['status'] = 'KEEP'

str_bbands = json.dumps(dict_bbands)
jsonFile = open("bb.json", "w")
jsonFile.write(str_bbands)
jsonFile.close()

# # Force lowercase (optional)
data.columns = [x.lower() for x in data.columns]

# Construct a 2 x 1 Plotly figure
fig = make_subplots(rows=1, cols=1)

# price Line
fig.append_trace(
    go.Scatter(
        x=data.index,
        y=data["close"],
        line=dict(color="#ff9900", width=1),
        name="close",
        # showlegend=False,
        legendgroup="1",
    ),
    row=1,
    col=1,
)
fig.append_trace(
    go.Scatter(
        x=data.index,
        y=data["bbl_20_2.0"],
        line=dict(color="#0800ff", width=1),
        name="BBL",
        # showlegend=False,
        legendgroup="1",
    ),
    row=1,
    col=1,
)

fig.append_trace(
    go.Scatter(
        x=data.index,
        y=data["bbu_20_2.0"],
        line=dict(color="#0800ff", width=1),
        name="BBU",
        # showlegend=False,
        legendgroup="1",
    ),
    row=1,
    col=1,
)
fig.append_trace(
    go.Scatter(
        x=data.index,
        y=data["bbm_20_2.0"],
        line=dict(color="#c0392b", width=1),
        name="BBM",
        # showlegend=False,
        legendgroup="1",
    ),
    row=1,
    col=1,
)

# Candlestick chart for pricing
fig.append_trace(
    go.Candlestick(
        x=data.index,
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        increasing_line_color="#27ae60",
        decreasing_line_color="red",
        showlegend=False,
    ),
    row=1,
    col=1,
)

# Make it pretty
layout = go.Layout(
    plot_bgcolor="#efefef",
    # Font Families
    font_family="Monospace",
    font_color="#000000",
    font_size=20,
    xaxis=dict(rangeslider=dict(visible=False)),
)

# Update options and show plot
fig.update_layout(layout)
fig.show()
