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

# f = open('hsiStockList.json')

# stocks = json.load(f)
# tickers_list = []
# for i in stocks:
#     tickers_list.append(i["symbol"])

# print(tickers_list)

data = yf.download("^HSI", progress=False)
data.ta.bbands(close="Close", length=20, std=2, append=True)

# View our data
pd.set_option("display.max_columns", None)
data = data.loc[data.index[-1] - timedelta(days=60):]
str_bbands = data.to_json(orient="index").replace(" ", "")
dict_bbands = json.loads(str_bbands)
keys = [*dict_bbands]

print(type(str_bbands), str_bbands)
