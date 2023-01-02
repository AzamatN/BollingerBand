
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
data.ta.bbands(close="Close", length=20, std=2, append=True)

# View our data
pd.set_option("display.max_columns", None)
data = data.loc[data.index[-1] - timedelta(days=60):]
# print(data.to_json(orient="index").replace(" ", ""))
str_bbands = data.to_json(orient="index").replace(" ", "")
dict_bbands = json.loads(str_bbands)
keys = [*dict_bbands]
# format of the json:
# index: open: high: low: close: asjclose: colume: bbl: bbm: bbu: bbb: bbp:


# def bbands_reverse():
#     #smth
def bbands_reverse(cur_day):
    buy = False
    sell = False
    if(dict_bbands[cur_day]["BBP_20_2.0"] <= 0):
        buy = True
    elif(dict_bbands[cur_day]["BBP_20_2.0"] >= 1):
        sell = True
    else:
        buy = False
        sell = False
    return buy, sell


def bbands_bbm_buy(cur_day, cnt):
    buy = False
    # FOR FUTURE
    # increasing_price = False
    # diff_percentage = dict_bbands[keys[cnt - 1]]["AdjClose"] / dict_bbands[keys[cnt]]["AdjClose"]
    # if(diff_percentage > xxxxx):

    if(dict_bbands[keys[cnt]]["AdjClose"] > dict_bbands[keys[cnt - 1]]["AdjClose"]):
        if(dict_bbands[cur_day]["AdjClose"] >= dict_bbands[cur_day]["BBM_20_2.0"]):
            buy = True
        else:
            buy = False
    return buy


def bbands_bull_market(cur_day):
    bull = False
    if(dict_bbands[cur_day]["AdjClose"] >= dict_bbands[cur_day]["BBM_20_2.0"] and dict_bbands[cur_day]["AdjClose"] <= dict_bbands[cur_day]["BBU_20_2.0"]):
        bull = True
    return bull


def bbands_decrease_bbu(cur_day):
    sell = False
    # hardcoding
    # find a cnt without passing it
    tmp_cnt = keys.index(cur_day)

    if(dict_bbands[keys[tmp_cnt]]["AdjClose"] < dict_bbands[keys[tmp_cnt - 1]]["AdjClose"]):
        # might need a change for a cur_day variable
        if(dict_bbands[cur_day]["AdjClose"] < dict_bbands[cur_day]["BBU_20_2.0"] and dict_bbands[keys[tmp_cnt - 1]]["AdjClose"] > dict_bbands[cur_day]["BBU_20_2.0"]):
            sell = True
        else:
            sell = False
    return sell


def bbands_decrease_bbm(cur_day, cnt):
    sell = False
    if(dict_bbands[keys[cnt]]["AdjClose"] < dict_bbands[keys[cnt - 1]]["AdjClose"]):
        # might need a change for a cur_day variable
        if(dict_bbands[cur_day]["AdjClose"] < dict_bbands[cur_day]["BBM_20_2.0"] and dict_bbands[keys[cnt - 1]]["AdjClose"] > dict_bbands[cur_day]["BBM_20_2.0"]):
            sell = True
        else:
            sell = False
    return sell


def bbands_short(cur_day):
    short = False
    if(dict_bbands[cur_day]["AdjClose"] >= dict_bbands[cur_day]["BBU_20_2.0"] and dict_bbands[cur_day]["AdjClose"] <= dict_bbands[cur_day]["BBM_20_2.0"]):
        short = True
    return short


sold_before = False
bought_before = False
buy_cnt = 0
sell_cnt = 0
reverse_buy = False
reverse_sell = False
bbm_buy = False
bull_market = False
decrease_bbu = False
decrease_bbm = False
short_market = False
for i in range(len(keys)):
    reverse_buy, reverse_sell = bbands_reverse(keys[i])
    bbm_buy = bbands_bbm_buy(keys[i], i)
    bull_market = bbands_bull_market(keys[i])
    decrease_bbu = bbands_decrease_bbu(keys[i])
    decrease_bbm = bbands_decrease_bbm(keys[i], i)
    short_market = bbands_short(keys[i])
    dict_bbands[keys[i]]["reverse buy"] = reverse_buy
    dict_bbands[keys[i]]["reverse sell"] = reverse_sell
    dict_bbands[keys[i]]["bbm buy"] = bbm_buy
    dict_bbands[keys[i]]["bull market"] = bull_market
    dict_bbands[keys[i]]["decrease bbu sell"] = decrease_bbu
    dict_bbands[keys[i]]["decrease bbm sell"] = decrease_bbm
    dict_bbands[keys[i]]["short market"] = short_market
    if(reverse_buy or bbm_buy or bull_market):
        dict_bbands[keys[i]]["status"] = "BUY"
    elif(reverse_sell or decrease_bbu or decrease_bbm or short_market):
        dict_bbands[keys[i]]["status"] = "SELL"
    else:
        dict_bbands[keys[i]]["status"] = "HOLD"


str_bbands = json.dumps(dict_bbands)
jsonFile = open("bb_epoch.json", "w")
jsonFile.write(str_bbands)
jsonFile.close()
df = pd.read_json(r'./bb_epoch.json')
df.to_csv(r'./bbands_hsi.csv', index=None)

# # Force lowercase (optional)
data.columns = [x.lower() for x in data.columns]

# Construct a 2 x 1 Plotly figure
# fig = make_subplots(rows=1, cols=1)

# # price Line
# fig.append_trace(
#     go.Scatter(
#         x=data.index,
#         y=data["close"],
#         line=dict(color="#ff9900", width=1),
#         name="close",
#         # showlegend=False,
#         legendgroup="1",
#     ),
#     row=1,
#     col=1,
# )
# fig.append_trace(
#     go.Scatter(
#         x=data.index,
#         y=data["bbl_20_2.0"],
#         line=dict(color="#0800ff", width=1),
#         name="BBL",
#         # showlegend=False,
#         legendgroup="1",
#     ),
#     row=1,
#     col=1,
# )

# fig.append_trace(
#     go.Scatter(
#         x=data.index,
#         y=data["bbu_20_2.0"],
#         line=dict(color="#0800ff", width=1),
#         name="BBU",
#         # showlegend=False,
#         legendgroup="1",
#     ),
#     row=1,
#     col=1,
# )
# fig.append_trace(
#     go.Scatter(
#         x=data.index,
#         y=data["bbm_20_2.0"],
#         line=dict(color="#c0392b", width=1),
#         name="BBM",
#         # showlegend=False,
#         legendgroup="1",
#     ),
#     row=1,
#     col=1,
# )

# # Candlestick chart for pricing
# fig.append_trace(
#     go.Candlestick(
#         x=data.index,
#         open=data["open"],
#         high=data["high"],
#         low=data["low"],
#         close=data["close"],
#         increasing_line_color="#27ae60",
#         decreasing_line_color="red",
#         showlegend=False,
#     ),
#     row=1,
#     col=1,
# )

# # Make it pretty
# layout = go.Layout(
#     plot_bgcolor="#efefef",
#     # Font Families
#     font_family="Monospace",
#     font_color="#000000",
#     font_size=20,
#     xaxis=dict(rangeslider=dict(visible=False)),
# )

# # Update options and show plot
# fig.update_layout(layout)
# fig.show()
