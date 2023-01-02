import scipy.spatial.distance as dist
import tslearn.metrics
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import matplotlib.dates as mdates
import json
import time

# Fetch the data
import yfinance as yf
import sys

start_time = time.time()

tickers_list = ["^HSI"]
data = yf.download(tickers_list)
hsi_df = data.reset_index()

hsi_df = hsi_df[['Date', 'Close']]
hsi_df.columns = ['Date', 'Value']
hsi_df['Value'] = pd.to_numeric(hsi_df['Value'], errors='coerce')
'''
print(np.min(hsi_df['Date'] ),np.max(hsi_df['Date'] ))
'''
hsi_df = hsi_df.sort_values('Date', ascending=True)
hsi_df = hsi_df.dropna(how='any')

'''
print(hsi_df.head())
'''
'''
#Overview of the HSI history
fig1, ax1 = plt.subplots(figsize=(16, 8))
plt.plot(hsi_df['Date'], hsi_df['Value'], label='Value', color='red')
plt.title('HSI ' + str(np.min(hsi_df['Date'])) + ' - ' + str(np.max(hsi_df['Date'])))
plt.legend(loc='upper left')
plt.grid()
plt.show()
'''

# function to split the data to several pieces from the HSI dataframe


def split_seq(seq, num_pieces):
    start = 0
    for i in range(num_pieces):
        stop = start + len(seq[i::num_pieces])
        yield seq[start:stop]
        start = stop


# function for calculating the correlation between 2 pattern
def pearson(s1, s2):
    s1_c = s1 - np.mean(s1)
    s2_c = s2 - np.mean(s2)
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))


# DTW function (forming the distance matrix)
def dp(dist_mat):
    N, M = dist_mat.shape

    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],  # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)


# sorting the HSI data, to ensure it is in ascending order)
hsi_df = hsi_df.sort_values('Date', ascending=True)

# The period that we are going to analyze, for example 14 day, 60 day, 90day, etc.
lookback = int(sys.argv[1])
dates1 = hsi_df['Date']
prices1 = list(hsi_df['Value'].values)
counter_ = -1
price_series1 = []
for day1 in dates1:
    counter_ += 1
    if counter_ >= lookback:
        price_series1.append(prices1[counter_ - lookback:counter_])

timeseries_df1 = pd.DataFrame(price_series1)
'''
print(timeseries_df1.shape)
'''

# Same as the lookback, need to adjust if we change the lookback period
complexity = int(sys.argv[1])
s_data2 = []
# Spliting the data in the range of complexity for every trading day.
# for example, if the complexity is 5, data will be  14/3 - 18/3/2022, 15/3 - 21/3/2022, ...
for index2, row2 in timeseries_df1.iterrows():
    simplified_values2 = []
    for r2 in split_seq(list(row2.values), complexity):
        simplified_values2.append(np.mean(r2))
    s_data2.append(simplified_values2)

# Same as the function above, it is used to analyze other stock data originally by creating the timeseries_df2
# In this case, I only compare with the HSI itself due to the time limited for the capstone project.
s_data = []
for index, row in timeseries_df1.iterrows():
    simplified_values = []
    for r in split_seq(list(row.values), complexity):
        simplified_values.append(np.mean(r))
    s_data.append(simplified_values)

'''
print(len(s_data))
'''

# list to be storing the index and the correlation of the similar pattern
index_list = []
index_list2 = []
record = 0


# Finding out the pattern by comparing each dataset that splited above
for og in range(len(s_data)-1, len(s_data)):
    for cd in range(5000, len(s_data2)):
        correz = pearson(s_data[og], s_data2[cd])
        # Target correlation for the pattern, can be adjust if there is only a few samples.
        for i in np.arange(0.8, 0.55, -0.05):
            if (correz > i):
                # To avoid the duplicated data with the current data/ pattern that already found
                if cd - record >= complexity and cd - og <= -complexity:
                    index_list.append(cd)
                    index_list2.append(correz)
                    record = cd
            if (len(index_list) != 0):
                break

index_list2, index_list = zip(
    *sorted(zip(index_list2, index_list), reverse=True))


# Sort and filter the top m similar pattern according to the correlation, m can be adjust at below *
ilist = []
ilist2 = []
if len(index_list) < 10:   # *
    volumn = len(index_list)
else:
    volumn = 10    # *
for top in range(0, volumn):
    ilist.append(index_list[top])
    ilist2.append(index_list2[top])


# Preparation of DTW
for index, row in timeseries_df1.iterrows():
    if index == len(s_data)-1:
        x = np.array(row)
        # x is the current dataset

# Retrieve the data from the top m pattern
pending = []
for index, row in timeseries_df1.iterrows():
    for match in ilist:
        if index == match:
            z = np.array(row)
            pending.append(z)

# Normalizing the data between the current data and the pattern
pro = []
for arr in pending:
    op = []
    dif = arr[0] - x[0]
    for sub in arr:
        res = sub - dif
        op.append(res)
        cal = np.array(op)
    pro.append(cal)

# Calculating the DTW distance between each pattern and current data
score = []
for y in pro:
    similarity = tslearn.metrics.dtw(x, y)
    score.append(similarity)
'''
print(score)
'''

# Sort and filter the top n pattern according to the DTW distance, n can be adjust at below *
ind = []
corr = []
s_score = []
score, ilist2, ilist = zip(*sorted(zip(score, ilist2, ilist)))

if len(index_list) < 3:  # *
    volumn = len(ilist)
else:
    volumn = 3  # *

for top in range(0, volumn):
    ind.append(ilist[top])
    corr.append(ilist2[top])
    s_score.append(score[top])
'''
print("index   correlation      DTW distance")
'''
'''
for number in range(0, volumn):
    print(ind[number], corr[number], s_score[number])
'''

# Retrieve data from top n pattern
pending = []
for index, row in timeseries_df1.iterrows():
    for match in ind:
        if index == match:
            z = np.array(row)
            pending.append(z)

# Normalizing data
pro = []
for arr in pending:
    op = []
    dif = arr[0] - x[0]
    for sub in arr:
        res = sub - dif
        op.append(res)
        cal = np.array(op)
    pro.append(cal)

# Compute the Distance and Cost matrix
for y in pro:
    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = abs(x[i] - y[j])
    path, cost_mat = dp(dist_mat)
    plt.figure(figsize=(6, 4))
    plt.subplot(121)
    plt.title("Distance matrix")
    plt.imshow(dist_mat, cmap=plt.cm.binary,
               interpolation="nearest", origin="lower")
    plt.subplot(122)
    plt.title("Cost matrix")
    plt.imshow(cost_mat, cmap=plt.cm.binary,
               interpolation="nearest", origin="lower")
    x_path, y_path = zip(*path)
    plt.plot(y_path, x_path)

'''
print(i,j)
'''

# Plot and save the graph for each pattern and the current data
A = 1
today = datetime.date.today()
today = today.strftime("%Y%m%d")
for y in pro:
    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = abs(x[i] - y[j])

    img = plt.figure()
    img.set_size_inches(18.5, 10.5)

    for x_i, y_j in path:
        plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="silver")

    plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3", label="current")
    A = str(A)
    plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0", label="Pattern" + A)
    plt.axis("off")
    plt.legend()
    #plt.savefig("Pattern" + A +"_"+ today + ".png")
    A = int(A)
    A += 1

# Creating a csv file to store the current data
datelist = []
valuelist = []
for value in hsi_df.index:
    if value in range(len(s_data), len(s_data)+complexity):
        datelist.append(hsi_df['Date'][value])
        valuelist.append(hsi_df['Value'][value])
pattern = {}
pattern['Date'] = datelist
pattern['Value'] = valuelist
current_df = pd.DataFrame(pattern, columns=['Date', 'Value'])


#current_df.to_csv('current_'+today+'.csv', index=False)


# Creating n csv files to store the similar patterns
A = 1
datelist = []
valuelist = []
pattern_df_3 = []
for date in ind:
    for value in hsi_df.index:
        if value in range(date, date+complexity*2):
            datelist.append(hsi_df['Date'][value])
            valuelist.append(hsi_df['Value'][value])
    pattern = {}
    pattern['Date'] = datelist
    pattern['Value'] = valuelist
    pattern_df = pd.DataFrame(pattern, columns=['Date', 'Value'])
    pattern_df_3.append(pattern_df)
    A = str(A)
    #pattern_df.to_csv('pattern'+A+'_'+today+'.csv', index=False)
    A = int(A)
    A += 1
    datelist = []
    valuelist = []

if (len(pattern_df_3) == 3):
    X = {
        "currentDate": [
            {
                "startDate": str(current_df["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(current_df["Date"].tail(1).item().strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ],
        "pattern1": [
            {
                "startDate": str(pattern_df_3[0]["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(pattern_df_3[0]["Date"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ],
        "pattern2": [
            {
                "startDate": str(pattern_df_3[1]["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(pattern_df_3[1]["Date"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ],
        "pattern3": [
            {
                "startDate": str(pattern_df_3[2]["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(pattern_df_3[2]["Date"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ]
    }

    # convert into JSON:
    y = json.dumps(X)

    print(y)

elif (len(pattern_df_3) == 2):
    X = {
        "currentDate": [
            {
                "startDate": str(current_df["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(current_df["Date"].tail(1).item().strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ],
        "pattern1": [
            {
                "startDate": str(pattern_df_3[0]["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(pattern_df_3[0]["Date"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ],
        "pattern2": [
            {
                "startDate": str(pattern_df_3[1]["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(pattern_df_3[1]["Date"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ]
    }

    # convert into JSON:
    y = json.dumps(X)

    print(y)

else:
    X = {
        "currentDate": [
            {
                "startDate": str(current_df["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(current_df["Date"].tail(1).item().strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ],
        "pattern1": [
            {
                "startDate": str(pattern_df_3[0]["Date"][0].strftime("%Y-%m-%dT%H:%M:%S.000+0800")),
                "endDate": str(pattern_df_3[0]["Date"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.000+0800"))
            }
        ]
    }

    # convert into JSON:
    y = json.dumps(X)

    print(y)
