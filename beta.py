import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import statsmodels.api as sm
from statsmodels import regression
from scipy import stats

#load csv files with data, start= 28-02-13, end=28-02-18 for both files (monthly stock data from yahoo finance)
def colParse(df):
    cols = df.columns.str.replace('\s+', '_')  # in case there are multiple white spaces
    return cols

agl = pd.read_csv('agl.csv', parse_dates=True, index_col='Date',)
asx_200 = pd.read_csv('200.csv', parse_dates=True, index_col='Date')
#check to see if imported, columns names
print(agl.columns)

#parse columns and replace spaces with underscore
agl.columns = colParse(agl)
asx_200.columns = colParse(asx_200)
print(agl.columns)

# joining the closing prices of the two datasets
monthly_prices = pd.concat([agl['Close'], asx_200['Close']], axis=1)
monthly_prices.columns = ['AGL', 'ASX-200']

# check the head of the dataframe
print(monthly_prices.head())

# calculate monthly returns
monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row

#places cleaned returns data points into dataset to run linear regression
X = clean_monthly_returns['ASX-200']
y = clean_monthly_returns['AGL']

plt.figure(figsize=(20,10))
X.plot()
y.plot()
plt.ylabel("Monthly returns of AGL and ASX-200 over 5Y")
plt.show()

X1 = sm.add_constant(X)

# make regression model
model = sm.OLS(y, X1)

# fit model and print results
results = model.fit()
beta = results.summary()

#slope is also a measure of the beta value
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
print(slope)

with open("Beta.txt", "w") as text_file:
    print(f"{beta}", file=text_file)
