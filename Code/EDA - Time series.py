from requirements import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.interpolate import interp1d
from def_size_comparison import *

# import the dataset
df = pd.read_csv('Covid_France.csv')

# Select only the headers (columns names)
columns = df.columns.values.tolist()

# Inverse the list
columns_inverse = columns[::-1]

# Create a new DataFrame with the inverted columns
df_inverse = pd.DataFrame(columns=columns_inverse)

# Fill the new DataFrame with the values from the original DataFrame
for col in columns:
    df_inverse[col] = df[col]

# Display the percentage of missing values in each column
nan_values = df_inverse.isna().sum()/len(df_inverse)*100
print("Percentage of NaN values per columns in df_inverse (and in df by extension):")
print(nan_values)

# display the highest percentage of missing values
print()
print("Highest percentage of NaN values:", round(nan_values.max(), 2),
      "for the feature '", nan_values.idxmax(), "'.")

# compare the dimensions of the raw version of df and a cleaned version, without NaN
df_cleaned = df.dropna()
print("Dimensions of df_cleaned:", df_cleaned.shape)
print(f"Dimensions of Df {df.shape} with NaN values")
print(
    f"Loss of data {(df.shape[0] - df_cleaned.shape[0])/df.shape[0]*100:.2f}%")

# remove Ehpad related columns
df_cleaned = df.drop(
    columns=['total_cas_possibles_ehpad', 'total_cas_confirmes_ehpad'])
# compare the dimensions of the raw version of df and a cleaned version, without NaN
size_comparison(df, df_cleaned)
# display the percentage of missing values in each column
df_cleaned_percent = df_cleaned.isna().sum()/len(df_cleaned)*100
print("Percentage of NaN values per columns in df_cleaned:")
print(df_cleaned_percent)

# visualize the data
df_bis = df_cleaned.dropna(axis=1, how='all')
df_bis['date'] = pd.to_datetime(df_bis['date'], format='%Y-%m-%d')

plt.figure(figsize=(10, 6))
plt.plot(df_bis['date'], df['total_cas_confirmes'])
# Set the x-axis major locator and formatter
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.yscale('log')
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Total confirmed cases')
plt.title('Total confirmed cases over time')
plt.show()

# We would like to have multiple scales for comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# First plot with linear scale
axes[0].plot(df_bis['date'], df['total_cas_confirmes'])
axes[0].set_yscale('linear')
# Set y-axis limits to avoid cutting the graph
axes[0].set_ylim(0, df['total_cas_confirmes'].max()*1.1)
axes[0].set_title('Linear scale')
axes[0].set_xlabel('Date (dd-mm-yyy)')
axes[0].set_ylabel('Total confirmed cases')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
axes[0].xaxis.set_major_locator(mdates.MonthLocator())
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.5)

# Second plot, with logarithmic scale
axes[1].plot(df_bis['date'], df['total_cas_confirmes'])
axes[1].set_yscale('log')
axes[1].set_title('Logarithmic scale')
axes[1].set_ylabel('Total confirmed cases')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# First, we notice that the graphs have stopped at 2021-02-01,
# which is not the last date available in the dataset, maybe a
# part of 'total_cas_confirmes' is missing.

# Count the number of total_cas_confirmes missing since 01-02-2021
missing_dates = df_bis[df_bis['total_cas_confirmes'].isna()]['date']
missing_dates = missing_dates[missing_dates >= '2021-02-01']

len_missing = len(missing_dates)
len_2021 = len(df_bis[df_bis['date'] >= '01-02-2021'])

if len_missing < len_2021:
    print(
        f"There is/are {len_2021-len_missing} date(s) with total_cas_confirmes missing since 01-02-2021, out of {len_missing} dates.")
    print(
        f"This means there is {len_missing/len_2021*100:.2f}% of dates with total_cas_confirmes missing since 01-02-2021.")
else:
    print("There is no confirmed cases since 01-02-2021.")

# split the data into train and test sets
df_train = df[df['date'] < '2020-07-01'].copy()
df_test = df[df['date'] < '2020-07-01'].copy()
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

# add random NaN values to the train set
nan_indices = np.random.choice(
    df_train.index, size=int(len(df_train)*0.1), replace=False)
df_train.loc[nan_indices, 'total_cas_confirmes'] = np.nan

# fill the NaN values in the train set using the moving average
window_size = 10
df_train['mv_avg'] = df_train['total_cas_confirmes'].fillna(
    df_train['total_cas_confirmes'].rolling(window=window_size, min_periods=1).mean())

# fill the NaN values in the train set using the backward moving average
df_train['mv_avg_b'] = df_train['total_cas_confirmes'].fillna(
    df_train['total_cas_confirmes']
    .shift(-window_size + 1)
    .rolling(window=window_size, min_periods=1)
    .mean())

# fill the NaN values in the train set using the linear interpolation
df_train['linear_interp'] = df_train['total_cas_confirmes'].interpolate(
    method='linear')

# fill the NaN values in the train set using the LOESS/LOWESS
df_lowess = df_train.copy()
df_lowess['date_ord'] = df_lowess['date'].map(pd.Timestamp.toordinal)
df_nonan = df_lowess.dropna(subset=['total_cas_confirmes'])
# LOWESS
lowess_result = sm.nonparametric.lowess(
    endog=df_nonan['total_cas_confirmes'],
    exog=df_nonan['date_ord'],
    frac=0.1
)
x_lowess = lowess_result[:, 0]
y_lowess = lowess_result[:, 1]
f_interp = interp1d(x_lowess, y_lowess, kind='linear',
                    fill_value='extrapolate')
df_lowess['lowess_filled'] = f_interp(df_lowess['date_ord'])
df_lowess['total_cas_lowess_imputed'] = df_lowess['total_cas_confirmes']
df_lowess.loc[df_lowess['total_cas_confirmes'].isna(
), 'total_cas_lowess_imputed'] = df_lowess['lowess_filled']

df_train = df_train.merge(
    df_lowess[['date', 'total_cas_lowess_imputed']].rename(
        columns={'total_cas_lowess_imputed': 'lowess'}),
    on='date',
    how='left'
)

# fill the NaN values in the train set using the forward fill & bacward fill method
df_train['ffill'] = df_train['total_cas_confirmes'].ffill()
df_train['bfill'] = df_train['total_cas_confirmes'].bfill()

# evaluate the 5 methods using MSE metrics
mse_mv_avg = mean_squared_error(
    df_test['total_cas_confirmes'], df_train['mv_avg'])
mse_mv_avg_b = mean_squared_error(
    df_test['total_cas_confirmes'], df_train['mv_avg_b'])
mse_linear_interp = mean_squared_error(
    df_test['total_cas_confirmes'], df_train['linear_interp'])
mse_lowess = mean_squared_error(
    df_test['total_cas_confirmes'], df_train['lowess'])
mse_ffill = mean_squared_error(
    df_test['total_cas_confirmes'], df_train['ffill'])
mse_bfill = mean_squared_error(
    df_test['total_cas_confirmes'], df_train['bfill'])

# create a dictionary with the results to use them in an histogram
mse_scores = {
    'Moving average': mse_mv_avg,
    'Backward moving average': mse_mv_avg_b,
    'Linear interpolation': mse_linear_interp,
    'LOESS/LOWESS': mse_lowess,
    'Forward fill': mse_ffill,
    'Backward fill': mse_bfill
}

# plot
fig, axes = plt.subplots(2, 1, figsize=(6, 12))

# plot the original values
axes[0].plot(df_test['date'], df_test['total_cas_confirmes'],
             label='Original', color='blue')
# plot the moving average
axes[0].plot(df_train['date'], df_train['mv_avg'],
             label='Moving average', color='green', alpha=0.5)
# plot the backward moving average
axes[0].plot(df_train['date'], df_train['mv_avg_b'],
             label='Backward moving average', color='red', alpha=0.5)
# plot the linear interpolation
axes[0].plot(df_train['date'], df_train['linear_interp'],
             label='Linear interpolation', color='purple', alpha=0.5)
# plot the LOWESS
axes[0].plot(df_train['date'], df_train['lowess'],
             label='LOESS/LOWESS', color='orange', alpha=0.5)
# plot the forward fill
axes[0].plot(df_train['date'], df_train['ffill'],
             label='Forward fill', color='pink', alpha=0.5)
# plot the backward fill
axes[0].plot(df_train['date'], df_train['bfill'],
             label='Backward fill', color='brown', alpha=0.5)

axes[0].set_xlabel("Date")
axes[0].set_ylabel("Total confirmed cases")
axes[0].set_title("Moving average on the train set")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
axes[0].xaxis.set_major_locator(mdates.MonthLocator())
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# plot the histograms of the MSE
axes[1].bar(mse_scores.keys(), mse_scores.values(), color='skyblue')
axes[1].set_title("MSE of the different methods")
axes[1].set_xlabel("Methods")
axes[1].set_ylabel("MSE")
axes[1].set_yscale('log')
axes[1].tick_params(axis='x', rotation=90)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Make a summary of the MSE scores
print("Summary of the MSE scores:")
print("===================================================\n")
# Order the results
mse_scores = dict(sorted(mse_scores.items(), key=lambda item: item[1]))
# Display the results
for i, (method, mse) in enumerate(mse_scores.items(), start=1):
    print(f"{i} -> {method}: {mse:.2f}")

"""Conclusion:

As the human eyes suspected, the `Linear interpolation` and 
the `LOESS/LOWESS`are the 2 best way to fill the missing values 
in the time serie.

It also appear that there is a huge gap between these 2 methods and 
the others, going from 82.752 to 425.571 to 10.410.656. 

Even though the number of confirmed cases are very high, 
which can explain the high values of the MSE, the results are speaking 
for themselves. """
