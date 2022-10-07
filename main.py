import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter


def load_data(name):
    data = pd.read_csv(name)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    data['Mean'] = data.mean(axis=1)
    data_weekly_mean = data[data.columns].resample('W').mean()
    data_weekly_mean['Mean'] = data_weekly_mean.mean(axis=1)
    data_monthly_mean = data[data.columns].resample('M').mean()
    data_monthly_mean['Mean'] = data_monthly_mean.mean(axis=1)
    data_yearly_mean = data[data.columns].resample('Y').mean()
    data_yearly_mean['Mean'] = data_yearly_mean.mean(axis=1)

    return data, data_weekly_mean, data_monthly_mean, data_yearly_mean


def get_regression_line(eur_usd, huf_usd):
    m, b = np.polyfit(eur_usd, huf_usd, 1)
    return m, b


def plot_coefficient(window, m, start_date, end_date):
    if window == 60:
        plt.figure(1)
        plt.hlines(y=m, xmin=pd.to_datetime(start_date, format='%Y-%m-%d'),
                   xmax=pd.to_datetime(end_date, format='%Y-%m-%d'), color='b')

    elif window == 90:
        plt.figure(2)
        plt.hlines(y=m, xmin=pd.to_datetime(start_date, format='%Y-%m-%d'),
                   xmax=pd.to_datetime(end_date, format='%Y-%m-%d'), color='r')

    else:
        plt.figure(3)
        plt.hlines(y=m, xmin=pd.to_datetime(start_date, format='%Y-%m-%d'),
                   xmax=pd.to_datetime(end_date, format='%Y-%m-%d'), color='g')

    plt.xlim([pd.to_datetime('2020-01-01', format='%Y-%m-%d'),
              pd.to_datetime('2022-10-01', format='%Y-%m-%d')])

    plt.ylim([0, 4])


def rolling_n_days(start_date, end_date, df1, df2, column, window, frequency):
    for i in range(len(df1)):
        mask = (df1.index > start_date) & (df1.index <= end_date)
        masked_df1 = df1.loc[mask]
        masked_df2 = df2.loc[mask]
        m, b = get_regression_line(masked_df1[column], masked_df2[column])
        coefficients.loc[len(coefficients.index)] = [window, start_date, end_date, m, b]

        plot_coefficient(window, m, start_date, end_date)

        start_date = (pd.DatetimeIndex([start_date]) + pd.DateOffset(frequency)).format()[0]
        end_date = (pd.DatetimeIndex([start_date]) + pd.DateOffset(window)).format()[0]

    return coefficients


def get_constant(df, i):
    c = df._get_value(i, 'Coefficient')
    return c


def get_next_day(coefficients, end_date, window):
    if window == 60:
        next = (pd.DatetimeIndex([end_date]) + pd.DateOffset(3)).format()[0]
        next = coefficients[coefficients['Start date'] == next]
    elif window == 90:
        next = (pd.DatetimeIndex([end_date]) + pd.DateOffset(1)).format()[0]
        next = coefficients[coefficients['Start date'] == next]
    else:
        next = (pd.DatetimeIndex([end_date]) + pd.DateOffset(2)).format()[0]
        next = coefficients[coefficients['Start date'] == next]

    return next


def get_previous_day(coefficients, start_date):
    previous_60 = (pd.DatetimeIndex([start_date]) - pd.DateOffset(63)).format()[0]
    previous_60 = coefficients[coefficients['Start date'] == previous_60]
    previous_60 = previous_60[previous_60['Window'] == 60]
    previous_90 = (pd.DatetimeIndex([start_date]) - pd.DateOffset(91)).format()[0]
    previous_90 = coefficients[coefficients['Start date'] == previous_90]
    previous_90 = previous_90[previous_90['Window'] == 90]
    previous_180 = (pd.DatetimeIndex([start_date]) - pd.DateOffset(182)).format()[0]
    previous_180 = coefficients[coefficients['Start date'] == previous_180]
    previous_180 = previous_180[previous_180['Window'] == 180]

    df = pd.concat([previous_60, previous_90])
    return pd.concat([df, previous_180])


def min_coeff(next, previous_coeff):
    min_win = 60
    min_diff = 10

    for index, row in next.iterrows():
        if (previous_coeff - row['Coefficient']) < min_diff:
            min_win = row['Window']
            min_diff = previous_coeff - row['Coefficient']
    return next.loc[next['Window'] == min_win]


if __name__ == '__main__':
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True, figsize=(11, 10))
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )
    plot_params = dict(
        color="0.75",
        style=".-",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
        legend=False,
    )

    data_huf_usd,  data_huf_usd_weekly_mean, data_huf_usd_monthly_mean, data_huf_usd_yearly_mean = load_data('HistoricalPrices_HUFUSD.csv')
    data_eur_usd, data_eur_usd_weekly_mean, data_eur_usd_monthly_mean, data_eur_usd_yearly_mean = load_data('HistoricalPrices_EURUSD.csv')
    ref_point_huf_usd = data_huf_usd.loc['2020-01-01', 'Mean']
    ref_point_eur_usd = data_eur_usd.loc['2020-01-01', 'Mean']

    data_huf_usd_weekly_mean['Difference in %'] = data_huf_usd_weekly_mean['Mean'].div(data_huf_usd_weekly_mean['Mean'].shift(1))
    data_huf_usd_weekly_mean['Difference in %'] = (data_huf_usd_weekly_mean['Difference in %'] * 100) - 100
    data_huf_usd_weekly_mean['Difference in %'] = data_huf_usd_weekly_mean['Difference in %'].fillna(0)
    data_eur_usd_weekly_mean['Difference in %'] = data_eur_usd_weekly_mean['Mean'].div(data_eur_usd_weekly_mean['Mean'].shift(1))
    data_eur_usd_weekly_mean['Difference in %'] = (data_eur_usd_weekly_mean['Difference in %'] * 100) - 100
    data_eur_usd_weekly_mean['Difference in %'] = data_eur_usd_weekly_mean['Difference in %'].fillna(0)
    print(data_eur_usd_weekly_mean['Difference in %'])

    coefficients = pd.DataFrame(columns=['Window', 'Start date', 'End date', 'Coefficient', 'Constant'])

    for row in [60, 90, 180]:
        start_date = '2020-01-01'
        end_date = (pd.DatetimeIndex([start_date]) + pd.DateOffset(row)).format()[0]
        coefficients = rolling_n_days(start_date, end_date, data_eur_usd_weekly_mean, data_huf_usd_weekly_mean, 'Difference in %', row, 7)

    max_coeff = coefficients['Coefficient'].max()
    max_coeff_start_date = coefficients.loc[coefficients['Coefficient'] == max_coeff, 'Start date'].values[0]
    max_coeff_end_date = coefficients.loc[coefficients['Coefficient'] == max_coeff, 'End date'].values[0]
    max_coeff_start_date_index = coefficients.loc[coefficients['Coefficient'] == max_coeff, 'Start date'].index.values[0]
    max_coeff_end_date_index = coefficients.loc[coefficients['Coefficient'] == max_coeff, 'End date'].index.values[0]
    max_coeff_window = coefficients.loc[coefficients['Coefficient'] == max_coeff, 'Window'].values[0]

    #win_60 = mlines.Line2D([], [], color='blue', label='Window: 60')
    #win_90 = mlines.Line2D([], [], color='red', label='Window: 90')
    #win_180 = mlines.Line2D([], [], color='green', label='Window: 180')

    #plt.legend(handles=[win_60, win_90, win_180])
    #plt.legend(["Window: 60", "Window: 90", "Window: 180"])

    plt.figure(4)
    plt.hlines(y=max_coeff, xmin=pd.to_datetime(max_coeff_start_date, format='%Y-%m-%d'),
               xmax=pd.to_datetime(max_coeff_end_date, format='%Y-%m-%d'), color='g')
    plt.xlim([pd.to_datetime('2020-01-01', format='%Y-%m-%d'),
              pd.to_datetime('2022-10-01', format='%Y-%m-%d')])
    plt.ylim([0, 4])

    next = get_next_day(coefficients, max_coeff_end_date, max_coeff_window)
    nearest = next.loc[next['Window'] == 90]
    while nearest['Start date'].values[0] < '2022-09-28':
        print(nearest)
        plt.hlines(y=nearest['Coefficient'].values[0], xmin=pd.to_datetime(nearest['Start date'].values[0], format='%Y-%m-%d'),
                   xmax=pd.to_datetime(nearest['End date'].values[0], format='%Y-%m-%d'), color='g')
        plt.xlim([pd.to_datetime('2020-01-01', format='%Y-%m-%d'),
                  pd.to_datetime('2022-10-01', format='%Y-%m-%d')])
        plt.ylim([0, 4])
        next = get_next_day(coefficients, nearest['End date'].values[0], nearest['Window'].values[0])
        nearest = min_coeff(next, max_coeff)

    previous = get_previous_day(coefficients, max_coeff_start_date)
    print(previous)
    previous = min_coeff(previous, max_coeff)
    while True:
        print(previous)
        plt.hlines(y=previous['Coefficient'].values[0],
                   xmin=pd.to_datetime(previous['Start date'].values[0], format='%Y-%m-%d'),
                   xmax=pd.to_datetime(previous['End date'].values[0], format='%Y-%m-%d'), color='g')
        plt.xlim([pd.to_datetime('2020-01-01', format='%Y-%m-%d'),
                  pd.to_datetime('2022-10-01', format='%Y-%m-%d')])
        plt.ylim([0, 4])

        if pd.to_datetime(previous['Start date'].values[0], format='%Y-%m-%d') > pd.to_datetime('2020-02-19', format='%Y-%m-%d'):
            previous = get_previous_day(coefficients, previous['Start date'].values[0])
            print(previous)
            previous = min_coeff(previous, max_coeff)
        else:
            previous = coefficients[coefficients['Start date'] == '2020-01-01']
            plt.hlines(y=previous['Coefficient'].values[0],
                       xmin=pd.to_datetime(previous['Start date'].values[0], format='%Y-%m-%d'),
                       xmax=pd.to_datetime('2020-02-19', format='%Y-%m-%d'), color='g')
            plt.xlim([pd.to_datetime('2020-01-01', format='%Y-%m-%d'),
                      pd.to_datetime('2022-10-01', format='%Y-%m-%d')])
            plt.ylim([-7,4])
            break

    plt.plot(data_huf_usd_weekly_mean.index, data_huf_usd_weekly_mean['Difference in %'], color='r')
    plt.plot(data_eur_usd_weekly_mean.index, data_eur_usd_weekly_mean['Difference in %'])
    plt.show()
    coefficients.to_csv('HUF_USD_EUR_USD_Coefficients.csv')
