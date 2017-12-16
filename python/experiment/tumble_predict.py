import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier as gdbt
from sklearn.metrics import precision_score, accuracy_score, recall_score
from matplotlib import pyplot as plt


def get_max_crash(series):
    curr_max = series[0]
    max_crash = 0
    for i in range(1, len(series)):
        curr_max = max(curr_max, series[i])
        max_crash = min(max_crash, series[i] - curr_max)
        
    return abs(max_crash / curr_max)


def get_features(dataframe, upper_shadow_threshold=0.2):
    feature = list()
    close_diff = dataframe['close'].diff()
    close_diff[close_diff >= 0] = 0
    close_diff[close_diff < 0] = 1
    consecutive_decrease = close_diff * (close_diff.groupby((close_diff != close_diff.shift()).cumsum()).cumcount() 
                                         + 1)
    feature.append(consecutive_decrease.max())
        
    ma5_diff = dataframe['ma5'].diff()
    ma10_diff = dataframe['ma10'].diff()
    ma20_diff = dataframe['ma20'].diff()
    ma5_diff[ma5_diff >= 0] = 0
    ma10_diff[ma10_diff >= 0] = 0
    ma20_diff[ma20_diff >= 0] = 0
    ma5_diff[ma5_diff < 0] = 1
    ma10_diff[ma10_diff < 0] = 1
    ma20_diff[ma20_diff < 0] = 1
    
    ma5_decrease = ma5_diff * (ma5_diff.groupby((ma5_diff != ma5_diff.shift()).cumsum()).cumcount() + 1)
    ma10_decrease = ma10_diff * (ma10_diff.groupby((ma10_diff != ma10_diff.shift()).cumsum()).cumcount() + 1)
    ma20_decrease = ma20_diff * (ma20_diff.groupby((ma20_diff != ma20_diff.shift()).cumsum()).cumcount() + 1)
    
    feature.append(ma5_decrease.max())
    feature.append(ma10_decrease.max())
    feature.append(ma20_decrease.max())

    ma10 = dataframe['ma10']
    ma20 = dataframe['ma20']
    condition_green = dataframe['close'] < dataframe['open']
    
    if (ma10[-1] > ma10[0]) and (ma20[-1] > ma20[0]):
        condition_upper_shadow = (dataframe['high'] - dataframe['open']) / (dataframe['open'] - dataframe['close'])
        condition_upper_shadow = condition_upper_shadow > upper_shadow_threshold
        condition = condition_green & condition_upper_shadow
        feature.append(condition[condition == True].shape[0])
    else:
        feature.append(0)
    
    ma_max = dataframe[['ma5', 'ma10', 'ma20']].max(1)
    ma_min = dataframe[['ma5', 'ma10', 'ma20']].min(1)
    condition_sickle = (dataframe['open'] > ma_max) & (dataframe['close'] < ma_min)
    condition_sickle = condition_green & condition_sickle
    feature.append(condition_sickle[condition_sickle == True].shape[0] > 0)
    return feature
    

def generate_train_test_samples(df, train_interval, test_interval, upper_shadow_threshold=0.2):
    features = list()
    labels = list()
    for i in range(0, df.shape[0]-train_interval-test_interval):
        df_train = df.iloc[i: i+train_interval]
        feature = get_features(df_train, upper_shadow_threshold=upper_shadow_threshold)
        features.append(feature)
        df_test = df.iloc[i+train_interval: i+train_interval+test_interval]
        max_crash = get_max_crash(df_test['close'])
        labels.append(max_crash)
    return np.array(features), np.array(labels)


def tumble_predict_experiment(model, df_train, df_test, train_interval, test_interval, upper_shadow_threshold,
                              crash_threshold):

    x_train, y_train = generate_train_test_samples(df_train, train_interval, test_interval,
                                                   upper_shadow_threshold=upper_shadow_threshold)
    x_test, y_test = generate_train_test_samples(df_test, train_interval, test_interval,
                                                 upper_shadow_threshold=upper_shadow_threshold)

    label_train = np.copy(y_train)
    label_train[label_train > crash_threshold] = 1
    label_train[label_train <= crash_threshold] = 0

    label_test = np.copy(y_test)
    label_test[label_test > crash_threshold] = 1
    label_test[label_test <= crash_threshold] = 0

    model.fit(x_train, label_train)
    label_train_predict = model.predict(x_train)
    label_test_predict = model.predict(x_test)

    print('In train data:')
    _experiment(df_train, label_train, label_train_predict)
    print('In test data:')
    _experiment(df_test, label_test, label_test_predict)


def _experiment(df, label, label_predict):
    print('All samples num: {}'.format(label.shape[0]))
    print('Positive samples num: {}'.format(label[label == 1].shape[0]))

    p = precision_score(label, label_predict)
    r = recall_score(label, label_predict)
    print('Precision: {}, recall: {}'.format(p, r))

    tp_fp = label_predict[label_predict == 1].shape[0]
    tp = tp_fp * p
    fn = (1 - r) * tp / r
    print('TP: {}, FP: {}, FN: {}'.format(tp, tp_fp - tp, fn))

    positive_indices = [i for i, v in enumerate(label) if v == 1]
    predict_positive_indices = [i for i, v in enumerate(label_predict) if v == 1]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    fig.set_figwidth(16)
    fig.set_figheight(12)

    ax1.plot(df['close'].values)
    ax2.plot(df['close'].values)
    ax3.plot(df['close'].values)
    ax4.plot(df['close'].values)

    tp = [x for x in positive_indices if x in predict_positive_indices]
    fn = [x for x in positive_indices if x not in predict_positive_indices]
    fp = [x for x in predict_positive_indices if x not in positive_indices]

    print('tp: {}, fn: {}, np: {}'.format(len(tp), len(fn), len(fp)))

    for i in positive_indices:
        ax1.axvline(x=i, c='yellow')

    for i in tp:
        ax2.axvline(x=i, c='yellow')
    for i in fn:
        ax3.axvline(x=i, c='red')
    for i in fp:
        ax4.axvline(x=i, c='blue')
    
    plt.show()
    plt.close()

'''
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

positive_indices = [i for i, v in enumerate(label) if v == 1]

fig, ax = plt.subplots()
fig.set_tight_layout(True)
fig.set_figheight(6)
fig.set_figwidth(16)

# Plot a scatter that persists (isn't redrawn) and the initial line.
ax.plot(df['close'].values)

line1 = ax.axvline(x=positive_indices[0], c='red')
line2 = ax.axvline(x=positive_indices[0] + 42, c='yellow')
line3 = ax.axvline(x=positive_indices[0] + 63, c='blue')

def update(i):
    label = 'Crash rate: {}'.format(y[positive_indices[i]])
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    line1.set_xdata(positive_indices[i])
    line2.set_xdata(positive_indices[i] + 42)
    line3.set_xdata(positive_indices[i] + 63)
    ax.set_xlabel(label)
    return line1, line2, line3, ax

anim = FuncAnimation(fig, update, frames=len(positive_indices), interval=200)
anim.save('test1.gif', dpi=80, writer='imagemagick')
'''
