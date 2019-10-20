from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from keras.models import Sequential
from keras.layers import Dense
import os

tic = datetime.now()

cell = [110, 30]


class Para:
    pca_component = 90
    input_cells = 178
    drop_rate = 0
    learning_rate = 0.15
    iteration = 201
    n_hidden_1 = cell[0]
    n_hidden_2 = cell[1]
    seed_number = 1
    path_results = 'D:/QuantCN/nn_result/nn_together/等权组合/Oct_Week2/Sat/test4/' + str(n_hidden_1) + '-' \
                   + str(n_hidden_2) + '-seed' + str(seed_number) + '/'


para = Para()
tf.set_random_seed(para.seed_number)
np.random.seed(para.seed_number)
# make a path to storw the train results
os.mkdir(para.path_results)
info_file = open(para.path_results + 'info.txt', 'a')
info_file.write('input =' + str(para.input_cells) + '\n' + 'pca_select = ' + str(para.pca_component) + '\n' +
                'number of cells in layer1 = ' + str(para.n_hidden_1) + '\n' + 'number of cells in layer2 = ' +
                str(para.n_hidden_2) + '\n' + 'drop_rate = ' + str(para.drop_rate) + '\n' + 'learning_rate = ' +
                str(para.learning_rate) + '\n' + 'iteration = ' + str(para.iteration) + '\n')
info_file.close()

filename1 = 'D:/QuantCN/nn_result/nn_together/等权组合/Oct_Week2/Mon/factors/features_label.csv'
df = pd.read_csv(filename1, encoding='gb18030', low_memory=False)

# first 4 columns are labels: 'OptionCode', 'EndDate', 'rtn', 'rtn_roll_-1'
df['EndDate'] = df.loc[:, 'EndDate'].apply(lambda x: pd.Timestamp(x))
df = df.sort_values(by=['EndDate'], ascending=True)
df = df.reset_index(drop=True)
df_row, df_col = df.shape

features_columns = df.iloc[:, 4:].columns.to_list()
option = df.OptionCode.unique()
num_option = len(option)
End_Date = df.EndDate.unique()
num_date = len(End_Date)

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    drop_rate = tf.compat.v1.placeholder(tf.float32)
    xs = tf.compat.v1.placeholder(tf.float32, [None, para.pca_component], name='x_input')
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_input')
    x_raw = tf.compat.v1.placeholder(tf.float32, [None, para.input_cells])

x_mean, x_var = tf.nn.moments(x_raw, [0])
x_norm = tf.nn.batch_normalization(x_raw, mean=x_mean, variance=x_var, offset=None, scale=1, variance_epsilon=0.001)

o1 = tf.layers.dense(xs, para.n_hidden_1, tf.nn.tanh)
d1 = tf.layers.dropout(o1, rate=para.drop_rate, training=False)

o2 = tf.layers.dense(d1, para.n_hidden_2, tf.nn.tanh)
d2 = tf.layers.dropout(o2, rate=para.drop_rate, training=False)
out = tf.layers.dense(d2, 1)

loss = 0.5 * tf.losses.mean_squared_error(ys, out)
train_step = tf.compat.v1.train.AdamOptimizer(para.learning_rate).minimize(loss)
init = tf.compat.v1.global_variables_initializer()

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
sess = tf.Session(config=session_conf)

pred_df = pd.DataFrame(columns=['OptionCode', 'EndDate', 'rtn', 'rtn_roll_-1'] + list(
    range(para.pca_component)) + ['pred_rtn'])

for i in range(251, num_date):
    print('Train:' + str(End_Date[i]))
    error_file = open(para.path_results + 'error.txt', 'a')
    error_file.write(str(End_Date[i]) + '\n')
    df2 = df.loc[np.isin(df.loc[:, 'EndDate'].values, End_Date[i - 251: i + 1])]
    df2.reset_index(drop=True, inplace=True)
    x_factors = df2.iloc[:, 4:].values
    x_label = df2.iloc[:, :4]

    x_factors = sess.run(x_norm, feed_dict={x_raw: x_factors})
    pca = decomposition.PCA(n_components=para.pca_component)
    pca.fit(x_factors)
    x_factors = pca.transform(x_factors)

    x_factors_df = pd.DataFrame(x_factors, columns=list(range(para.pca_component)))
    df3 = pd.concat([x_label, x_factors_df], axis=1)

    df4 = df3.loc[np.isin(df3.loc[:, 'EndDate'].values, End_Date[i - 251: i - 1])]
    x_data = df4.iloc[:, 4:].values
    x_data_pred_df = df3.loc[np.isin(df3.loc[:, 'EndDate'].values, End_Date[i])]

    y_data = df4.loc[:, 'rtn_roll_-1'].values.reshape(df4.iloc[:, 0].size, 1)

    sess.run(init)

    for k in range(para.iteration):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data, drop_rate: para.drop_rate})

        if k % 50 == 0:
            error_file.write(
                str(sess.run(loss, feed_dict={xs: x_data, ys: y_data, drop_rate: para.drop_rate})) + '\n')

    error_file.close()

    pred_inputs = x_data_pred_df.iloc[:, 4:].values
    pred_rtn = sess.run(out, feed_dict={xs: pred_inputs, drop_rate: 0})

    num_col = x_data_pred_df.shape[1]
    x_data_pred_df.insert(loc=num_col, column='pred_rtn', value=pred_rtn)
    pred_df = pd.concat([pred_df, x_data_pred_df], axis=0, ignore_index=True)

pred_df_all = pd.DataFrame()
os.mkdir(para.path_results + 'pred/')
os.mkdir(para.path_results + 'graphs/')
for j in range(num_option):
    code = option[j]
    print('plot the graph indicating the net value for option ' + str(code))
    sub_df = pred_df.loc[pred_df.loc[:, 'OptionCode'] == code]
    sub_df_size = sub_df.iloc[:, 0].size
    if sub_df_size > 0:
        sub_df.reset_index(drop=True, inplace=True)

        pred_rtn = sub_df.loc[:, 'pred_rtn'].values
        tmp_indicator = np.sign(pred_rtn)
        indicator = np.roll(tmp_indicator, 1)
        indicator[0] = 0

        actual_rtn = sub_df.loc[:, 'rtn'].values * indicator

        net_value = (actual_rtn + 1).cumprod()

        tmp_max = np.maximum.accumulate(net_value)
        draw_down = net_value / tmp_max - 1
        daily_indicator = pd.DataFrame({'indicator': indicator, 'actual_rtn': actual_rtn, 'net_value': net_value,
             'draw_down': draw_down})

        sub_df = pd.concat([sub_df, daily_indicator], axis=1)
        sub_df.to_excel(para.path_results + 'pred/' + str(code) + '.xlsx', encoding='gb18030', index=False)
        pred_df_all = pd.concat([pred_df_all, sub_df], axis=0, ignore_index=True)
        # plot
        x = list(range(sub_df_size))
        y = net_value
        z = draw_down

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        l1, = ax1.plot(x, y, 'b', label='net value')
        ax1.set_ylabel('Net Value')
        ax1.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2])
        ax1.set_title(str(code))

        ax2 = ax1.twinx()  # this is the important function
        l2, = ax2.plot(x, z, 'r', label='draw down')
        ax2.fill_between(x, z, where=(z < 0), facecolor='red')
        ax2.set_ylabel('Draw Down')
        ax2.set_yticks([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0])
        Date = sub_df.EndDate.unique()
        num_Date = len(Date)
        Date = pd.DataFrame(Date)
        Date = Date.loc[:, 0].apply(lambda x: str(x)[:-9])
        plt.xticks(ticks=list(np.linspace(0, num_Date, 5, endpoint=False, dtype=int)), labels=list(
            Date.loc[list(np.linspace(0, num_Date, 5, endpoint=False, dtype=int))].values))
        plt.xticks(rotation=90)
        plt.legend(handles=[l1, l2], labels=['net value', 'draw down'], loc='lower left')
        plt.savefig(para.path_results + 'graphs/' + str(int(code)) + '.png', dpi=120)
        plt.close(fig)

pred_df_all.to_excel(para.path_results + 'pred_all.xlsx', encoding='gb18030', index=False)

Date1 = pred_df_all.EndDate.unique()
num1 = len(Date1)
Date = pd.DataFrame(Date1)
Date = Date.loc[:, 0].apply(lambda x: str(x)[:-9])
rtn = np.zeros(num1)

for i in range(num1):
    sub_df = pred_df_all.loc[pred_df_all.loc[:, 'EndDate'] == Date1[i]]
    rtn[i] = np.mean(sub_df.loc[:, 'actual_rtn'].values)

net_value_all = (1 + rtn).cumprod()

tmp_max_all = np.maximum.accumulate(net_value_all)
draw_down_all = net_value_all / tmp_max_all - 1

data = {'date': Date, 'rtn': list(rtn), 'net_value': list(net_value_all), 'draw_down': list(draw_down_all)}
port_indicator = pd.DataFrame(data)

port_indicator.to_excel(para.path_results + 'port_indicator.xlsx', encoding='gb18030', index=False)

x = range(1, num1 + 1)
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
l1, = ax1.plot(x, net_value_all, 'b', label='net value')
ax1.set_ylabel('Net Value')
ax1.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2])
ax1.set_title('portfolio indicators')

ax2 = ax1.twinx()  # this is the important function
l2, = ax2.plot(x, draw_down_all, 'r', label='draw down')
ax2.fill_between(x, draw_down_all, where=(draw_down_all < 0), facecolor='red')

ax2.set_ylabel('Draw Down')
ax2.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0])
ax2.set_xlabel('End Date')

plt.xticks(ticks=list(np.linspace(0, num1, 5, endpoint=False, dtype=int)), labels=list(
    Date.loc[list(np.linspace(0, num1, 5, endpoint=False, dtype=int))].values))
plt.legend(handles=[l1, l2], labels=['net value', 'draw down'], loc='lower left')
plt.savefig(para.path_results + 'port_indicator.png', dpi=120)
plt.close(fig2)
toc = datetime.now()
print(toc)
print(toc - tic)
