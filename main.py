# 导入库
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

# 忽略警告
warnings.filterwarnings('ignore')


# 原始变量变化情况
def plot_variable(data, variable):
    for i in range(len(variable)):
        plt.figure()
        plt.plot(data.loc[:, variable[i]])
        plt.title(variable[i])
    plt.show()


# 训练数据和测试数据生成
def train_test(data, mode):
    # 训练数据和测试数据划分
    data_train, data_test = train_test_split(data[:, [0, 3]], test_size=0.2, shuffle=False)
    train_size, test_size = data_train.shape[0], data_test.shape[0]

    # 用任意组合来构造训练数据
    if mode == '1':
        train_set = np.zeros((train_size * (train_size - 1), 4))
        temp = np.concatenate((data_train, data_train))
        index = 0
        for i in range(train_size - 1):
            train_set[index:index + train_size, :] = np.concatenate((data_train, temp[i + 1:i + train_size + 1, :]),
                                                                    axis=1)
            index = index + train_size
        test_set = np.concatenate((data_test[1:, :], np.tile(data_test[0, :], (test_size - 1, 1))), axis=1)

    # 按照时序顺序来构造训练数据（由于相邻时刻油高接近，导致训练的模型直接使用前一时刻的油高作为当前油高的预测结果）
    elif mode == '2':
        train_set = np.concatenate((data_train[1:, :], data_train[:-1, :]), axis=1)
        test_set = np.concatenate((data_test[1:, :], data_test[:-1, :]), axis=1)

    # 将前几个时刻的温度和当前温度用于预测当前油高
    elif mode == '3':
        length = 0
        train_set = data_train[length:, :]
        test_set = data_test[length:, :]
        for i in range(length):
            train_set = np.concatenate((train_set, data_train[i:i + train_set.shape[0], 1].reshape(-1, 1)), axis=1)
            test_set = np.concatenate((test_set, data_test[i:i + test_set.shape[0], 1].reshape(-1, 1)), axis=1)

    return train_set, test_set


# 主函数
def main():
    # demo
    print('=====Import demo=====')
    file_folder = 'demo/'
    data_demo = []
    data_demo.append(pd.read_excel(file_folder + '图影站BB004液位仪历史记录（92#）.xls', index_col='读取时间').iloc[376:])
    data_demo.append(pd.read_excel(file_folder + '图影站BB005液位仪历史记录（92#）.xls', index_col='读取时间').iloc[:399])
    data_demo.append(pd.read_excel(file_folder + '图影站BB005液位仪历史记录（92#）.xls', index_col='读取时间').iloc[675:])
    for i in range(len(data_demo)):
        data_demo[i].index = pd.to_datetime(data_demo[i].index)
    variable = ['油高(单位:mm)', '油水总体积', '纯油体积', '温度', '水高', '视密度', '泄漏值']
    # plot_variable(data_demo[0], variable)
    # plot_variable(data_demo[1], variable)
    # plot_variable(data_demo[2], variable)

    # 液位仪系统
    print('=====Import liquid level meter=====')
    file_folder = '液位仪系统数据'
    file_name = os.listdir(file_folder)
    data_list = [pd.read_excel(file_folder + '/' + i, header=2) for i in file_name]
    data = pd.DataFrame(np.vstack(data_list), columns=data_list[0].columns)
    data['读取时间'] = pd.to_datetime(data['读取时间'])
    g = data.groupby('油罐编码')
    data_tank = [i.set_index('读取时间').sort_index() for _, i in g]
    variable = ['油高(单位:mm)', '油水总体积', '纯油体积', '标准体积', '温度', '水高', '视密度', '泄漏值']
    # for i in range(len(data_tank)):
    #     plot_variable(data_tank[i], variable)

    # 油气回收系统
    print('=====Import oil-gas recycle=====')
    file_folder = '油气回收系统数据/'
    data = pd.read_excel(file_folder + '湖州长兴图影站油气回收表_05.21-09.11.xls', header=2, index_col=0).sort_index()
    data.index = pd.to_datetime(data.index)
    g1 = data.groupby('加油机标识')
    data_machine = [i for _, i in g1]
    g2 = data.groupby('加油枪标识')
    data_gun = [i for _, i in g2]
    variable = ['加油机标识', '加油枪标识', '气液比', '油气流速', '油气流量', '燃气流速', '燃气流量', '液阻']
    # for i in range(len(data_machine)):
    #     plot_variable(data_machine[i], variable)

    # 数据划分
    train_size, test_size = [0, ], [0, ]
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in data_demo:
        train_set, test_set = train_test(i.loc[:, variable].values, mode='1')
        train_size.append(train_set.shape[0])
        test_size.append(test_set.shape[0])
        X_train.append(train_set[:, 1:])
        y_train.append(train_set[:, 0])
        X_test.append(test_set[:, 1:])
        y_test.append(test_set[:, 0])
    train_size = np.cumsum(train_size)
    test_size = np.cumsum(test_size)
    X_train = np.concatenate(X_train).reshape(-1, train_set.shape[1] - 1)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test).reshape(-1, test_set.shape[1] - 1)
    y_test = np.concatenate(y_test)

    # 标准化
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # 软测量模型训练和测试
    regressor = PLSRegression(n_components=X_train.shape[1]).fit(X_train, y_train)
    y_fit = regressor.predict(X_train)
    y_pred = regressor.predict(X_test)
    for i in range(len(data_demo)):
        plt.figure()
        plt.plot(y_fit[train_size[i]:train_size[i + 1]])
        plt.plot(y_train[train_size[i]:train_size[i + 1]])
        plt.grid()
        plt.legend(['Fit', 'Ground Truth'])
        plt.title('Fitting performance for oil level (data {})'.format(i + 1))
        plt.figure()
        plt.plot(y_pred[test_size[i]:test_size[i + 1]])
        plt.plot(y_test[test_size[i]:test_size[i + 1]])
        plt.grid()
        plt.legend(['Prediction', 'Ground Truth'])
        plt.title('Predicting performance for oil level (data {})'.format(i + 1))
        plt.show()
    print('Coefficient: {}'.format(regressor.coef_))
    print('Fit: {:.2f}% {:.3f}'.format(r2_score(y_train, y_fit) * 100, np.sqrt(mean_squared_error(y_train, y_fit))))
    print('Pred: {:.2f}% {:.3f}'.format(r2_score(y_test, y_pred) * 100, np.sqrt(mean_squared_error(y_test, y_pred))))


if __name__ == '__main__':
    main()
