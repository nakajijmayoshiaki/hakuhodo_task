# -*- coding: utf-8 -*-
from chainer import Chain, optimizers, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython import embed

#from sklearn import datasets, model_selection

input_dim = 5 #hour min price amount name
hid_dim = 40
num_epoch = 1000
NN_type = "gender" #学習する属性を選択

#iris = datasets.load_iris()

class NNforGender(Chain):
    """ 性別判定用ニューラルネットワーク"""
    def __init__(self, n_hid=100, n_in=5 ,n_out=2):
        super().__init__()

        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hid)
            self.l2 = L.Linear(n_hid, int(n_hid/2))
            self.l3 = L.Linear(int(n_hid/2), n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class NNforJob(Chain):
    """ 職種判定用ニューラルネットワーク"""
    def __init__(self, n_hid=100, n_in=5 ,n_out=10):
        super().__init__()

        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hid)
            self.l2 = L.Linear(n_hid, int(n_hid/2))
            self.l3 = L.Linear(int(n_hid/2), n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class NNforAge(Chain):
    """年齢判定用ニューラルネットワーク"""
    def __init__(self, n_hid=100, n_in=5 ,n_out=4):
        super().__init__()

        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hid)
            self.l2 = L.Linear(n_hid, int(n_hid/2))
            self.l3 = L.Linear(int(n_hid/2), n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# read input data
df_input = pd.read_csv("data/input_data0.csv")
train_data = df_input.values #data frameからNumpyに変換
train_data = train_data[:,1:] #最初のindexを排除

df_test = pd.read_csv("data/test_data0.csv")
test_data = df_test.values #data frameからNumpyに変換
test_data = test_data[:,1:] #最初のindexを排除

df_label = pd.read_csv("data/label_data.csv")

if NN_type == "gender":
    Model = NNforGender(hid_dim, input_dim, 2)
    label_no = 0
    #df_label = pd.read_csv("data/label_gender.csv")
elif NN_type == "age":
    Model = NNforAge(hid_dim, input_dim, 4)
    #df_label = pd.read_csv("data/label_age.csv")
    label_no = 1
elif NN_type == "job":
    Model = NNforJob(hid_dim, input_dim, 10)
    #df_label = pd.read_csv("data/label_job.csv")
    label_no = 2
else:
    print("NN Type Error: Declear the Type of NN")
    exit()


train_label = df_label.values#data frameからNumpyに変換
train_label = train_label[:,1:] #最初のindexを排除

optimizer = optimizers.Adam()
optimizer.setup(Model)

# dataをVariableに変換します。float32は重要
train_data_variable = Variable(train_data.astype(np.float32))
train_label_variable = Variable(train_label[:,label_no].astype(np.int32))

test_data_variable = Variable(test_data.astype(np.float32))
#print(np.where(train_label != train_label))
#print(train_label[:,label_num])

# 学習の様子を記録するために、リストを作っておきます。
loss_log = []

for epoch in range(num_epoch):
    # パラメータの勾配を初期化
    Model.cleargrads()
    # フォワードプロパゲーション
    prod_label = Model(train_data_variable)
    # 損失関数を計算
    loss = F.softmax_cross_entropy(prod_label, train_label_variable)
    #print(prod_label, train_label_variable)
    # 損失関数の値を元に、パラメータの更新量を計算
    loss.backward()
    # パラメータを更新
    optimizer.update()
    # 損失関数の値を保存しておきましょう。後で確認できます。
    loss_log.append(loss.data)

    print("Epoch:", epoch, " Loss:", loss_log[epoch])

plt.plot(loss_log)
plt.show()

#以下テストプロット
y = Model(test_data_variable)

y = F.softmax(y)
result_label = np.argmax(y.data, 1)

# 正答率を計算
acc = np.sum(test_label == result_label) / np.sum(result_label.shape)
embed()

#結果を保存する
result = np.array(result_label)
df_result = pd.DataFrame(result, columns={NN_type})

if NN_type == "gender":
    df_result = df_result.gender.map({0:"男性", 1:"女性"})
elif NN_type == "age":
    df_result = df_result.age.map({0:"20代", 1:"30代", 2:"40代", 3:"50代"})
elif NN_type  == "job":
    df_result = df_result.job.map({0:"PR職", 1:"ナレッジ開発職", 2:"マネジメントプロデュース職", 3:"クリエイティブ職", 4:"アカウントプロデュース職", 5:"メディアプラニング職", 6:"メディアプロデュース職", 7:"ストラテジックプラニング職", 8:"コンテンツプロデュース職", 9:"ビジネスディベロップメント職"})
else:
    print("ERROR : NN_typeの値をチェックしてください。")
    exit()

df_result.to_csv("result/"+NN_type+".csv")
