# まずはchainerのimport
from chainer import Chain, optimizers, Variable
import chainer.functions as F
import chainer.links as L

# 次はnumpyのimport
import numpy as np

# sklearnのimport
from sklearn import datasets, model_selection

# 可視化ライブラリのimport
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed

class MLP(Chain):
    """ 三層のニューラルネットワーク（MLP）
    """
    def __init__(self, n_hid=100, n_out=10):
        # Chainer.Chainクラスを継承して、Chainクラスの機能を使うためにsuper関数を使う
        super().__init__()

        # 結合重み（weight）やバイアス（bias）などのパラメータを持つ関数を定義
        with self.init_scope():
            self.l1 = L.Linear(4, n_hid)
            self.l2 = L.Linear(n_hid, n_out)

    def __call__(self, x):
        hid = F.relu(self.l1(x))
        return self.l2(hid)

# sklearnでirisデータを読み込み
iris = datasets.load_iris()

# データを教師データ、テストデータ、教師データの正解クラス、テストデータの正解クラスに分割
train_data, test_data, train_label, test_label = model_selection.train_test_split(iris.data.astype(np.float32), iris.target)


# 先程作ったニューラルネットのユニット数を指定して、インスタンスを作ります。
model = MLP(20, 3)

# ニューラルネットの学習方法を指定します。SGDは最も単純なものです。
optimizer = optimizers.SGD()

# 学習させたいパラメータを持ったChainをオプティマイザーにセットします。
optimizer.setup(model)

# dataをVariableに変換します。
train_data_variable = Variable(train_data.astype(np.float32))
train_label_variable = Variable(train_label.astype(np.int32))

# 学習の様子を記録するために、リストを作っておきます。
loss_log = []
# 学習ステップです。ここでは200回行います。
for epoch in range(200):
    # パラメータの勾配を初期化
    model.cleargrads()
    # フォワードプロパゲーション
    prod_label = model(train_data_variable)
    # 損失関数を計算
    loss = F.softmax_cross_entropy(prod_label, train_label_variable)
    # 損失関数の値を元に、パラメータの更新量を計算
    loss.backward()
    # パラメータを更新
    optimizer.update()

    # 損失関数の値を保存しておきましょう。後で確認できます。
    loss_log.append(loss.data)


plt.plot(loss_log)
plt.show()


# テストデータをVaribleにしましょう
test_data_variable = Variable(test_data.astype(np.float32))

# フォワードプロパゲーション
y = model(test_data_variable)

embed()

# Softmax関数でクラス確率を出します
y = F.softmax(y)

# ３クラス中で一番確率の大きいものをデータが割り当てられたクラスとします
pred_label = np.argmax(y.data, 1)

# 正答率を計算
acc = np.sum(pred_label == test_label) / np.sum(test_label)

print(acc)
