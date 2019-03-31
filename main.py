# -*- coding: utf-8 -*-

#2019年3月18日　博報堂の新卒課題　データ前処理用
import pandas as pd
import numpy as np

TryFlg = False #少ないデータで試す用
make_label = False
#print(df["haku_id"][10:50]) #job行の10~100を抽出
#print(df.query('haku_id == "2e56fce3465b40ec0484803e205adf45"')) #あるid の人の属性をだす

print("Reading xlsx Data...")
df = pd.read_excel("data/train.xlsx")
print("Reading xlsx Data has been finished")

"""
教師データラベルを作る
"""
if make_label == True:

    #ラベルを数値に変換して教師データとして保存する
    label = df.iloc[:,10:13]
    label.gender = df.gender.map({"男性":0, "女性":1})
    label.age = df.age.map({"20代":0, "30代":1, "40代":2, "50代":3})
    label.job = df.job.map({"ＰＲ職":0, "ナレッジ開発職":1, "マネジメントプロデュース職":2, "クリエイティブ職":3, "アカウントプロデュース職":4, "メディアプラニング職":5, "メディアプロデュース職":6, "ストラテジックプラニング職":7, "コンテンツプロデュース職":8, "ビジネスディベロップメント職":9})

    print("Making label data...")
    label.to_csv("data/label_data.csv")#全体のラベルデータ

    #Gender Label
    matrix = np.zeros((len(label.index),2)) #データの型をnumpyで作る
    label_gender = pd.DataFrame(matrix, columns={"男性","女性"})
    for i in range(len(label.index)):
        if label.gender[i] == 1:
            label_gender.iloc[i,0] = 1 #女性に0
        else:
            label_gender.iloc[i,1] = 1 #男性に1

    print("Making gender label data...")
    label_gender.to_csv("data/label_gender.csv")

    #Age Label
    matrix = np.zeros((len(label.index),4)) #データの型をnumpyで作る
    label_age = pd.DataFrame(matrix, columns={"20代","30代","40代","50代"})
    for i in range(len(label.index)):
        for j in range(len(label_age.columns)):
            if label.age[i] == j:
                label_age.iloc[i,j] = 1

    print("Making age label data...")
    label_age.to_csv("data/label_age.csv")

    #Job Label
    matrix = np.zeros((len(label.index),10)) #データの型をnumpyで作る
    label_job = pd.DataFrame(matrix, columns={"PR職", "ナレッジ開発職", "マネジメントプロデュース職", "クリエイティブ職", "アカウントプロデュース職", "メディアプラニング職", "メディアプロデュース職", "ストラテジックプラニング職", "コンテンツプロデュース職", "ビジネスディベロップメント職"})
    for i in range(len(label.index)):
        for j in range(len(label_job.columns)):
            if label.job[i] == j:
                label_job.iloc[i,j] = 1

    print("Making job data...")
    label_job.to_csv("data/label_job.csv")

    exit()

"""
Trainデータを作る
"""

if TryFlg == True:
    df = df.head(1000) #テスト用 #最初の1000行だけ使う

#train データの必要な部分を取ってくる
df_time = df.iloc[:,3:5] #hourとminを取得
df_other = df.iloc[:,6:9] #priとminを取得
df = pd.concat([df_time, df_other], axis=1)#入力に必要なものをまとめる

length_train = len(df.index) #trainのデータ数を保存しておいてあとで切り離す

#test データの必要な部分を取ってくる
df_test = pd.read_excel("data/test.xlsx")
df_time = df_test.iloc[:,3:5] #hourとminを取得
df_other = df_test.iloc[:,6:9] #priとamountとnameを取得
df_test = pd.concat([df_time, df_other], axis=1)

df = pd.concat([df, df_test], ignore_index=True) #testとtrainを縦に合体 ignore_indexでindexを振り直す

#####購買数が100以下の商品名をその他に変換し、数値に変換する前処理######
#1. 辞書に購入品目と数を入れる
menu_list = [] #出て来たメニュー名を追記

for i in range(len(df)):
    found = False
    if i == 0: #最初は入れる
        menu_list.append(df["menu_name"][i])
        menu_dic = {df["menu_name"][i]: 1} #dictionaryの作成
    else:
        for j in range(len(menu_list)):
            if menu_list[j] == df["menu_name"][i]:
                menu_dic[menu_list[j]] = menu_dic[menu_list[j]]+1 #見つかったら個数を増やしてbreak
                found = True
                break
        if found == False: #過去のリストに見つからなかったら追加
            print(df["menu_name"][i], "Added", i)
            menu_list.append(df["menu_name"][i])
            menu_dic.update({df["menu_name"][i]:1})

#2. 購入品目と購入数から100以下の[以外]の品目のみのリストを作る
new_menu_list = []
for i in range(len(menu_dic)):
    if menu_dic[menu_list[i]] >=  0: #ここの閾値がハイパーパラメータ
        new_menu_list.append(menu_list[i])

print(new_menu_list)

#3. DataFrameの品目から、上記リスト以外のものを「その他」に変換する
for i in range(len(df.index)):
    overFlg = False
    for j in range(len(new_menu_list)):
        if new_menu_list[j] == df["menu_name"][i]: #リストに名前があったら
            overFlg = True #その他回避
            break #ループ抜ける

    if overFlg == False: #最後までloopが回ったらその他にする
        df.iloc[i,4] = "その他" #pandasでは df[df.col1 == 2].col1 = 100 のように、2回以上に分けて抽出した結果に何かを代入しても、元のdfの値は置き換わらない。

#4. 購買メニューを数値化する
df.menu_name = df.menu_name.astype("category")
df.menu_name = df.menu_name.cat.codes

"""正規化"""
print(df)
for i in range(len(df.columns)):
    df.iloc[:,i] = df.iloc[:,i]/df.iloc[:,i].max()
print(df)

#testとtrainに切り離して保存
df_test = df.iloc[length_train:len(df.index),:]
df = df.iloc[:length_train,:]

print("Saving input data...")
df.to_csv("data/input_data0.csv")
print("Saving test data...")
df_test.to_csv("data/test_data.csv")

print("Done")
#df.to_csv("data/new_data.csv", encoding="utf-8")


#わかったこと　→　職種: 10職,  期間: 2018年11月~2019年1月　
