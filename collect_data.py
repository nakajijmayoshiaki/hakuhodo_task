# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from IPython import embed

task = "finish"

"""testデータの内容から属性をだす"""
if task=="predict":
    data = pd.read_excel("data/test.xlsx")

    df_gender = pd.read_csv("result/gender.csv")
    df_gender.columns = ["", "gender"]
    df_gender = df_gender.iloc[:,1:]
    df_age = pd.read_csv("result/age.csv")
    df_age.columns = ["", "age"]
    df_age = df_age.iloc[:,1:]
    df_job = pd.read_csv("result/job.csv")
    df_job.columns = ["", "job"]
    df_job = df_job.iloc[:,1:]

    data = pd.concat([data, df_gender], axis=1)#入力に必要なものをまとめる
    data = pd.concat([data, df_age], axis=1)#入力に必要なものをまとめる
    data = pd.concat([data, df_job], axis=1)#入力に必要なものをまとめる

    data.to_csv("result/predict.csv")

"""predictから特定のidだけ抽出して、タスクを完成させる部分"""

if task == "finish":
    data = pd.read_csv("result/predict.csv")
    df_out = pd.read_excel("data/sample_submission.xlsx")

    for i in range(len(df_out)):
        notfound = False
        for j in range(len(data)):
            if df_out.haku_id[i] == data.haku_id[j]:
                print("Data processed No", i)
                df_out.gender[i] = data.gender[j]
                df_out.age[i] = data.age[j]
                df_out.job[i] = data.job[j]
                notfound = True
                break
        if notfound == False:
            print("Error! Target data is not found in predict data")
            exit()

    df_out.to_csv("result/answer.csv")
