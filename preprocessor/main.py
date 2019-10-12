import numpy as np
import pandas as pd
from datetime import timedelta

df_companies = pd.read_csv('../datasets/company-codes.csv')
df_bovespa = pd.read_csv('../datasets/kaggle/bovespa.csv')
df_articles = pd.read_csv('../datasets/kaggle/articles.csv')

def isBigger(open, close):
    if close > open:
        return 1
    if open > close:
        return -1
    else:
        return 0

def getEffectDate(date):
    effectDate = date
    while (df_bovespa[df_bovespa.date == effectDate].size < 1):
        effectDate = effectDate + timedelta(days=1)
    return effectDate

df_articles = df_articles[df_articles.category == 'mercado']
artigos_unused_columns = ['text', 'category', 'subcategory', 'link']
df_articles.drop(artigos_unused_columns, inplace=True, axis=1)
df_articles['date'] = df_articles.apply(lambda row: np.int64(row['date'].replace('-', '')), axis=1)
df_articles['date'] = pd.to_datetime(df_articles['date'].astype(str), format='%Y%m%d')

df_bovespa.columns = map(str.lower, df_bovespa.columns)
df_bovespa = df_bovespa[df_bovespa.codneg.str.strip().isin(df_companies.code)]
df_bovespa['date'] = pd.to_datetime(df_bovespa['date'].astype(str), format='%Y%m%d')
df_bovespa = df_bovespa[df_bovespa.date >= df_articles.date.min()]
bovespa_unused_columns = ['company', 'typereg', 'bdicode', 'markettype', 'spec', 'prazot', 'currency', 'max', 'min', 'med', 'preofc', 'preofv', 'totneg', 'quatot']
df_bovespa.drop(bovespa_unused_columns, inplace=True, axis=1)
df_bovespa = df_bovespa.sort_values('date')

df_articles['date'] = df_articles.apply(lambda row: getEffectDate(row.date), axis=1)
df_articles = df_articles.sort_values('date')

# VALE3
df_bovespa_vale3 = df_bovespa[df_bovespa.codneg.str.strip() == 'VALE3']
df_bovespa_vale3 = df_bovespa_vale3.assign(close1d=df_bovespa_vale3['close'].transform( lambda group: group.shift(-1)))
df_bovespa_vale3 = df_bovespa_vale3.assign(close2d=df_bovespa_vale3['close'].transform( lambda group: group.shift(-2)))
df_bovespa_vale3 = df_bovespa_vale3.assign(close3d=df_bovespa_vale3['close'].transform( lambda group: group.shift(-3)))
df_bovespa_vale3 = df_bovespa_vale3.assign(close4d=df_bovespa_vale3['close'].transform( lambda group: group.shift(-4)))
df_bovespa_vale3 = df_bovespa_vale3.assign(close5d=df_bovespa_vale3['close'].transform( lambda group: group.shift(-5)))
df_bovespa_vale3.drop('close', inplace=True, axis=1)
df_vale3 = pd.merge(df_articles, df_bovespa_vale3, on='date', how='left')
