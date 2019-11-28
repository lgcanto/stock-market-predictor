import numpy as np
import pandas as pd
import csv
import os
from datetime import timedelta

OUT_DIR = './dataset_out'

df_companies = pd.read_csv('../datasets/company-codes.csv')
df_bovespa = pd.read_csv('../datasets/kaggle/bovespa.csv')
df_articles = pd.read_csv('../datasets/kaggle/articles.csv')

def getAppreciation(before, after):
    if after > before:
        return 1
    if before > after:
        return -1
    else:
        return 0

def getEffectDate(date):
    effectDate = date
    while (df_bovespa[df_bovespa.date == effectDate].size < 1):
        effectDate = effectDate + timedelta(days=1)
    return effectDate

def exportToTSV(dataframe, filename):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    fullpath = '%s/%s' % (OUT_DIR, filename)
    dataframe.to_csv(fullpath, sep='\t', quoting=csv.QUOTE_NONE, index=False, header=False)

df_articles = df_articles[df_articles.category == 'mercado']
artigos_unused_columns = ['text', 'category', 'subcategory', 'link']
df_articles.drop(artigos_unused_columns, inplace=True, axis=1)
df_articles['date'] = df_articles.apply(lambda row: np.int64(row['date'].replace('-', '')), axis=1)
df_articles['date'] = pd.to_datetime(df_articles['date'].astype(str), format='%Y%m%d')

df_bovespa.columns = map(str.lower, df_bovespa.columns)
df_bovespa = df_bovespa[df_bovespa.codneg.str.strip().isin(df_companies.code)]
df_bovespa['date'] = pd.to_datetime(df_bovespa['date'].astype(str), format='%Y%m%d')
df_bovespa = df_bovespa[df_bovespa.date >= df_articles.date.min()]
bovespa_unused_columns = ['open', 'company', 'typereg', 'bdicode', 'markettype', 'spec', 'prazot', 'currency', 'max', 'min', 'med', 'preofc', 'preofv', 'totneg', 'quatot']
df_bovespa.drop(bovespa_unused_columns, inplace=True, axis=1)
df_bovespa = df_bovespa.sort_values('date')

df_articles['date'] = df_articles.apply(lambda row: getEffectDate(row.date), axis=1)
df_articles = df_articles.sort_values('date')

for index, row in df_companies.iterrows():
    print('Generating for ' + row['code'])
    df_bovespa_company = df_bovespa[df_bovespa.codneg.str.strip() == row['code']]
    df_bovespa_company = df_bovespa_company.assign(close0=df_bovespa_company['close'].transform(lambda group: group.shift(1)))
    df_bovespa_company = df_bovespa_company.assign(close1d=df_bovespa_company['close'].transform(lambda group: group.shift(-1)))
    df_bovespa_company = df_bovespa_company.assign(close2d=df_bovespa_company['close'].transform(lambda group: group.shift(-2)))
    df_bovespa_company = df_bovespa_company.assign(close3d=df_bovespa_company['close'].transform(lambda group: group.shift(-3)))
    df_bovespa_company = df_bovespa_company.assign(close4d=df_bovespa_company['close'].transform(lambda group: group.shift(-4)))
    df_bovespa_company = df_bovespa_company.assign(close5d=df_bovespa_company['close'].transform(lambda group: group.shift(-5)))
    df_bovespa_company = df_bovespa_company[~np.isnan(df_bovespa_company.close0)]
    df_bovespa_company.drop('close', inplace=True, axis=1)
    df_company = pd.merge(df_articles, df_bovespa_company, on='date', how='inner')

    if df_company.size > 0:
        print(' 1 day window:')
        df_company_1d = df_company.drop(['date', 'codneg', 'close2d', 'close3d', 'close4d', 'close5d'], axis=1)
        df_company_1d = df_company_1d[~np.isnan(df_company_1d.close1d)]
        df_company_1d['label'] = df_company_1d.apply(lambda row: getAppreciation(row['close0'], row['close1d']), axis=1)
        df_company_1d = df_company_1d.drop(['close0', 'close1d'], axis=1)
        print('  number of positive registers: ' + str(df_company_1d[df_company_1d.label == 1].size))
        print('  number of neutral registers: ' + str(df_company_1d[df_company_1d.label == 0].size))
        print('  number of negative registers: ' + str(df_company_1d[df_company_1d.label == -1].size))
        
        print(' 2 days window:')
        df_company_2d = df_company.drop(['date', 'codneg', 'close1d', 'close3d', 'close4d', 'close5d'], axis=1)
        df_company_2d = df_company_2d[~np.isnan(df_company_2d.close2d)]
        df_company_2d['label'] = df_company_2d.apply(lambda row: getAppreciation(row['close0'], row['close2d']), axis=1)
        df_company_2d = df_company_2d.drop(['close0', 'close2d'], axis=1)
        print('  number of positive registers: ' + str(df_company_2d[df_company_2d.label == 1].size))
        print('  number of neutral registers: ' + str(df_company_2d[df_company_2d.label == 0].size))
        print('  number of negative registers: ' + str(df_company_2d[df_company_2d.label == -1].size))
        
        print(' 3 days window:')
        df_company_3d = df_company.drop(['date', 'codneg', 'close1d', 'close2d', 'close4d', 'close5d'], axis=1)
        df_company_3d = df_company_3d[~np.isnan(df_company_3d.close3d)]
        df_company_3d['label'] = df_company_3d.apply(lambda row: getAppreciation(row['close0'], row['close3d']), axis=1)
        df_company_3d = df_company_3d.drop(['close0', 'close3d'], axis=1)
        print('  number of positive registers: ' + str(df_company_3d[df_company_3d.label == 1].size))
        print('  number of neutral registers: ' + str(df_company_3d[df_company_3d.label == 0].size))
        print('  number of negative registers: ' + str(df_company_3d[df_company_3d.label == -1].size))
        
        print(' 4 days window:')
        df_company_4d = df_company.drop(['date', 'codneg', 'close1d', 'close2d', 'close3d', 'close5d'], axis=1)
        df_company_4d = df_company_4d[~np.isnan(df_company_4d.close4d)]
        df_company_4d['label'] = df_company_4d.apply(lambda row: getAppreciation(row['close0'], row['close4d']), axis=1)
        df_company_4d = df_company_4d.drop(['close0', 'close4d'], axis=1)
        print('  number of positive registers: ' + str(df_company_4d[df_company_4d.label == 1].size))
        print('  number of neutral registers: ' + str(df_company_4d[df_company_4d.label == 0].size))
        print('  number of negative registers: ' + str(df_company_4d[df_company_4d.label == -1].size))
        
        print(' 5 days window:')
        df_company_5d = df_company.drop(['date', 'codneg', 'close1d', 'close2d', 'close3d', 'close4d'], axis=1)
        df_company_5d = df_company_5d[~np.isnan(df_company_5d.close5d)]
        df_company_5d['label'] = df_company_5d.apply(lambda row: getAppreciation(row['close0'], row['close5d']), axis=1)
        df_company_5d = df_company_5d.drop(['close0', 'close5d'], axis=1)
        print('  number of positive registers: ' + str(df_company_5d[df_company_5d.label == 1].size))
        print('  number of neutral registers: ' + str(df_company_5d[df_company_5d.label == 0].size))
        print('  number of negative registers: ' + str(df_company_5d[df_company_5d.label == -1].size))
        
        exportToTSV(df_company_1d, row['code'] + '_1d.tsv')
        exportToTSV(df_company_2d, row['code'] + '_2d.tsv')
        exportToTSV(df_company_3d, row['code'] + '_3d.tsv')
        exportToTSV(df_company_4d, row['code'] + '_4d.tsv')
        exportToTSV(df_company_5d, row['code'] + '_5d.tsv')