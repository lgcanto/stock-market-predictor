import numpy as np
import pandas as pd
import csv
import os
from datetime import timedelta

OUT_DIR = './dataset_out'
MAX_DAYS = 5
TRAIN_PERCENTAGE_SIZE = 70/100
EVAL_PERCENTAGE_SIZE = 10/100
TEST_PERCENTAGE_SIZE = 20/100

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

df_analysis = pd.DataFrame(columns=['dataset','1s','0s', '-1s'])

for index, row in df_companies.iterrows():
    print('Generating for ' + row['code'])
    df_full = df_bovespa[df_bovespa.codneg.str.strip() == row['code']]
    df_full = df_full.assign(close_before=df_full['close'].transform(lambda group: group.shift(1)))
    df_full = df_full[~np.isnan(df_full.close_before)]
    for d in range(MAX_DAYS):
        interval = d + 1
        df_interval = df_full.assign(close_after=df_full['close'].transform(lambda group: group.shift(-interval)))
        df_interval = df_interval[~np.isnan(df_interval.close_after)]
        df_interval.drop('close', inplace=True, axis=1)
        df_company = pd.merge(df_articles, df_interval, on='date', how='inner')
        if df_company.size > 0:
            df_company.drop(['date', 'codneg'], inplace=True, axis=1)
            df_company['label'] = df_company.apply(lambda row: getAppreciation(row['close_before'], row['close_after']), axis=1)
            df_company.drop(['close_before', 'close_after'], inplace=True, axis=1)
            
            df_company_positive = df_company[df_company.label == 1]
            df_company_neutral = df_company[df_company.label == 0]
            df_company_negative = df_company[df_company.label == -1]
            
            analysis = pd.Series({"dataset": row['code'] + '_' + str(interval) + 'd.tsv', "1s": len(df_company_positive), "0s": len(df_company_neutral), "-1s": len(df_company_negative)})
            df_analysis = df_analysis.append(analysis, ignore_index=True)
            
            trainPositiveSize = round(len(df_company_positive)*(TRAIN_PERCENTAGE_SIZE))
            evalPositiveSize = round(len(df_company_positive)*(EVAL_PERCENTAGE_SIZE))
            testPositiveSize = round(len(df_company_positive)*(TEST_PERCENTAGE_SIZE))

            trainNeutralSize = round(len(df_company_neutral)*(TRAIN_PERCENTAGE_SIZE))
            evalNeutralSize = round(len(df_company_neutral)*(EVAL_PERCENTAGE_SIZE))
            testNeutralSize = round(len(df_company_neutral)*(TEST_PERCENTAGE_SIZE))

            trainNegativeSize = round(len(df_company_negative)*(TRAIN_PERCENTAGE_SIZE))
            evalNegativeSize = round(len(df_company_negative)*(EVAL_PERCENTAGE_SIZE))
            testNegativeSize = round(len(df_company_negative)*(TEST_PERCENTAGE_SIZE))
            
            df_company_train = df_company_positive.head(trainPositiveSize)
            df_company_positive = df_company_positive.iloc[trainPositiveSize:]
            df_company_train = df_company_train.append(df_company_neutral.head(trainNeutralSize))
            df_company_neutral = df_company_neutral.iloc[trainNeutralSize:]
            df_company_train = df_company_train.append(df_company_negative.head(trainNegativeSize))
            df_company_negative = df_company_negative.iloc[trainNegativeSize:]

            df_company_eval = df_company_positive.head(evalPositiveSize)
            df_company_positive = df_company_positive.iloc[evalPositiveSize:]
            df_company_eval = df_company_eval.append(df_company_neutral.head(evalNeutralSize))
            df_company_neutral = df_company_neutral.iloc[evalNeutralSize:]
            df_company_eval = df_company_eval.append(df_company_negative.head(evalNegativeSize))
            df_company_negative = df_company_negative.iloc[evalNegativeSize:]
            
            df_company_test = df_company_positive.head(testPositiveSize)
            df_company_positive = df_company_positive.iloc[testPositiveSize:]
            df_company_test = df_company_test.append(df_company_neutral.head(testNeutralSize))
            df_company_neutral = df_company_neutral.iloc[testNeutralSize:]
            df_company_test = df_company_test.append(df_company_negative.head(testNegativeSize))
            df_company_negative = df_company_negative.iloc[testNegativeSize:]

            exportToTSV(df_company_train, row['code'] + '_' + str(interval) + 'd_train.tsv')
            exportToTSV(df_company_eval, row['code'] + '_' + str(interval) + 'd_eval.tsv')
            exportToTSV(df_company_test, row['code'] + '_' + str(interval) + 'd_test.tsv')