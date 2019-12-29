import re
import numpy as np
import pandas as pd
import csv
import os
from datetime import timedelta
from unicodedata import normalize
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

OUT_DIR = './dataset_out'
MAX_DAYS = 5
TRAIN_PERCENTAGE_SIZE = 80/100
TEST_PERCENTAGE_SIZE = 20/100
REGEXP_REMOVE_SPECIAL = re.compile('[^a-zA-Z0-9 ]+')
ONLY_ONE_CODE = False
ONLY_ONE_CODE_NAME = 'VALE3'
STOPWORDS = stopwords.words('portuguese')
STEMMER = SnowballStemmer('portuguese')
ARTIGOS_TEXT_COLUMN = 'text'
ARTIGOS_UNUSED_COLUMNS = ['title', 'category', 'subcategory', 'link']
BOVESPA_UNUSED_COLUMNS = ['open', 'company', 'typereg', 'bdicode', 'markettype', 'spec', 'prazot', 'currency', 'max', 'min', 'med', 'preofc', 'preofv', 'totneg', 'quatot']

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

def getCleanText(text):
  finalTextArray = []
  lowerText = text.lower()
  for word in lowerText.split():
    if word not in STOPWORDS:
      finalTextArray.append(STEMMER.stem(word))
  finalText = ' '.join(finalTextArray)
  finalText = normalize('NFKD', finalText).encode('ASCII', 'ignore').decode('ASCII')
  finalText = REGEXP_REMOVE_SPECIAL.sub('', finalText)
  finalText = re.sub(' +', ' ', finalText)
  return finalText

df_articles = df_articles[df_articles.category == 'mercado']
df_articles.drop(ARTIGOS_UNUSED_COLUMNS, inplace=True, axis=1)
df_articles['date'] = df_articles.apply(lambda row: np.int64(row['date'].replace('-', '')), axis=1)
df_articles['date'] = pd.to_datetime(df_articles['date'].astype(str), format='%Y%m%d')

df_bovespa.columns = map(str.lower, df_bovespa.columns)
df_bovespa = df_bovespa[df_bovespa.codneg.str.strip().isin(df_companies.code)]
df_bovespa['date'] = pd.to_datetime(df_bovespa['date'].astype(str), format='%Y%m%d')
df_bovespa = df_bovespa[df_bovespa.date >= df_articles.date.min()]
df_bovespa.drop(BOVESPA_UNUSED_COLUMNS, inplace=True, axis=1)
df_bovespa = df_bovespa.sort_values('date')

df_articles['date'] = df_articles.apply(lambda row: getEffectDate(row.date), axis=1)
df_articles[ARTIGOS_TEXT_COLUMN] = df_articles.apply(lambda row: getCleanText(row[ARTIGOS_TEXT_COLUMN]), axis=1)
df_articles = df_articles.sort_values('date')

df_analysis = pd.DataFrame(columns=['dataset','1s','0s', '-1s'])

for index, row in df_companies.iterrows():
  if not ONLY_ONE_CODE or (ONLY_ONE_CODE and row['code'] == ONLY_ONE_CODE_NAME):
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
        df_company = df_company[['label', ARTIGOS_TEXT_COLUMN]]
        
        df_company_positive = df_company[df_company.label == 1].sample(frac=1)
        df_company_neutral = df_company[df_company.label == 0].sample(frac=1)
        df_company_negative = df_company[df_company.label == -1].sample(frac=1)

        analysis = pd.Series({"dataset": row['code'] + '_' + str(interval) + 'd.tsv', "1s": len(df_company_positive), "0s": len(df_company_neutral), "-1s": len(df_company_negative)})
        df_analysis = df_analysis.append(analysis, ignore_index=True)
        
        trainPositiveSize = round(len(df_company_positive)*(TRAIN_PERCENTAGE_SIZE))
        testPositiveSize = round(len(df_company_positive)*(TEST_PERCENTAGE_SIZE))

        trainNeutralSize = round(len(df_company_neutral)*(TRAIN_PERCENTAGE_SIZE))
        testNeutralSize = round(len(df_company_neutral)*(TEST_PERCENTAGE_SIZE))

        trainNegativeSize = round(len(df_company_negative)*(TRAIN_PERCENTAGE_SIZE))
        testNegativeSize = round(len(df_company_negative)*(TEST_PERCENTAGE_SIZE))
        
        df_company_train = df_company_positive.head(trainPositiveSize)
        df_company_positive = df_company_positive.iloc[trainPositiveSize:]
        df_company_train = df_company_train.append(df_company_negative.head(trainNegativeSize))
        df_company_negative = df_company_negative.iloc[trainNegativeSize:]

        df_company_test = df_company_positive.head(testPositiveSize)
        df_company_positive = df_company_positive.iloc[testPositiveSize:]
        df_company_test = df_company_test.append(df_company_negative.head(testNegativeSize))
        df_company_negative = df_company_negative.iloc[testNegativeSize:]

        exportToTSV(df_company_train, row['code'] + '_2c_' + str(interval) + 'd_train.tsv')
        exportToTSV(df_company_test, row['code'] + '_2c_' + str(interval) + 'd_test.tsv')

        df_company_train = df_company_train.append(df_company_neutral.head(trainNeutralSize))
        df_company_neutral = df_company_neutral.iloc[trainNeutralSize:]
        df_company_test = df_company_test.append(df_company_neutral.head(testNeutralSize))
        df_company_neutral = df_company_neutral.iloc[testNeutralSize:]
        
        exportToTSV(df_company_train, row['code'] + '_3c_' + str(interval) + 'd_train.tsv')
        exportToTSV(df_company_test, row['code'] + '_3c_' + str(interval) + 'd_test.tsv')