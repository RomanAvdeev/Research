import pandas as pd
import numpy as np
import pymorphy2
import openpyxl
import shutil
import random
import io, codecs, re, os, sys, json, string, csv, datetime

from collections import Counter
from collections import defaultdict

# from pymystem3 import Mystem
from string import punctuation
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

from scipy.spatial.distance import cdist
from statistics import mean, mode

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from tqdm import tqdm


def marking_toloka_poles(toloka):
    toloka.insert(4, 'pole_mark', np.nan)
    for i in range(toloka.shape[0]):
        if toloka.at[i, 'OUTPUT:pole'] == 'other_topic':
            toloka.at[i, 'pole_mark'] = -1
        elif toloka.at[i, 'OUTPUT:pole'] == 'without_polarization':
            toloka.at[i, 'pole_mark'] = 0
        elif 'pole_' in str(toloka.at[i, 'OUTPUT:pole']):
            mark = int(toloka.at[i, 'OUTPUT:pole'][5])
            if mark == 0:
                mark = 9
            toloka.at[i, 'pole_mark'] = mark


# рассмотрим метки каждого документа: для каждого документа выпишем три метки (тк три асессора) для последующего анализа. Разметку каждого из экспертов считает верной
def check_docs_labels(toloka):
    '''def most_common(List):
        return(mode(List))'''

    data_doc_labels = pd.DataFrame(columns=['doc_id', 'label_1', 'label_2', 'label_3'])
    data_doc_labels['doc_id'] = list(
        (toloka['OUTPUT:doc_id'].value_counts().index).astype(int))  # лист всех документов из толоки - 452 шт

    data_doc_labels = data_doc_labels[['doc_id', 'label_1', 'label_2', 'label_3']]

    # добавим столбец, куда будем записывать id разметчиков (пусть будет + контроль, что взяли метки у РАЗНЫХ асессоров)
    # ВАЖНО: workers - id экспертов из толоки, corpus_id - номер корпуса из df_sovpad (т.е. из исходного датасета)
    data_doc_labels.insert(4, 'workers', 'a')
    data_doc_labels.insert(1, 'corpus_id', 'a')

    for i in range(toloka.shape[0]):
        curr_doc_id = toloka.at[i, 'OUTPUT:doc_id']
        doc_index_in_df = data_doc_labels[data_doc_labels['doc_id'] == curr_doc_id].index[
            0]  # находим строку в data_doc_labels с doc_id равным curr_doc_id

        if np.isnan(data_doc_labels.at[doc_index_in_df, 'label_1']):
            data_doc_labels.at[doc_index_in_df, 'label_1'] = toloka.at[i, 'pole_mark']
            worker_list = [toloka.at[i, 'ASSIGNMENT:worker_id']]
            data_doc_labels.at[doc_index_in_df, 'workers'] = worker_list
        elif np.isnan(data_doc_labels.at[doc_index_in_df, 'label_2']):
            data_doc_labels.at[doc_index_in_df, 'label_2'] = toloka.at[i, 'pole_mark']
            data_doc_labels.at[doc_index_in_df, 'workers'].append(toloka.at[i, 'ASSIGNMENT:worker_id'])
        else:
            np.isnan(data_doc_labels.at[doc_index_in_df, 'label_3'])
            data_doc_labels.at[doc_index_in_df, 'label_3'] = toloka.at[i, 'pole_mark']
            data_doc_labels.at[doc_index_in_df, 'workers'].append(toloka.at[i, 'ASSIGNMENT:worker_id'])

    #    data_doc_labels.insert(4, 'final_label', np.nan)

    #    for i in range(data_doc_labels.shape[0]):
    #        if (data_doc_labels.at[i, 'label_1'] == data_doc_labels.at[i, 'label_2']) or (data_doc_labels.at[i, 'label_1'] == data_doc_labels.at[i, 'label_3']) or (data_doc_labels.at[i, 'label_2'] == data_doc_labels.at[i, 'label_3']):
    #            data_doc_labels.at[i, 'final_label'] = int(most_common(list(data_doc_labels.iloc[i].values[:3])))

    return data_doc_labels


# записать номера корпусов из df_sovpad в data_doc_labels

def mark_corpus_id(data_doc_labels, df_sovpad):
    for i in range(data_doc_labels.shape[0]):
        curr_doc_id = data_doc_labels.at[i, 'doc_id']
        line_index_where_doc_id = df_sovpad[df_sovpad['doc_id'] == curr_doc_id].index[0]
        data_doc_labels.at[i, 'corpus_id'] = df_sovpad.at[line_index_where_doc_id, 'text_group_id']

    return data_doc_labels


# преобразовали столбец 'sentiment' из столбца строк в столбец словарей

def transform_to_dict(data):
    import ast
    data['sentiment'] = data['sentiment'].apply(lambda row: ast.literal_eval(row))


# выписать все слова из 'sentiment' для каждого документа в список

def extract_sent_words(data):
    data.insert(4, 'sent_words', '')
    all_sent_words = []
    for i in range(data.shape[0]):
        doc_sent_words = []
        for key in data.at[i, 'sentiment']['pos'].keys():
            doc_sent_words.append(key)
        for key in data.at[i, 'sentiment']['neg'].keys():
            doc_sent_words.append(key)
        data.at[i, 'sent_words'] = doc_sent_words
        all_sent_words += doc_sent_words
    return all_sent_words


#  выписать все слова из 'sentiment' для каждого документа в словарь БЕЗ разделения на neg и pos

def extract_sent_doc_to_dict(data):
    data.insert(4, 'sent_words_dict', '')
    for i in range(data.shape[0]):
        doc_sent = {}
        for key, value in data.at[i, 'sentiment']['neg'].items():
            if key in doc_sent.keys():
                doc_sent[key] += value
            else:
                doc_sent[key] = value
        for key, value in data.at[i, 'sentiment']['pos'].items():
            if key in doc_sent.keys():
                doc_sent[key] += value
            else:
                doc_sent[key] = value
        data.at[i, 'sent_words_dict'] = doc_sent


#  получаем большой общий словарь по всем документам (разделенный на neg и pos)

def create_all_sent_dict(data):
    all_sent_dict = {'neg': {}, 'pos': {}}
    for i in range(data.shape[0]):
        for key, value in data.at[i, 'sentiment']['neg'].items():
            if key in all_sent_dict['neg'].keys():
                all_sent_dict['neg'][key] += value
            else:
                all_sent_dict['neg'][key] = value
        for key, value in data.at[i, 'sentiment']['pos'].items():
            if key in all_sent_dict['pos'].keys():
                all_sent_dict['pos'][key] += value
            else:
                all_sent_dict['pos'][key] = value
    return all_sent_dict


#  получаем большой общий словарь по всем документам (БЕЗ разделения на neg и pos)

def create_joined_all_sent_dict(data):
    all_sent_dict = {}
    for i in range(data.shape[0]):
        for key, value in data.at[i, 'sentiment']['neg'].items():
            if key in all_sent_dict.keys():
                all_sent_dict[key] += value
            else:
                all_sent_dict[key] = value
        for key, value in data.at[i, 'sentiment']['pos'].items():
            if key in all_sent_dict.keys():
                all_sent_dict[key] += value
            else:
                all_sent_dict[key] = value
    return all_sent_dict


# tf-idf для каждого документа на основе сентиментов

def sent_tf_idf(data):
    # размерность вектора tf-idf - кол-во всех уникальных слов в нашем датасете
    # размерность матрицы = (кол-во документов, размерность вектора tf-df)
    # вектор tf-df для документа - соответствующая строка матрицы

    # transform_to_dict(data) # прменяем только один раз, когда в первый раз перевели столбец к типу словарь
    extract_sent_doc_to_dict(data)

    all_sent = create_joined_all_sent_dict(data)

    tf_idf_matrix = np.zeros((data.shape[0], len(all_sent.keys())))

    for i in range(data.shape[0]):

        tfidf_df = pd.DataFrame({'word': [], 'tfidf': []})
        tfidf_df['word'] = all_sent.keys()
        tfidf_df['tfidf'] = 0

        all_doc_sent_amount = sum(data.at[
                                      i, 'sent_words_dict'].values())  # кол-во уникальных слов в соответствующей колонке для данного документа

        for key, value in data.at[i, 'sent_words_dict'].items():

            tf = value / all_doc_sent_amount

            doc_key_count = 0
            for j in range(data.shape[0]):
                if key in data.at[j, 'sent_words_dict'].keys():
                    doc_key_count += 1

            idf = np.log(data.shape[0] / doc_key_count)

            tfidf = tf * idf
            word_index = tfidf_df[tfidf_df['word'] == key].index[0]
            tfidf_df.at[word_index, 'tfidf'] = tfidf

        tf_idf_matrix[i] = tfidf_df['tfidf'].to_numpy()

    return tf_idf_matrix


# выбор лучшего числа кластеров для kmeans

def inertia_comparing(inertia):
    '''y_axis = inertia
    k_coeffs = []

    for i in range(2, len(inertia)):
      M1 = np.array([[float(i-1), 1.], [float(i), 1.]])
      v1 = np.array([float(inertia[i-2]), float(inertia[i-1])])
      k, b = np.linalg.solve(M1, v1)
      k_coeffs.append(k)

    max_delta = 0
    n_clust = 0
    for i in range(1, len(k_coeffs)):
      if abs(k_coeffs[i-1] - k_coeffs[i]) > max_delta:
        max_delta = abs(k_coeffs[i-1] - k_coeffs[i])
        n_clust = i
    # АХТУНГ в оригинале снизу нет прибавления единицы'''

    elbow = KneeLocator(np.arange(1, 10), inertia, curve='convex', direction='decreasing').knee
    return elbow


# извлекаем соответствующий корпус из толоки

def extract_corpus_from_toloka(toloka, docs_ids):
    toloka_corpus_data = pd.DataFrame(columns=toloka.columns.values)
    for i in range(toloka.shape[0]):
        if toloka.at[i, 'INPUT:id'] in docs_ids:
            toloka_corpus_data = toloka_corpus_data.append(toloka.iloc[i], ignore_index=True)
    return toloka_corpus_data


# проверяем соответствие id корпуса из данных и из толоки(они в толоке и дате разные, а id документов одинаковые)

def equal_toloka_data_group_ids(toloka_gr_id, data_gr_id, toloka_data_corpuses_ids):
    flag = False
    for i in range(toloka_data_corpuses_ids.shape[0]):
        if toloka_data_corpuses_ids.at[i, 'toloka'] == toloka_gr_id and toloka_data_corpuses_ids.at[
            i, 'data'] == data_gr_id:
            flag = True
            break
    return flag


def corpus_to_tf_idf(corpus):
    text = corpus['lemm_sentences']
    # text = join_strs(corpus['text'])
    # text = preprocessing(text)
    tfidfconverter = TfidfVectorizer(stop_words=nltk_stopwords.words('russian'))
    tfidf = tfidfconverter.fit_transform(text)
    return tfidf


def train_kmeans(X, klasters):
    kmeans = KMeans(n_clusters=klasters, algorithm='full', init='k-means++', max_iter=300)
    y = kmeans.fit_predict(X.reshape(-1, 1))
    return kmeans


def modeling_for_curr_corpus(tfidf):
    inertias = []
    K = range(1, 10)

    for k in K:
        # Building and fitting the model
        if k > tfidf.shape[0]:
            k = tfidf.shape[0]
        kmeansModel = train_kmeans(tfidf, k)
        inertias.append(kmeansModel.inertia_)

    inertias_plot.append(inertias)
    n_cls.append(inertia_comparing(inertias))

    kmeansModel = train_kmeans(tfidf, inertia_comparing(inertias))
    return kmeansModel


# измерение качества, метрика M1 (точность и полнота кластеризации мнений)
# полагаем, что на вход дается X с мнениями-кластерами и Y - экспертна разметка

def M1(X, Y):
    X = list(X)
    Y = list(np.array(Y).astype(int))
    P_list = []
    R_list = []

    for i in range(len(X)):
        x_i = X[i]
        y_i = Y[i]
        sum_labels_i = 0
        sum_x = 0
        sum_y = 0
        for k in range(len(X)):
            if X[k] == x_i and Y[k] == y_i:
                sum_labels_i += 1
                sum_x += 1
                sum_y += 1
            elif X[k] == x_i:
                sum_x += 1
            elif Y[k] == y_i:
                sum_y += 1

        if sum_x == 0:
            P_i = 0
        else:
            P_i = sum_labels_i / sum_x

        if sum_y == 0:
            R_i = 0
        else:
            R_i = sum_labels_i / sum_y

        P_list.append(P_i)
        R_list.append(R_i)

    P = np.mean(P_list)
    R = np.mean(R_list)

    return 2 * P * R / (P + R)


# измерение качества, метрика M3 (точность и полнота определения релевантных сообщений)
# полагаем, что на вход дается list X с other_topic докуменатми (y_pred) и Y - список other_topic из толоки (y_true)

def M3_relevant(X, Y, show=False):
    X_clear, Y_clear = delete_nans(X, Y)
    X_binarized, Y_binarized = prepare_non_relevant_labling(X_clear, Y_clear)

    sum_labels = 0
    for i in range(len(X_binarized)):
        if Y_binarized[i] != -1 and X_binarized[i] != -1:
            sum_labels += 1

    sum_x = list(X_binarized).count(1)
    sum_y = list(Y_binarized).count(1)
    if sum_x == 0:
        P_1 = 0
    else:
        P_1 = sum_labels / sum_x

    if sum_y == 0:
        if show:
            print('В разметки нет нерелевантных, M3 =', np.nan)
        return np.nan
    else:
        R_1 = sum_labels / sum_y
        if show:
            print('precision=', P_1)
            print('recall=', R_1)
        if (P_1 + R_1) == 0:
            if show:
                print('M3=', np.nan)
            return np.nan
        else:
            if show:
                print('M3=', 2 * P_1 * R_1 / (P_1 + R_1))
            return 2 * P_1 * R_1 / (P_1 + R_1)


def M2_non_neutral(X, Y, show=False):
    X_clear, Y_clear = delete_nans(X, Y)
    X_binarized, Y_binarized = prepare_neutral_labling(X_clear, Y_clear)

    sum_labels = 0
    for i in range(len(X_binarized)):
        if Y_binarized[i] > 0 and X_binarized[i] > 0:
            sum_labels += 1

    sum_x = len(list(X_binarized)) - list(X_binarized).count(0) - list(X_binarized).count(-1)
    sum_y = len(list(Y_binarized)) - list(Y_binarized).count(0) - list(Y_binarized).count(-1)

    if sum_x == 0:
        P_0 = 0
    else:
        P_0 = sum_labels / sum_x

    if sum_y == 0:
        if show:
            print('В разметки нет нейтральных, M2 =', np.nan)
        R_0 = 0
    else:
        R_0 = sum_labels / sum_y
        if show:
            print('precision=', P_0)
            print('recall=', R_0)
    if (P_0 + R_0) == 0:
        if show:
            print('M2=', 0)
        return 0
    else:
        if show:
            print('M2=', 2 * P_0 * R_0 / (P_0 + R_0))
        return 2 * P_0 * R_0 / (P_0 + R_0)


# измерение качества, метрика M4 (точность определения числа мнений)

# X - ответ алогритма, Y - экспертная разметка

def M4(X, Y):
    K_x = float(len(set(list(X))))
    K_y = float(len(set(Y)))
    return min([K_x, K_y]) / max([K_x, K_y])
