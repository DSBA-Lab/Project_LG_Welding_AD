import numpy as np
import pandas as pd
from pathlib import Path
from extract_rule_based_features import ExtractFeatures, detect_bead
from matplotlib import pyplot as plt
from collections import Counter
import os
import torch
#multi-GPU
import torch.nn as nn
from tqdm import tqdm

import random
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score

from datetime import timedelta
import dateutil

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def slice_bead_data(path, set_bead_100=False, set_end_to_end=False):
    "path: data 저장 경로 (str)"
    extractor = ExtractFeatures(path)
    data = extractor.data
    if 'Abnormal' in path:
        data['label'] = 1
        data['dataset_idx'] = path[-12:-10]
    else:
        data['label'] = 0
        data['dataset_idx'] = str(path).split('/')[-1].split('.')[0]
    bead_array = detect_bead(data['LO'])
    data['bead_num'] = '0'
    
    if not set_end_to_end:
        if not set_bead_100:
            total_data = data.iloc[bead_array[0][0]:bead_array[0][1]+1]
        else:
            total_data = data.iloc[bead_array[0][0]:bead_array[0][0]+100]
        
        if 'Abnormal' in path:
            total_data['identifier'] = 'Abnormal_'+ path[-12:-10] + '_' + str(0)
        
        else:
            total_data['identifier'] = 'Normal_'+ str(path).split('/')[-1].split('.')[0] + '_' + str(0)

        if not set_bead_100:
            for i in range(1, len(bead_array)):
                bead_data = data.iloc[bead_array[i][0]:bead_array[i][1]]
                bead_data['bead_num'] = str(i)
                if 'Abnormal' in path:
                    bead_data['identifier'] = 'Abnormal_'+ path[-12:-10] + '_' + str(i)
                
                else:
                    bead_data['identifier'] = 'Normal_'+ str(path).split('/')[-1].split('.')[0] + '_' + str(i)
                    
                total_data = pd.concat([total_data, bead_data])

        else:
            for i in range(1, len(bead_array)):
                bead_data = data.iloc[bead_array[i][0]:bead_array[i][0]+100]
                bead_data['bead_num'] = str(i)
                if 'Abnormal' in path:
                    bead_data['identifier'] = 'Abnormal_'+ path[-12:-10] + '_' + str(i)
                
                else:
                    bead_data['identifier'] = 'Normal_'+ str(path).split('/')[-1].split('.')[0] + '_' + str(i)
                    
                total_data = pd.concat([total_data, bead_data])
                    
        return total_data

    else:
        start = bead_array[0,0]
        end = bead_array[-1,1]
        total_data = data[start:end]
        bead_num=np.array([])
        identifier=np.array([])
        total_rest_length=[]
        for i in range(len(bead_array)):
            bead_length=bead_array[i][1]-bead_array[i][0]
            if not i==len(bead_array)-1:
                rest_length=bead_array[i+1][0]-bead_array[i][1]
            bead_num=np.append(bead_num, np.ones(bead_length)*i) #bead에만 해당, 휴지 기간에는 0 을 할당하자
            if not i==len(bead_array)-1:
                bead_num=np.append(bead_num, -np.ones(rest_length))
            if 'Abnormal' in path:
                identifier=np.append(identifier, ['Abnormal_'+ path[-12:-10] + '_' + str(i)]*int(bead_length))
                if not i==len(bead_array)-1:
                    identifier=np.append(identifier, ['Abnormal_'+ path[-12:-10] + '_' + str(i) + '_rest']*int(rest_length))
            else:
                identifier=np.append(identifier, ['Normal_'+ str(path).split('/')[-1].split('.')[0] + '_' + str(i)]*int(bead_length))
                if not i==len(bead_array)-1:
                    identifier=np.append(identifier, ['Normal_'+ str(path).split('/')[-1].split('.')[0] + '_' + str(i) + '_rest']*int(rest_length))
            total_rest_length.append(rest_length)

        avg_rest_length=int(np.mean(total_rest_length))

        # 다른 데이터와의 concat 과정에서도 휴지기간이 존재하도록 하기 위해 뒤에 평균 휴지 기간만큼 데이터 추가
        #total_data=total_data +data[end:end+avg_rest_length]
        total_data=pd.concat([total_data, data[end:end+avg_rest_length]])

        bead_num=np.append(bead_num, -np.ones(avg_rest_length))
        if 'Abnormal' in path:
            identifier=np.append(identifier, ['Abnormal_'+ path[-12:-10] + '_' + str(len(bead_array)-1) + '_rest']*int(avg_rest_length))
        else:
            identifier=np.append(identifier, ['Normal_'+ str(path).split('/')[-1].split('.')[0] + '_' + str(len(bead_array)-1) + '_rest']*int(avg_rest_length))
        
        #bead_num: int
        bead_num=bead_num.astype(int)
        total_data['bead_num']=bead_num
        total_data['identifier']=identifier
        
        return total_data
    

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def PA_percentile(score, label,
                  threshold=None,
                  pred=None,
                  K=100,
                  calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    anomalies = []

    for i in range(len(actual)):
        if actual[i]:
            if not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                anomalies.append([i, i])
            else:
                anomalies[-1][-1] = i
        else:
            anomaly_state = False

    for i, [start, end] in enumerate(anomalies):
        collect = Counter(predict[start:end + 1])[1]
        anomaly_count += collect
        collect_ratio = collect / (end - start + 1)

        if collect_ratio * 100 >= K and collect > 0:
            predict[start:end + 1] = True
            latency += (end - start + 1) - collect

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, K=0, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        #predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict, latency = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        t.append(latency)
        return t
    else:
        roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        # predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        return t


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, K=0, verbose=True) -> object:
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, K=K, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    return m, m_t

def bp_search(score, label, start, end=None, step_num=1, display_freq=1, K=0, verbose=True) -> object:
    """
    Find the best-precision score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-precision
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, K=K, calc_latency=True)
        if target[1] > m[1]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    return m, m_t


def get_window_bead_num(
                data : pd.DataFrame,
                window_size: int,
                slide_size: int):
    data = data[['bead_num']]
    data['bead_num'] = data['bead_num'].astype(int)
    data = data.reset_index(drop=True)
    data = data.dropna()
    ts = np.array(data.index)

    valid_idxs = []

    for L in range(0, len(ts) - window_size + 1, slide_size):
        R = L + window_size - 1
        try:
            if dateutil.parser.parse(ts[R]) - dateutil.parser.parse(
                    ts[L]
            ) == timedelta(seconds=window_size - 1):
                valid_idxs.append(L)
        except:
            if ts[R] - ts[L] == window_size - 1:
                valid_idxs.append(L)

    window_bead_num_list = []
    for i in valid_idxs:
        window_bead_num_list.append(data['bead_num'][i:i+window_size].values)
    return window_bead_num_list


def check_graph(xs, att=None, piece=1, threshold=None, custom_ylim=None):
    """
    anomaly score and anomaly label visualization

    Parameters
    ----------
    xs : np.ndarray
        anomaly scores
    att : np.ndarray
        anomaly labels
    piece : int
        number of figures to separate
    threshold : float(default=None)
        anomaly threshold

    Return
    ------
    fig : plt.figure
    """
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(12, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = np.arange(L, R)
        if piece == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.plot(xticks, xs[L:R], color='#0C090A')
        ymin, ymax = ax.get_ylim()
        ymin = 0
        ax.set_ylim(ymin, ymax)
        if len(xs[L:R]) > 0 and att is not None:
            ax.vlines(xticks[np.where(att[L:R] == 1)], ymin=ymin, ymax=ymax, color='#FED8B1',
                          alpha=0.6, label='true anomaly')
        ax.plot(xticks, xs[L:R], color='#0C090A', label='anomaly score')
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.8, label=f'threshold:{threshold:.4f}')
        ax.legend()
        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)
    return fig