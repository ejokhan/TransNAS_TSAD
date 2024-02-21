import numpy as np

from src.spot import SPOT

from sklearn.metrics import *
import time
import numpy as np

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
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


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


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
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
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def calculate_mat(scores, window_size):
    """
    Calculate the Moving Average Threshold (MAT) for anomaly scores.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    return np.convolve(scores, np.ones(window_size) / window_size, mode='same')

def calculate_rolling_stats(scores, window_size):
    """
    Calculate rolling mean and standard deviation for anomaly scores.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    mean = np.convolve(scores, np.ones(window_size) / window_size, mode='same')
    squared_diffs = (scores - mean) ** 2
    rolling_std = np.sqrt(np.convolve(squared_diffs, np.ones(window_size) / window_size, mode='same'))
    return mean, rolling_std




from src.spot import SPOT  # Ensure SPOT is imported correctly from your project structure

def pot_eval(config, trial_timeout, init_score, score, label, q=1e-5, level=0.02, window_size=10):
    """
    Performs the Peak Over Threshold (POT) method on given anomaly scores to evaluate the detection performance,
    enhanced with Moving Average Threshold (MAT) and rolling statistics for nuanced detection.

    Args:
        config: Configuration dictionary containing dataset and model information.
        trial_timeout: Timeout limit in seconds for the POT method initialization.
        init_score: Initial anomaly scores of the training set used to initialize the threshold.
        score: Anomaly scores of the test set to evaluate.
        label: True labels of the test set.
        q: Quantile level for threshold selection.
        level: Significance level for the test.
        window_size: Window size for calculating MAT and rolling statistics.

    Returns:
        A dictionary containing various evaluation metrics (F1 score, precision, recall, etc.),
        and an array of predictions based on the POT method with augmentative strategies.
    """

    # Dataset-specific parameters for initializing the POT method
    lm_d = {
        'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
        'SWaT': [(0.993, 1), (0.993, 1)],
        'UCR': [(0.993, 1), (0.99935, 1)],
        'NAB': [(0.991, 1), (0.99, 1)],
        'SMAP': [(0.99, 1), (0.99, 1)],
        'MSL': [(0.97, 1), (0.999, 1.04)],
        'WADI': [(0.99, 1), (0.999, 1)],
        'MBA': [(0.87, 1), (0.93, 1.04)],
    }
    index = 1 if 'TransNAS_TSAD' in config.model else 0
    lm = lm_d[config.dataset][index]
    lms = lm[0]

    start_time = time.time()

    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except Exception as e:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"Initialization timed out after {trial_timeout} seconds. Last exception message: {e}")
                return {}, np.array([])
            lms *= 0.999
        else:
            break

    ret = s.run(dynamic=False)
    pot_th = np.mean(ret['thresholds']) * lm[1]

    # Calculate MAT
    mat_scores = np.convolve(score, np.ones(window_size) / window_size, mode='same')

    # Calculate rolling statistics
    mean, rolling_std = np.convolve(score, np.ones(window_size) / window_size, mode='same'), \
                         np.sqrt(np.convolve((score - np.convolve(score, np.ones(window_size) / window_size, mode='same')) ** 2, np.ones(window_size) / window_size, mode='same'))

    # Example logic for dynamic threshold adjustment
    dynamic_threshold = pot_th + np.mean(mat_scores - mean) / np.mean(rolling_std) * level

    # Adjust predictions based on the calculated dynamic threshold
    pred, p_latency = adjust_predicts(score, label, dynamic_threshold, calc_latency=True)

    # Calculate evaluation metrics
    p_t = calc_point2point(pred, label)

    # Return the evaluation results and predictions
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': dynamic_threshold,
    }, np.array(pred)


import numpy as np
from sklearn.metrics import ndcg_score


def hit_att(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		hit_score = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
			if l:
				size = round(p * len(l) / 100)
				a_p = set(a[:size])
				intersect = a_p.intersection(l)
				hit = len(intersect) / len(l)
				hit_score.append(hit)
		res[f'Hit@{p}%'] = np.mean(hit_score)
	return res

def ndcg(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		ndcg_scores = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			labs = list(np.where(l == 1)[0])
			if labs:
				k_p = round(p * len(labs) / 100)
				try:
					hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
				except Exception as e:
					return {}
				ndcg_scores.append(hit)
		res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
	return res