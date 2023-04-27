import pdb

import numpy as np
import pandas as pd
import json
import argparse
import os
import re


def detect_bead(signal, degree: float = 0.3, verbose=False):
    bead_index = []
    max_value = max(signal)
    start = None
    end = None
    for i in range(1, len(signal)):
        if signal[i] - signal[i - 1] >= max_value * degree:
            if start is None:
                start, end = i - 1, i - 1
            else:
                print(f'error index: {i}')
        elif signal[i - 1] - signal[i] >= max_value * degree:
            if end is not None:
                bead_index.append(np.array([start, i]))
                if verbose:
                    print(f"Value at index start: {start}, end: {i}, length: {i - start}")
                start, end = None, None
            else:
                print(f'error index: {i}')
    print(f'Detected bead num: {len(bead_index)}')
    return np.array(bead_index)


def height_mean(sequence, alpha: float = 0.9):
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.mean(valid_values)


def height_min(sequence, alpha: float = 0.9):
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.min(valid_values)


def height_max(sequence, alpha: float = 0.9):
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.max(valid_values)


def height_std(sequence, alpha: float = 0.9):
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.std(valid_values)


def fwhm(sequence):
    return len(sequence)


def area(sequence):
    return np.sum(sequence)


def peak(sequence):
    return np.max(sequence)


def peak_time(sequence):
    peak_value = peak(sequence)
    return np.where(sequence == peak_value)[0][0]


def diff_peak(sequence):
    peak_value = peak(sequence)
    return peak_value - sequence[0]


def reach_time(sequence, alpha: float = 0.5):
    peak_value = peak(sequence)
    return np.where(sequence >= (peak_value * alpha))[0][0]


def over_area(sequence, alpha: float = 1):
    return np.sum(sequence[np.where(sequence >= alpha)])


def extra_time(sequence, bead_len, alpha: float = 0.03):
    return np.where(sequence >= alpha)[0][-1] - bead_len if len(np.where(sequence >= alpha)[0]) > 0 else 0


def extra_area(sequence, bead_len, alpha: float = 0.03):
    return np.sum(sequence[bead_len:np.where(sequence >= alpha)[0][-1]]) if len(
        np.where(sequence >= alpha)[0]) > 0 else 0


def sum_extra_area(sequence, bead_len, alpha: float = 0.5):
    over_time = reach_time(sequence, alpha)
    return np.sum(sequence[over_time:bead_len])


def lo_feature_extract(lo_sequence):
    lo_height_mean = height_mean(lo_sequence)
    lo_height_min = height_min(lo_sequence)
    lo_height_peak = height_max(lo_sequence)
    lo_height_std = height_std(lo_sequence)
    lo_fwhm = fwhm(lo_sequence)
    lo_area = area(lo_sequence)

    # return np.array([
    #     lo_height_mean, lo_height_min, lo_height_peak, lo_height_std,
    #     lo_fwhm, lo_area
    # ])
    return {
        'lo_height_mean': lo_height_mean,
        'lo_height_min': lo_height_min,
        'lo_height_peak': lo_height_peak,
        'lo_height_std': lo_height_std,
        'lo_fwhm': lo_fwhm,
        'lo_area': lo_area
    }


def br_feature_extract(br_sequence):
    br_peak = peak(br_sequence)
    br_peak_time = peak_time(br_sequence)
    br_diff_peak = diff_peak(br_sequence)
    # return np.array([br_peak, br_peak_time, br_diff_peak])
    return {
        'br_peak': br_peak,
        'br_peak_time': br_peak_time,
        'br_diff_peak': br_diff_peak
    }


def nir_feature_extractor(nir_sequence, bead_len):
    nir_peak = peak(nir_sequence)
    nir_reach_time = reach_time(nir_sequence)
    nir_over_area = over_area(nir_sequence)
    nir_extra_time = extra_time(nir_sequence, bead_len)
    nir_extra_area = extra_area(nir_sequence, bead_len)
    nir_sum_area = sum_extra_area(nir_sequence, bead_len)
    # return np.array([
    #     nir_peak, nir_reach_time, nir_over_area, nir_extra_time, nir_extra_area,
    #     nir_sum_area
    # ])
    return {
        'nir_peak': nir_peak,
        'nir_reach_time': nir_reach_time,
        'nir_over_area': nir_over_area,
        'nir_extra_time': nir_extra_time,
        'nir_extra_area': nir_extra_area,
        'nir_sum_area': nir_sum_area
    }


def vis_feature_extractor(vis_sequence, bead_len):
    vis_peak = peak(vis_sequence)
    vis_reach_time = reach_time(vis_sequence)
    vis_over_area = over_area(vis_sequence, alpha=0.7)
    vis_extra_time = extra_time(vis_sequence, bead_len)
    vis_extra_area = extra_area(vis_sequence, bead_len)
    # return np.array([
    #     vis_peak, vis_reach_time, vis_over_area, vis_extra_time, vis_extra_area
    # ])

    return {
        'vis_peak': vis_peak,
        'vis_reach_time': vis_reach_time,
        'vis_over_area': vis_over_area,
        'vis_extra_time': vis_extra_time,
        'vis_extra_area': vis_extra_area
    }


class ExtractFeatures:
    def __init__(self, filepath):
        self.file_name = None
        self.result = None
        self.bead_index = None
        self.__data = pd.read_csv(filepath, header=None).dropna(axis=1)
        self.__data.columns = ['LO', 'BR', 'NIR', 'VIS']
        self.file_name = re.search(r'(\w+)(_Total.csv)', filepath).group(1)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, filepath):
        self.__data = pd.read_csv(filepath, header=None).dropna(axis=1)
        self.__data.columns = ['LO', 'BR', 'NIR', 'VIS']
        self.file_name = re.search(r'(\w+)(_Total.csv)', filepath).group(1)

    def extract(self, verbose=False):
        self.result = dict()
        self.bead_index = detect_bead(np.array(self.__data['LO']), verbose=verbose)

        for i, [start, end] in enumerate(self.bead_index):
            next_start = end if i == len(self.bead_index) - 1 else self.bead_index[i + 1][0]
            features = self.extract_one_bead(start, end, next_start)
            self.result[f'Bead_{i + 1:02d}'] = features

        print(f'Extracted {len(self.bead_index)} beads from {self.file_name}')

        return self.result

    def extract_one_bead(self, start, end, next_start=None):
        next_start = end if next_start is None else next_start

        lo_sequence = np.array(self.__data['LO'][start:end])
        br_sequence = np.array(self.__data['BR'][start:end])
        nir_sequence = np.array(self.__data['NIR'][start:next_start])
        vis_sequence = np.array(self.__data['VIS'][start:next_start])

        # lo_features = lo_feature_extract(lo_sequence)
        # br_features = br_feature_extract(br_sequence)
        # nir_features = nir_feature_extractor(nir_sequence, len(lo_sequence))
        # vis_features = vis_feature_extractor(vis_sequence, len(lo_sequence))
        #
        # return np.concatenate([lo_features, br_features, nir_features, vis_features])

        extracted_features = {
            'lo_features': lo_feature_extract(lo_sequence),
            'br_features': br_feature_extract(br_sequence),
            'nir_features': nir_feature_extractor(nir_sequence, len(lo_sequence)),
            'vis_features': vis_feature_extractor(vis_sequence, len(lo_sequence))
        }

        return extracted_features

    def save_json(self, output_dir, encoder=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        outpath = os.path.join(output_dir, f'{self.file_name}_rule_based.json')
        with open(outpath, 'w') as f:
            json.dump(self.result, f, indent=4, cls=encoder)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    extractor = ExtractFeatures(args.filepath)
    extractor.extract(args.verbose)
    extractor.save_json(args.output_dir, encoder=NpEncoder)
    pdb.set_trace()
