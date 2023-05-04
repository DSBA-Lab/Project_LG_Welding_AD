import numpy as np
import pandas as pd
import json
import argparse
import os
import re
import pdb
from numpy import ndarray


def detect_bead(signal: np.ndarray, degree: float = 0.3, verbose: bool = False) -> np.ndarray:
    """
    bead detection
    :param signal: np.ndarray
        input signal (LO only)
    :param degree: float
        degree of bead detection
    :param verbose: bool
        verbose
    :return: np.ndarray
        bead index
    """
    bead_index = []
    max_value = max(signal)
    start = None
    end = None
    for i in range(1, len(signal)):
        # A sharp increase in the current value over the previous value is considered a start
        if signal[i] - signal[i - 1] >= max_value * degree:
            if start is None:
                start, end = i - 1, i - 1
            else:
                print(f'warning index: {i}')
        # A sharp decrease in the current value over the previous value is considered a end
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


def height_mean(sequence: np.ndarray, alpha: float = 0.9) -> float:
    """
    Re-averaging of values above 90% of the mean value within the valid interval.
    :param sequence: np.ndarray
        input sequence
    :param alpha: float
        alpha
    :return: float
        mean value
    """
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.mean(valid_values)


def height_min(sequence: np.ndarray, alpha: float = 0.9) -> float:
    """
    The lowest value that is at least 90% of the mean value within the valid interval.
    :param sequence: np.ndarray
        input sequence
    :param alpha: float
        alpha
    :return: float
        min value
    """
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.min(valid_values)


def height_max(sequence: np.ndarray, alpha: float = 0.9) -> float:
    """
    The highest value that is at least 90% of the mean value within the valid interval.
    :param sequence: np.ndarray
        input sequence
    :param alpha: float
        alpha
    :return: float
        max value
    """
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.max(valid_values)


def height_std(sequence: np.ndarray, alpha: float = 0.9) -> float:
    """
    The standard deviation of the values above 90% of the mean value within the valid interval.
    :param sequence: np.ndarray
        input sequence
    :param alpha: float
        alpha
    :return: float
        std value
    """
    mean_values = np.mean(sequence)
    valid_values = sequence[np.where(sequence >= (mean_values * alpha))]
    return np.std(valid_values)


def fwhm(sequence: np.ndarray) -> int:
    """
    Length of the valid interval
    :param sequence: np.ndarray
        input sequence
    :return: int
        fwhm value
    """
    return len(sequence)


def area(sequence: np.ndarray) -> float:
    """
    Cumulative sum within the valid interval.
    :param sequence: np.ndarray
        input sequence
    :return: float
        area value
    """
    return np.sum(sequence)


def peak(sequence: np.ndarray) -> float:
    """
    The highest value within the valid interval.
    :param sequence: np.ndarray
        input sequence
    :return: float
        peak value
    """
    return np.max(sequence)


def peak_time(sequence: np.ndarray) -> float:
    """
    Time to peak from the start of the valid interval
    :param sequence: np.ndarray
        input sequence
    :return: float
        peak time value
    """
    peak_value = peak(sequence)
    return np.where(sequence == peak_value)[0][0]


def diff_peak(sequence: np.ndarray) -> float:
    """
    Difference between peak and end-of-period values
    :param sequence: np.ndarray
        input sequence
    :return: float
        diff peak value
    """
    peak_value = peak(sequence)
    return peak_value - sequence[-1]


def reach_time(sequence: np.ndarray, alpha: float = 0.5) -> int:
    """
    Time from start of the valid interval to peak*alpha point
    :param sequence: np.ndarray
        input sequence
    :param alpha: float
        alpha
    :return: int
        reach time value
    """
    peak_value = peak(sequence)
    return np.where(sequence >= (peak_value * alpha))[0][0]


def over_area(sequence: np.ndarray, alpha: float = 1) -> float:
    """
    Cumulative sum over a base value
    :param sequence: np.ndarray
        input sequence
    :param alpha: float
        alpha
    :return: float
        over area value
    """
    return np.sum(sequence[np.where(sequence >= alpha)])


def extra_time(sequence: np.ndarray, bead_len: int, alpha: float = 0.03) -> int:
    """
    Time since the valid interval while above the threshold
    :param sequence: np.ndarray
        input sequence
    :param bead_len: int
        bead length
    :param alpha: float
        alpha
    :return: int
        extra time value
    """
    return np.where(sequence >= alpha)[0][-1] - bead_len if len(np.where(sequence >= alpha)[0]) > 0 else 0


def extra_area(sequence: np.ndarray, bead_len: int, alpha: float = 0.03) -> float:
    """
    Cumulative sum of time above threshold after the valid interval
    :param sequence: np.ndarray
        input sequence
    :param bead_len: int
        bead length
    :param alpha: float
        alpha
    :return: float
        extra area value
    """
    return np.sum(sequence[bead_len:np.where(sequence >= alpha)[0][-1]]) if len(np.where(sequence >= alpha)[0]) > 0 else 0


def sum_extra_area(sequence: np.ndarray, bead_len: int, alpha: float = 0.5) -> float:
    """
    Cumulative sum from Peak*alpha to the end of the valid interval
    :param sequence: np.ndarray
        input sequence
    :param bead_len: int
        bead length
    :param alpha: float
        alpha
    :return: float
        sum extra area value
    """
    over_time = reach_time(sequence, alpha)
    return np.sum(sequence[over_time:bead_len])


def lo_feature_extract(lo_sequence: np.ndarray) -> dict:
    """
    Rule-based feature extraction of the lo variable
    :param lo_sequence: np.ndarray
        lo sequence
    :return: dict
        lo feature dict
    """
    lo_height_mean = height_mean(lo_sequence)
    lo_height_min = height_min(lo_sequence)
    lo_height_peak = height_max(lo_sequence)
    lo_height_std = height_std(lo_sequence)
    lo_fwhm = fwhm(lo_sequence)
    lo_area = area(lo_sequence)

    return {
        'lo_height_mean': lo_height_mean,
        'lo_height_min': lo_height_min,
        'lo_height_peak': lo_height_peak,
        'lo_height_std': lo_height_std,
        'lo_fwhm': lo_fwhm,
        'lo_area': lo_area
    }


def br_feature_extract(br_sequence: np.ndarray) -> dict:
    """
    Rule-based feature extraction of the br variable
    :param br_sequence: np.ndarray
        br sequence
    :return: dict
        br feature dict
    """
    br_peak = peak(br_sequence)
    br_peak_time = peak_time(br_sequence)
    br_diff_peak = diff_peak(br_sequence)

    return {
        'br_peak': br_peak,
        'br_peak_time': br_peak_time,
        'br_diff_peak': br_diff_peak
    }


def nir_feature_extractor(nir_sequence: np.ndarray, bead_len: int = None) -> dict:
    """
    Rule-based feature extraction of the nir variable
    :param nir_sequence: np.ndarray
        nir sequence
    :param bead_len: int
        bead length
    :return: dict
        nir feature dict
    """
    bead_len = len(nir_sequence) if bead_len is None else bead_len
    nir_peak = peak(nir_sequence)
    nir_reach_time = reach_time(nir_sequence)
    nir_over_area = over_area(nir_sequence)
    nir_extra_time = extra_time(nir_sequence, bead_len)
    nir_extra_area = extra_area(nir_sequence, bead_len)
    nir_sum_area = sum_extra_area(nir_sequence, bead_len)

    return {
        'nir_peak': nir_peak,
        'nir_reach_time': nir_reach_time,
        'nir_over_area': nir_over_area,
        'nir_extra_time': nir_extra_time,
        'nir_extra_area': nir_extra_area,
        'nir_sum_area': nir_sum_area
    }


def vis_feature_extractor(vis_sequence: np.ndarray, bead_len: int = None) -> dict:
    """
    Rule-based feature extraction of the vis variable
    :param vis_sequence: np.ndarray
        vis sequence
    :param bead_len: int
        bead length
    :return: dict
        vis feature dict
    """
    bead_len = len(vis_sequence) if bead_len is None else bead_len
    vis_peak = peak(vis_sequence)
    vis_reach_time = reach_time(vis_sequence)
    vis_over_area = over_area(vis_sequence, alpha=0.7)
    vis_extra_time = extra_time(vis_sequence, bead_len)
    vis_extra_area = extra_area(vis_sequence, bead_len)

    return {
        'vis_peak': vis_peak,
        'vis_reach_time': vis_reach_time,
        'vis_over_area': vis_over_area,
        'vis_extra_time': vis_extra_time,
        'vis_extra_area': vis_extra_area
    }


class ExtractFeatures:
    result = None
    file_name = None
    __data = None
    bead_index = None

    def __init__(self, filepath: str):
        """
        Extract features from the csv file
        :param filepath: str
            csv file path
        """
        self.result = None
        self.bead_index = None
        self.__data = pd.read_csv(filepath, header=None).dropna(axis=1)
        self.__data.columns = ['LO', 'BR', 'NIR', 'VIS']
        self.file_name = re.search(r'(\w+)(_Total.csv)', filepath).group(1)

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the data
        :return: pd.DataFrame
            data
        """
        return self.__data

    @data.setter
    def data(self, filepath: str) -> None:
        """
        Set the data
        :param filepath: str
            csv file path
        :return: None
        """
        self.__data = pd.read_csv(filepath, header=None).dropna(axis=1)
        self.__data.columns = ['LO', 'BR', 'NIR', 'VIS']
        self.file_name = re.search(r'(\w+)(_Total.csv)', filepath).group(1)

    @classmethod
    def extract(cls, data: np.ndarray, verbose: bool = False) -> dict:
        """
        Extract features from the data
        :param data: np.ndarray
            data
        :param verbose: bool
            print the process or not
        :return: dict
            features dict
        """
        result = dict()
        bead_index = detect_bead(np.array(data['LO']), verbose=verbose)

        for i, [start, end] in enumerate(bead_index):
            next_start = end if i == len(bead_index) - 1 else bead_index[i + 1][0]
            features = cls.extract_one_bead(data, start, end, next_start)
            result[f'Bead_{i + 1:02d}'] = features

        print(f'Extracted {len(bead_index)} beads from data')

        return result

    @staticmethod
    def extract_one_bead(data, start: int, end: int, next_start=None) -> dict:
        """
        Extract features from one bead
        :param data: np.ndarray
            data
        :param start: int
            bead start index
        :param end: int
            bead end index
        :param next_start: int
            next bead start index
        :return: dict
            features dict
        """
        next_start = end if next_start is None else next_start

        lo_sequence = np.array(data['LO'][start:end])
        br_sequence = np.array(data['BR'][start:end])
        nir_sequence = np.array(data['NIR'][start:next_start])
        vis_sequence = np.array(data['VIS'][start:next_start])

        extracted_features = {
            'lo_features': lo_feature_extract(lo_sequence),
            'br_features': br_feature_extract(br_sequence),
            'nir_features': nir_feature_extractor(nir_sequence, len(lo_sequence)),
            'vis_features': vis_feature_extractor(vis_sequence, len(lo_sequence))
        }

        return extracted_features

    def save_json(self, output_dir: str, encoder: bool = None) -> None:
        """
        Save the result to json file
        :param output_dir: str
            output directory
        :param encoder: bool
            json encoder
        :return: None
        """
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
    extractor.extract(extractor.data, args.verbose)
    extractor.save_json(args.output_dir, encoder=NpEncoder)
