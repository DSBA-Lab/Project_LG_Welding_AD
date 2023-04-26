import sys
import numpy as np
import pandas as pd
import tsfresh.feature_extraction.feature_calculators
import pywt
from scipy import signal
from scipy import stats
from scipy.signal import hilbert, chirp
from scipy.signal import butter, lfilter, freqz
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import minimum_filter1d

## Feature Extraction
  

# Input Bead 3EA, Return Value 1EA
def PeakArea(list): # *불가, input type = 2-dim list : ex = [[1,2,3], [4,5,6], [7,8,9]]
    
    bead_matrix = np.array(list) 
    
    bead_MA_array         = bead_matrix[0] # 검사할 bead    
    tuning_MA_plus_array  = bead_matrix[1] # moving avg. 상한
    tuning_MA_minus_array = bead_matrix[2] # moving avg. 하한
    
    area_plus = np.sum(bead_MA_array[bead_MA_array > tuning_MA_plus_array] - tuning_MA_plus_array[bead_MA_array > tuning_MA_plus_array])
    area_minus = np.sum(tuning_MA_minus_array[bead_MA_array < tuning_MA_minus_array]- bead_MA_array[bead_MA_array < tuning_MA_minus_array])
    total_area = area_plus + area_minus
    
    return float(total_area)


# Input Bead 3EA, Return Value 1EA
def PeakCount(list): # *불가, input type = 2-dim list : ex = [[1,2,3], [4,5,6], [7,8,9]]
    
    bead_matrix = np.array(list) 
    
    bead_Ori_array         = bead_matrix[0] # 검사할 bead    
    tuning_Ori_plus_array  = bead_matrix[1] # Original bead 상한
    tuning_Ori_minus_array = bead_matrix[2] # Original bead 하한
    
    peakCount_plus = len(bead_Ori_array[bead_Ori_array > tuning_Ori_plus_array])
    peakCount_minus = len(bead_Ori_array[bead_Ori_array < tuning_Ori_minus_array])
    total_peakCount = peakCount_plus + peakCount_minus
        
    return float(total_peakCount)


# Input Bead 1EA, Return Value 2EA
def spike(list): # *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array        = np.array(list) 
    peaks, properties = find_peaks(bead_array, prominence=0.03, width=0.05) # parameter 1 : prominence(0.03) / parameter 2 : width(0.05)
    prominences       = np.sum(properties["prominences"])
    widths            = np.sum(properties["widths"])
    
    return (prominences, widths)


# Input Bead 1EA, Return Value 2EA
# Cu 검출력 강화 시 전처리 z-score가 필요하며 이 경우, 2번째 인자인 widths의 값이 이상하여 0으로 제거. 즉, 전처리 적용 시 1번 인자만 쓰기 위한 spike 함수
def spike_second_zero(list): # *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array          = np.array(list) 
    peaks, properties   = find_peaks(bead_array, prominence=0.03, width=0.05) # parameter 1 : prominence(0.03) / parameter 2 : width(0.05)
    prominences         = np.sum(properties["prominences"])
    widths              = np.sum(properties["widths"])
    temp_widths_to_zero = 0
    
    return (prominences, temp_widths_to_zero)

# Input Bead 1EA, Return Value 1EA
def Stability(*list): # * 가능, input type = 1-dim list : ex= [1,2,3]
    
    bead_Ori_array  = np.array(list) 
    bead_mean       = np.mean(bead_Ori_array)
    
    Stability_plus  = np.sum(bead_Ori_array[bead_Ori_array > bead_mean] - bead_mean)
    Stability_minus = np.sum(bead_mean - bead_Ori_array[bead_Ori_array < bead_mean])
    total_Stability = Stability_plus + Stability_minus
    
    return float(total_Stability)


# Input Bead 1EA, Return Value 1EA
def totalEnergy(*list): # * 가능, input type = 1-dim list : ex = [1,2,3]
    
    bead_array             = np.array(list)
    bead_array_totalEnergy = np.sqrt(np.mean(bead_array**2))
        
    return float(bead_array_totalEnergy)


# Input Bead 1EA, Return Value 2EA
def CountAboveBelowMean(*list): # * 가능, input type = 1-dim list : ex = [1,2,3]
    
    bead_array     = np.array(list)
    CountAboveMean = tsfresh.feature_extraction.feature_calculators.count_above_mean(bead_array)
    CountBelowMean = tsfresh.feature_extraction.feature_calculators.count_below_mean(bead_array)
    
    return (CountAboveMean, CountBelowMean)


# Input Bead 1EA, Return Value 2EA
def LongestStrikeAboveBelowMean(list): # *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array             = np.array(list)
    LongestStrikeAboveMean = tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean(bead_array)
    LongestStrikeBelowMean = tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean(bead_array)
    
    return (LongestStrikeAboveMean, LongestStrikeBelowMean)


# Input Bead 1EA, Return Value 1EA
def BenfordCorrelation(list): # *불가, input type = 1-dim list : ex = [1,2,3]

    bead_array          = np.array(list)
    BenfordCorrelation  = tsfresh.feature_extraction.feature_calculators.benford_correlation(bead_array)
    
    return float(BenfordCorrelation)


# Input Bead 1EA, Return Value 1EA
def LempelZivComplexity(list): ## *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array          = np.array(list)
    LempelZivComplexity = tsfresh.feature_extraction.feature_calculators.lempel_ziv_complexity(bead_array, bins = 16)
    
    return float(LempelZivComplexity)


# Input Bead 1EA, Return Value 1EA
def kurtosis(list): # *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array  = np.array(list)
    kurtosis    = tsfresh.feature_extraction.feature_calculators.kurtosis(bead_array)
        
    return float(kurtosis)


# Input Bead 1EA, Return Value 1EA
def skewness(list): # *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array  = np.array(list)
    skewness    = tsfresh.feature_extraction.feature_calculators.skewness(bead_array)
    
    return float(skewness)


# Input Bead 1EA, Return Value 1EA
def abs_energy(list): # *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array = np.array(list)
    abs_energy = tsfresh.feature_extraction.feature_calculators.abs_energy(bead_array)
    
    return float(abs_energy)


# Input Bead 1EA, Return Value 1EA
def sample_entropy(list): # *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array      = np.array(list)
    sample_entropy  = tsfresh.feature_extraction.feature_calculators.sample_entropy(bead_array)
    
    return float(sample_entropy)


# Input Bead 1EA, Return Value 1EA
def autocorrelation(list): ## *불가, input type = 1-dim list : ex = [1,2,3]
    
    bead_array      = np.array(list)
    bead_len        = len(bead_array)
    autocorrelation = tsfresh.feature_extraction.feature_calculators.autocorrelation(bead_array, int(bead_len/2))
    
    return float(autocorrelation)


# Input Bead 1EA, Return Value 1EA
def fourier_entropy(list): # *불가, input type = 1-dim list : ex([1,2,3])
    
    bead_array      = np.array(list)
    fourier_entropy = tsfresh.feature_extraction.feature_calculators.fourier_entropy(bead_array, bins=16)
    
    return float(fourier_entropy)


## Signal processing_1


# Z-score

def z_score(list):
    bead_array        = np.array(list)
    bead_array_zscore = stats.zscore(bead_array)
    
    return bead_array_zscore


# 이동 평균
def ma_time_series(*list): #input type = 1-dim list and int : ex = [1,2,3], 10
    
    bead_matrix = list
    bead_array  = bead_matrix[0]       # 신호처리 대상 Bead
    window_size = int(bead_matrix[1])  # MV의 Window Size 
    
    bead_pd = pd.DataFrame(bead_array)
    result = bead_pd.rolling(window=window_size).mean()
    result_final = result[window_size-1:]
    result_final_format = sum(result_final.values.tolist(), [])
    
    return tuple(result_final_format)

# 지수 이동 평균(LBAD SW 적용 안 함)
def ewm_time_series(*list): #input type = 1-dim list and float : ex = [1,2,3], 10

    bead_matrix = list
    bead_array  = bead_matrix[0]            # 신호처리 대상 Bead
    ewma_hyper  = float(bead_matrix[1])     # ewma_hyper
    
    bead_pd = pd.DataFrame(bead_array)
    result = bead_pd.ewm(alpha=ewma_hyper).mean() # alpha : [ 0 < alpah <= 1]
    result_final_format = sum(result.values.tolist(), [])
    
    return tuple(result_final_format)


# lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowpass_time_series(list): # *불가, input type = 1-dim list and float : ex = [1,2,3]
    
    bead_array_z = stats.zscore(list)
    
    order = 10
    fs = 25000       
    cutoff = 2000
    
    b, a = butter_lowpass(cutoff, fs, order)
    
    T = len(bead_array_z) * 0.00004        # value taken in seconds
    n = int(T * fs) # indicates total samples
    t = np.linspace(0, T, n, endpoint=False)
    y = butter_lowpass_filter(bead_array_z, cutoff, fs, order)
    
    return y


# median filter
def median_time_series(*list): #input type = 1-dim list and int : ex = [1,2,3], 10
    
    bead_matrix = list
    bead_array  = bead_matrix[0]       # 신호처리 대상 Bead
    window_size = int(bead_matrix[1])  # Median filter의 Window Size 
    #window_size = 20
    
    result_final_format = median_filter(bead_ori_1_list, window_size)
    
    return tuple(result_final_format)


# gaussian filter
def gaussian_time_series(*list): #input type = 1-dim list and int : ex = [1,2,3], 10
    
    bead_matrix = list
    bead_array  = bead_matrix[0]      # 신호처리 대상 Bead
    sigma_size  = int(bead_matrix[1])  # gaussian filter의 sigma Size 
    #sigma_size = 3
    
    result_final_format = gaussian_filter1d(bead_ori_1_list, sigma_size)
    
    return tuple(result_final_format)


# maximum filter
def maximum_time_series(*list): #input type = 1-dim list and int : ex = [1,2,3], 10
    
    bead_matrix = list
    bead_array  = bead_matrix[0]      # 신호처리 대상 Bead
    filter_size = int(bead_matrix[1])  # maximum filter의 filter Size 
    #filter_size = 3
    
    result_final_format = maximum_filter1d(bead_ori_1_list, filter_size)
    
    return tuple(result_final_format)


## Signal processing_2 : 함수 input 인자 추가 방식이고 인자가 0일 경우, 예외처리 적용하여 defalut 값으로 신호처리 진행

# 이동 평균
def ma_time_series_arg(*list): #input type = 2-dim list and int : ex = [[1,2,3], [10]]
    
    bead_matrix = list
    bead_array  = bead_matrix[0][0]       # 신호처리 대상 Bead
    window_size = int(bead_matrix[0][1][0])  # MV의 Window Size
    default_window_size = 10
    
    if window_size == 0:       
        bead_pd = pd.DataFrame(bead_array)
        result = bead_pd.rolling(window=default_window_size).mean()
        result_final = result[default_window_size-1:]
        result_final_format = sum(result_final.values.tolist(), [])
        
    else:
        bead_pd = pd.DataFrame(bead_array)
        result = bead_pd.rolling(window=window_size).mean()
        result_final = result[window_size-1:]
        result_final_format = sum(result_final.values.tolist(), [])
    
    return tuple(result_final_format)


# median filter
def median_time_series_arg(*list): #input type = 2-dim list and int : ex = [[1,2,3], [10]]
    
    bead_matrix = list
    bead_array  = bead_matrix[0][0]       # 신호처리 대상 Bead
    window_size = int(bead_matrix[0][1][0])  # Median filter의 Window Size
    default_window_size = 20
    
    if window_size == 0:
        result_final_format = median_filter(bead_array, default_window_size)
    
    else:
        result_final_format = median_filter(bead_array, window_size)
    
    return tuple(result_final_format)


# gaussian filter
def gaussian_time_series_arg(*list): #input type = 2-dim list and int : ex = [[1,2,3], [10]]
    
    bead_matrix = list
    bead_array  = bead_matrix[0][0]       # 신호처리 대상 Bead
    sigma_size = int(bead_matrix[0][1][0])  # gaussian filter의 sigma Size
    default_sigma_size = 5
    
    if sigma_size == 0:
        result_final_format = gaussian_filter1d(bead_array, default_sigma_size)
        
    else:
        result_final_format = gaussian_filter1d(bead_array, sigma_size)
        
    return tuple(result_final_format)


# maximum filter
def maximum_time_series_arg(*list): #input type = 2-dim list and int : ex = [[1,2,3], [10]]
    
    bead_matrix = list
    bead_array  = bead_matrix[0][0]      # 신호처리 대상 Bead
    filter_size = int(bead_matrix[0][1][0])  # maximum filter의 filter Size 
    default_filter_size = 5
    
    if filter_size == 0:
        result_final_format = maximum_filter1d(bead_array, default_filter_size)
        
    else:
        result_final_format = maximum_filter1d(bead_array, filter_size)
    
    return tuple(result_final_format)


# lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowpass_time_series(list): # *불가, input type = 1-dim list and float : ex = [1,2,3]
    
    bead_array_z = stats.zscore(list)
    
    order = 10
    fs = 25000       
    cutoff = 2000
    
    b, a = butter_lowpass(cutoff, fs, order)
    
    T = len(bead_array_z) * 0.00004        # value taken in seconds
    n = int(T * fs) # indicates total samples
    t = np.linspace(0, T, n, endpoint=False)
    y = butter_lowpass_filter(bead_array_z, cutoff, fs, order)
    
    return y


# lowpass filter

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowpass_time_series_arg(*list): # input type = 2-dim list and float : ex = [[1,2,3],[cutoff,fs,order]]
    
    bead_matrix = list
    bead_array_z  = stats.zscore(bead_matrix[0][0])       # 신호처리 대상 Bead
    cutoff = bead_matrix[0][1][0]
    fs = bead_matrix[0][1][1]
    order = bead_matrix[0][1][2]
    default_cutoff = 2000
    default_fs = 25000
    default_order = 10
    
    if cutoff == 0 or fs == 0 or order == 0:
        b, a = butter_lowpass(default_cutoff, default_fs, default_order)
        T = len(bead_array_z) * 0.00004        # value taken in seconds
        n = int(T * fs) # indicates total samples
        t = np.linspace(0, T, n, endpoint=False)
        y = butter_lowpass_filter(bead_array_z, default_cutoff, default_fs, default_order)

    else:
        b, a = butter_lowpass(cutoff, fs, order)
        T = len(bead_array_z) * 0.00004        # value taken in seconds
        n = int(T * fs) # indicates total samples
        t = np.linspace(0, T, n, endpoint=False)
        y = butter_lowpass_filter(bead_array_z, cutoff, fs, order)
    
    return y