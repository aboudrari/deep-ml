import numpy as np
from collections import Counter

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    count = Counter(data).most_common(1)
    mean  = np.mean(data)
    median = np.median(data)
    mode = count[0][0]
    variance = np.var(data)
    std_dev = np.std(data)
    per_25 = np.percentile(data,25)
    per_50 = np.percentile(data,50)
    per_75 = np.percentile(data,75)
    interquartile_range = per_75 - per_25

    return {
    'mean': round(float(mean), 4),
    'median': round(float(median), 4),
    'mode': mode,
    'variance': round(float(variance), 4),
    'standard_deviation': round(float(std_dev), 4),
    '25th_perc