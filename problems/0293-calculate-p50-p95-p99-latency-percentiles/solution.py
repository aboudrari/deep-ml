import numpy as np
import math
def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Your code here
    if not latencies :
        return {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}
    latencies = sorted(latencies)
    n = len(latencies)
    position_50 = (50 / 100) * (n - 1)
    fraction_50 = position_50 - math.floor(position_50)
    P50 = latencies[math.floor(position_50)] + fraction_50 * (latencies[math.ceil(position_50)] - latencies[math.floor(position_50)]) 
    position_95 = (95 / 100) * (n - 1)
    fraction_95 = position_95 - math.floor(position_95)
    P95 = latencies[math.floor(position_95)] + fraction_95 * (latencies[math.ceil(position_95)] - latencies[math.floor(position_95)]) 
    position_99 = (99 / 100) * (n - 1)
    fraction_99 = position_99 - math.floor(position_99)
    P99 = latencies[math.floor(position_99)] + fraction_99 * (latencies[math.ceil(position_99)] - latencies[math.floor(position_99)])

    return {'P50': round(P50,4), 'P95': round(P95,4), 'P99': round(P99,4)}

