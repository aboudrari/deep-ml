import numpy as np

def k_nearest_neighbors(points, query_point, k):
    distance = []
    for point in points :
        dist = np.sqrt(np.sum((np.array(point) - np.array(query_point)) ** 2))
        distance.append(dist)
    sorted_indices = np.argsort(distance, kind='stable')[:k]
    result = []
    for index in sorted_indices :
        result.append(points[index])
    return result
        