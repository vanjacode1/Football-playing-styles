import numpy as np
from numba import njit

@njit
def dtw_distance_numba(ts_a: np.ndarray, ts_b: np.ndarray) -> float:
    len_a, len_b = len(ts_a), len(ts_b)
    dtw_matrix = np.full((len_a + 1, len_b + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            dx = ts_a[i - 1][0] - ts_b[j - 1][0]
            dy = ts_a[i - 1][1] - ts_b[j - 1][1]
            cost = (dx * dx + dy * dy) ** 0.5
            last_min = min(
                dtw_matrix[i - 1, j],    # insertion
                dtw_matrix[i, j - 1],    # deletion
                dtw_matrix[i - 1, j - 1] # match
            )
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[len_a, len_b]

def compute_dtw_distance_matrix(traj_list: list[np.ndarray]) -> np.ndarray:
    """
    Compute a symmetric DTW distance matrix for a list of trajectories
    """
    num_trajs = len(traj_list)
    distance_matrix = np.zeros((num_trajs, num_trajs))
    
    for i in range(num_trajs):
        if i % 5000 == 0:
            print(f"Processing trajectory {i}/{num_trajs}")
        for j in range(i + 1, num_trajs):
            distance = dtw_distance_numba(traj_list[i], traj_list[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  
    return distance_matrix


