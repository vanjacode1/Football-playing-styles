import numpy as np
from numba import njit
from tqdm import tqdm
from .dtw import dtw_distance_numba 
import random
from collections import defaultdict, Counter

def assign_to_nearest_medoids(medoid_indices: list[int], medoid_trajs: list[np.ndarray], all_movement_chain_coordinates: dict[str, list[list[list[float]]]]) -> dict[str, list[int]]:
    """
    Assign each trajectory to nearest medoid
    """
    assignments = {}

    for club_id, traj_list in tqdm(all_movement_chain_coordinates.items(), desc="Assigning trajectories"):
        club_assignments = []

        for traj in traj_list:
            traj_np = np.array(traj, dtype=np.double)

            # Find closest medoid (based on DTW)
            best_idx = 0
            best_dist = float("inf")
            for i in range(len(medoid_trajs)):
                dist = dtw_distance_numba(traj_np, medoid_trajs[i])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # Assign to actual medoid index
            club_assignments.append(medoid_indices[best_idx])

        assignments[club_id] = club_assignments

    return assignments

def split(movement_chain_clusters: dict[str, list[int]], seed=None) -> dict[str, tuple[Counter, Counter]]:
    if seed is not None:
        random.seed(seed)

    club_matches = defaultdict(list)

    for key, values in movement_chain_clusters.items():
        match_id, club = key.split('_')
        club_matches[club].append((match_id, values))

    club_A = {}
    club_B = {}

    for club, matches in club_matches.items():
        random.shuffle(matches)
        half = len(matches) // 2
        A_matches = matches[:half]
        B_matches = matches[half:]

        # Collect all values for A and B halves
        values_A = []
        values_B = []
        for _, vals in A_matches:
            values_A.extend(vals)
        for _, vals in B_matches:
            values_B.extend(vals)

        # Build Counters
        club_A[club] = Counter(values_A)
        club_B[club] = Counter(values_B)

    team_counters = {}

    for team in club_A:
        counter_A = club_A[team]
        counter_B = club_B.get(team, Counter())
        team_counters[team] = (counter_A, counter_B)

    return team_counters

def manhattan_dist(c1: dict[int, float], c2: dict[int, float]) -> float:
    """
    Compute L1 distance between two count/frequency dicts.
    """
    keys = sorted(set(c1) | set(c2))
    return sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in keys)

def compute_stability_metric(team_counters:dict[str, tuple[Counter, Counter]], normalize=True) -> tuple[float, float]:
    """
    Compute stability based on whether team A is closest to its own B split.
    """
    all_vectors = {}
    for team, (cA, cB) in team_counters.items():
        if normalize:
            totalA = sum(cA.values())
            totalB = sum(cB.values())

            # build frequency dicts
            freqA = {k: v / totalA for k, v in cA.items()} if totalA > 0 else {}
            freqB = {k: v / totalB for k, v in cB.items()} if totalB > 0 else {}
            all_vectors[f"{team}_A"] = freqA
            all_vectors[f"{team}_B"] = freqB
        else:
            all_vectors[f"{team}_A"] = cA
            all_vectors[f"{team}_B"] = cB

    correct_matches = 0
    correct_matches2 = 0

    n_teams = len(team_counters)

    for team in team_counters:
        A_key = f"{team}_A"
        B_key = f"{team}_B"
        A_vec = all_vectors[A_key]

        distances = []
        for other_key, other_vec in all_vectors.items():
            if other_key == A_key:
                continue
            dist = manhattan_dist(A_vec, other_vec)
            distances.append((other_key, dist))

        distances.sort(key=lambda x: x[1])

        if distances[0][0] == B_key:
            correct_matches += 1

        top_3 = [key for key, _ in distances[:3]]
        if B_key in top_3:
            correct_matches2 += 1

    avg_rank = correct_matches / n_teams if n_teams else 0.0
    avg_rank_top3 = correct_matches2 / n_teams if n_teams else 0.0
    return avg_rank, avg_rank_top3
