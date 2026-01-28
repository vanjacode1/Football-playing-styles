import numpy as np
from pathlib import Path
import sys
ROOT = Path.cwd().parent 
sys.path.insert(0, str(ROOT))
from playstyle_utils.noise import RemoveNoise

def test_noise_last_point():
    rn = RemoveNoise([])
    traj = [[10.0, 10.0], [50.0, 30.0], [105.0, 0.0]]
    assert rn.is_noise_trajectory(traj) is True

def test_noise_contains_0_0():
    rn = RemoveNoise([])
    traj = [[10.0, 10.0], [0.0, 0.0], [50.0, 30.0]]
    assert rn.is_noise_trajectory(traj) is True

def test_noise_last_y():
    rn = RemoveNoise([])
    traj = [[10.0, 10.0], [50.0, 30.0], [60.0, 68.0]]
    assert rn.is_noise_trajectory(traj) is True

def test_noise_small_start_end_distance_is_true():
    """
    Euclidean distance between first & last < 3.5 => noise
    """
    rn = RemoveNoise([])
    traj = [[10.0, 10.0], [11.0, 11.0]]

    assert rn.is_noise_trajectory(traj) is True