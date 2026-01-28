import numpy as np
from pathlib import Path
import sys
ROOT = Path.cwd().parent 
sys.path.insert(0, str(ROOT))
from playstyle_utils.dtw import dtw_distance_numba

def test_dtw_zero():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert dtw_distance_numba(a, a) == 0

def test_dtw_symmetry():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert dtw_distance_numba(a, b) == dtw_distance_numba(b, a)

