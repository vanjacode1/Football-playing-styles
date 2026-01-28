import numpy as np
from pathlib import Path
import sys
ROOT = Path.cwd().parent 
sys.path.insert(0, str(ROOT))
from playstyle_utils.compositional import aitchison_mean

def test_aitchison_mean():
    X = [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    m = aitchison_mean(X)
    assert np.allclose(m, np.array([0.2, 0.3, 0.5]), atol=1e-9)
    assert np.isclose(m.sum(), 1.0)