from pathlib import Path
import sys
import pandas as pd
ROOT = Path.cwd().parent 
sys.path.insert(0, str(ROOT))
from playstyle_utils.phases import SplitPossessionPhases, FilterPhases

def test_convert_nice_time_to_seconds():
    s = SplitPossessionPhases()
    assert s.convert_nice_time_to_seconds("3m40s") == 220

def test_time_diff_labeling():
    s = SplitPossessionPhases()
    df = pd.DataFrame({"nice_time": ["0m00s", "0m05s", "0m20s"]})
    df = s.add_time_seconds_column(df)
    df = s.compute_and_label_time_diff(df)
    # 0->5 is <=10 => No, 5->20 is 15 => Yes
    assert df["time_diff_label"].tolist() == ["No", "No", "Yes"]

def test_remove_unwanted_actions():
    phase = [[{"type": "pass"}, {"type": "foul"}, {"type": "shot"}]]
    out = FilterPhases(phase).filter()
    assert [e["type"] for e in out[0]] == ["pass", "shot"]

def test_drop_set_piece_start():
    phase = [[{"type": "corner"}, {"type": "pass"}], [{"type": "pass"}]]
    out = FilterPhases(phase).filter()
    assert len(out) == 1
    assert out[0][0]["type"] == "pass"

def test_drop_dribble():
    phase = [[{"type": "dribble"}, {"type": "pass"}, {"type": "shot"}]]
    out = FilterPhases(phase).filter()
    assert out[0][0]["type"] == "pass"
    assert len(out[0]) == 2

def test_remove_own_goal_sequence():
    phase = [[{"type": "dribble"}, {"type": "pass"}, {"type": "owngoal"}]]
    out = FilterPhases(phase).filter()
    assert len(out) == 0

