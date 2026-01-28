from pathlib import Path
import sys
import pandas as pd
ROOT = Path.cwd().parent 
sys.path.insert(0, str(ROOT))
from playstyle_utils.phases import MakeMovementChains

def event(player: str, i: int) -> dict:
    return {"player": player, "i": i, "type": "pass"}

def test_make_movement_chains_structure():
    phase = [[event("A", 0), event("A", 1), event("B", 2), event("B", 3), event("C", 4), event("C", 5)]]

    out = MakeMovementChains(phase)

    # top level: list
    assert isinstance(out, list)

    # each element: list (a chain chunk)
    assert all(isinstance(chunk, list) for chunk in out)

    # inside each chunk: dict events
    assert all(isinstance(evt, dict) for chunk in out for evt in chunk)

def test_make_movement_chains_on_empty_phase():
    out = MakeMovementChains([])
    assert out == []


def test_make_movement_chains_on_short_sequence():
    seq = [event("A", i) for i in range(4)]
    out = MakeMovementChains([seq])

    assert out == []

def test_make_movement_chains_player_changes():
    # players change every event: A, B, C, D, E
    # since 4 event make up one movement chain
    # there should be two movement chains, chain 1: [A, B, C, D] and chain 2: [B, C, D, E]
    seq = [event(p, i) for i, p in enumerate(["A", "B", "C", "D", "E"])]
    out = MakeMovementChains([seq])

    assert len(out) == 2

    # chain 1
    assert [e["player"] for e in out[0]] == ["A", "B", "C", "D"]
    # chain 2
    assert [e["player"] for e in out[1]] == ["B", "C", "D", "E"]

def test_make_movement_chains_one_player_no_changes():
    seq = [event("A", i) for i in range(10)]
    out = MakeMovementChains([seq])
    assert out == []