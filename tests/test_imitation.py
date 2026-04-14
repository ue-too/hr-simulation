import pytest
import numpy as np

from horse_racing.imitation import _encode_action, _snap_to_index, extract_demonstrations
from horse_racing.action import TANGENTIAL_LEVELS, NORMAL_LEVELS, NUM_NORMAL


class TestSnapToIndex:
    def test_exact_match(self):
        assert _snap_to_index(0.0, TANGENTIAL_LEVELS) == 1  # 0 is at index 1

    def test_closest_match(self):
        assert _snap_to_index(0.6, TANGENTIAL_LEVELS) == 3  # closest to 0.5

    def test_boundary(self):
        assert _snap_to_index(-1.0, NORMAL_LEVELS) == 0
        assert _snap_to_index(1.0, NORMAL_LEVELS) == 8


class TestEncodeAction:
    def test_cruise(self):
        # t=0, n=0 → ti=1, ni=4 → 1*9+4 = 13
        assert _encode_action(0.0, 0.0) == 13

    def test_full_push(self):
        # t=1, n=0 → ti=5, ni=4 → 5*9+4 = 49
        assert _encode_action(1.0, 0.0) == 49

    def test_roundtrip(self):
        from horse_racing.action import decode_action
        for idx in range(54):
            t, n = decode_action(idx)
            assert _encode_action(t, n) == idx


class TestExtractDemonstrations:
    def test_extracts_from_recording(self, tmp_path):
        import json
        recording = {
            "horseCount": 2,
            "finishOrder": [0, 1],
            "totalTicks": 3,
            "frames": [
                {
                    "tick": i,
                    "horses": [
                        {"id": 0, "finished": False, "obs": [float(i)] * 141},
                        {"id": 1, "finished": False, "obs": [float(i)] * 141},
                    ],
                    "inputs": {
                        "0": {"t": 1.0, "n": 0.0},
                        "1": {"t": 0.0, "n": 0.0},
                    },
                }
                for i in range(3)
            ],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(recording))

        obs, actions = extract_demonstrations(str(path), player_horse_id=0)
        assert obs.shape == (3, 141)
        assert actions.shape == (3,)
        assert all(a == 49 for a in actions)  # t=1, n=0

    def test_skips_finished_horse(self, tmp_path):
        import json
        recording = {
            "horseCount": 1,
            "finishOrder": [0],
            "totalTicks": 3,
            "frames": [
                {
                    "tick": 0,
                    "horses": [{"id": 0, "finished": False, "obs": [0.0] * 141}],
                    "inputs": {"0": {"t": 0.0, "n": 0.0}},
                },
                {
                    "tick": 1,
                    "horses": [{"id": 0, "finished": True, "obs": [1.0] * 141}],
                    "inputs": {"0": {"t": 0.0, "n": 0.0}},
                },
            ],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(recording))

        obs, actions = extract_demonstrations(str(path), player_horse_id=0)
        assert obs.shape == (1, 141)
