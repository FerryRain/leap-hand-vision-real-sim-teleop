from __future__ import annotations

import math
import unittest

from franka_bridge.protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    json_safe,
)


class ProtocolTests(unittest.TestCase):
    def test_strict_round_trip(self) -> None:
        message = {"type": "velocity", "linear": [0.01, 0.0, 0.0]}
        self.assertEqual(decode_message(encode_message(message)), message)

    def test_rejects_non_object_and_non_finite_number(self) -> None:
        with self.assertRaises(ProtocolError):
            decode_message("[]")
        with self.assertRaises(ProtocolError):
            decode_message('{"type":"velocity","x":NaN}')

    def test_json_safe_replaces_non_finite_values(self) -> None:
        value = json_safe({"nan": math.nan, "inf": math.inf})
        self.assertEqual(value, {"nan": None, "inf": None})


if __name__ == "__main__":
    unittest.main()
