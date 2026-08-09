from __future__ import annotations

import ast
import unittest
from pathlib import Path

import mujoco
import numpy as np
import yaml
from src.leap_hand_mapping import JOINT_NAMES
from src.leap_hand_scene import build_leap_demo_scene
from teleop import set_simulated_hand_target, simulated_joint_positions

ROOT = Path(__file__).resolve().parents[1]


class LeapHandMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))
        cls.open_targets = np.asarray(
            cls.config["mapping"]["joint_open_rad"], dtype=np.float64
        )

    def _scene(self):
        return build_leap_demo_scene(
            ROOT, self.config, open_joint_targets=self.open_targets
        )

    def test_scene_contains_only_fixed_wrist_leap_hand(self) -> None:
        scene = self._scene()
        self.assertEqual(scene.model.nu, 16)
        self.assertEqual(scene.model.nmocap, 1)
        body_names = {
            mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, index)
            for index in range(scene.model.nbody)
        }
        geom_names = {
            mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(scene.model.ngeom)
        }
        self.assertFalse(any(name and "panda" in name.lower() for name in body_names))
        self.assertNotIn("target_object", body_names)
        self.assertNotIn("teleop_floor", geom_names)

    def test_shared_command_drives_all_16_simulation_actuators(self) -> None:
        scene = self._scene()
        fixed_palm = tuple(self.config["wrist"]["neutral_position_m"])
        command = self.open_targets + 0.2
        applied = set_simulated_hand_target(scene, command, fixed_palm)
        np.testing.assert_allclose(scene.data.ctrl, applied)
        np.testing.assert_allclose(
            scene.data.mocap_pos[scene.palm_mocap_id], fixed_palm
        )
        for _ in range(20):
            mujoco.mj_step(scene.model, scene.data)
        positions = simulated_joint_positions(scene)
        self.assertEqual(positions.shape, (16,))
        self.assertTrue(np.isfinite(positions).all())

    def test_entry_imports_mujoco_but_no_arm_module(self) -> None:
        tree = ast.parse((ROOT / "teleop.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertTrue(any(name.startswith("mujoco") for name in imported))
        self.assertFalse(
            any("franka" in name or "robot_arm" in name for name in imported)
        )

    def test_joint_order_matches_mapping_order(self) -> None:
        self.assertEqual(len(JOINT_NAMES), 16)
        self.assertEqual(JOINT_NAMES[0:4], ("if_mcp", "if_rot", "if_pip", "if_dip"))


class RepositoryScopeTests(unittest.TestCase):
    def test_repository_contains_no_powershell_launcher(self) -> None:
        self.assertEqual(list(ROOT.rglob("*.ps1")), [])

    def test_readme_uses_direct_python_commands(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("python teleop.py --device real", readme)
        self.assertNotIn(".ps1", readme.lower())

    def test_safe_default_does_not_select_real_hardware(self) -> None:
        config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))
        self.assertEqual(config["run"]["device"], "mock")
        self.assertNotIn("port", config["hardware"])


if __name__ == "__main__":
    unittest.main()
