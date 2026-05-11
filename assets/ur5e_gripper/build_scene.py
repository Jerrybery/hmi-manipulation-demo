"""
Build ur5e_gripper/scene.xml by composing UR5e + Robotiq 2F-85 via MjSpec.attach().

Composition strategy:
- Load arm and gripper as separate MjSpec objects.
- Patch gripper mesh file paths to absolute paths (gripper has its own meshdir
  that differs from the arm's; MuJoCo only supports one meshdir per serialized
  file, so we must use absolute paths for the secondary mesh set).
- Call MjSpec.attach(gripper_spec, site=attachment_site, prefix='gripper_') to
  graft the gripper body tree onto the UR5e wrist flange.
- Update the existing 'home' keyframe qpos/ctrl to include the 12 nq / 7 nu
  dimensions of the combined model.
- Add a new 'unload' keyframe.
- Compile, serialize to XML, write to assets/ur5e_gripper/scene.xml.

Usage (from repo root):
    uv run python assets/ur5e_gripper/build_scene.py

Result:
    nq: 12  (arm joints 0-5, gripper joints 6-11)
    nv: 12
    nu:  7  (arm actuators 0-5, gripper_fingers_actuator at index 6)
    EE site name: 'attachment_site'  (not namespaced)
"""
import pathlib
import mujoco

HERE = pathlib.Path(__file__).parent.resolve()
MENAGERIE = HERE.parent / "mujoco_menagerie"

UR5E_XML  = MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"
F85_XML   = MENAGERIE / "robotiq_2f85_v4" / "2f85.xml"
OUT_XML   = HERE / "scene.xml"

# ── 1. Load sub-specs ────────────────────────────────────────────────────────
arm_spec     = mujoco.MjSpec.from_file(str(UR5E_XML))
gripper_spec = mujoco.MjSpec.from_file(str(F85_XML))

# ── 2. Fix gripper mesh paths to absolute ────────────────────────────────────
# MuJoCo serializes only one meshdir into the combined XML (the arm's).
# Without this, gripper STL files would be looked up under the arm's assets/
# directory and fail to load.
gripper_assets = MENAGERIE / "robotiq_2f85_v4" / "assets"
for mesh in gripper_spec.meshes:
    if mesh.file:
        mesh.file = str(gripper_assets / mesh.file)

# Set absolute meshdir for the arm so the output XML is relocatable.
arm_spec.meshdir = str(MENAGERIE / "universal_robots_ur5e" / "assets")

# ── 3. Attach gripper at UR5e wrist flange ───────────────────────────────────
# All gripper names are prefixed with "gripper_" to avoid collisions.
#   Combined joint order: shoulder_pan(0)..wrist_3(5), then
#   gripper_left_driver_joint(6), gripper_left_spring_link_joint(7),
#   gripper_left_follower(8), gripper_right_driver_joint(9),
#   gripper_right_spring_link_joint(10), gripper_right_follower_joint(11)
#   Actuator index 6: gripper_fingers_actuator (ctrlrange 0=open, 255=closed)
arm_spec.attach(gripper_spec, site=arm_spec.site("attachment_site"), prefix="gripper_")

# ── 4. Keyframes ─────────────────────────────────────────────────────────────
#  nq == 12 (arm 0-5, gripper 6-11)  nu == 7 (arm 0-5, gripper ctrl 6)
#  Gripper joints all-zero = open;   gripper ctrl 0 = open.

arm_home   = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
arm_unload = [ 0.0,    -1.5708, 1.5708, -1.5708,  0.0,    0.0]
gripper_open_qpos = [0.0] * 6   # 6 gripper joints
gripper_open_ctrl = [0.0]        # 1 gripper actuator (ctrl=0 → open)

# Reuse the existing 'home' key inherited from ur5e.xml — expand its
# qpos/ctrl to cover the newly attached gripper dimensions.
home_key = arm_spec.key("home")
home_key.qpos = arm_home + gripper_open_qpos
home_key.ctrl  = arm_home + gripper_open_ctrl

# Add 'unload' keyframe (arm pointing forward, gripper open).
unload_key = arm_spec.add_key()
unload_key.name = "unload"
unload_key.qpos = arm_unload + gripper_open_qpos
unload_key.ctrl  = arm_unload + gripper_open_ctrl

# ── 5. Compile and serialize ─────────────────────────────────────────────────
model = arm_spec.compile()
print(f"Compiled OK — nq:{model.nq}  nv:{model.nv}  nu:{model.nu}")

OUT_XML.write_text(arm_spec.to_xml())
print(f"Wrote {OUT_XML}")
