# hmi-manipulation-demo

单台 macOS 笔记本上跑通的"机械臂仿真 + 摄像头手势仲裁"完整闭环 demo：UR5e + Robotiq 2F-85 夹爪在 MuJoCo 物理引擎里画圆并预览未来 2 秒轨迹；操作员对镜头做"开掌正面"手势触发**卸载-回 home-待机**序列；在 IDLE 状态下做"握拳"手势重新启动；Esc 键作为独立急停。

用于验证"预测性意图可视化（轨迹球体预览）+ 双手势人机仲裁 + 自动卸载/恢复状态机"的整体软件架构，为后续 Safety Bubble、Authority Bar、完整 Ghost Arm 等可视化层迭代打底。

---

## 演示功能（MVP）

| 功能 | 说明 |
|---|---|
| 圆周轨迹跟踪 | UR5e 末端在桌面上方画圆（默认半径 0.15 m，圆心 (0.5, 0, 0.3)，~10 s 一圈） |
| 差分 IK | DLS（damped least squares）雅可比伪逆，仅约束位置、姿态固定朝下 |
| **未来 2 s 轨迹预览** | 20 个半透明球体沿预测圆周分布（每 100 ms 一个），近期绿 `RGB(0.1,0.9,0.2)` → 远期淡黄 `RGB(0.85,0.75,0.15)`，alpha=0.5；仅 RUNNING 状态显示 |
| **开掌手势 → 卸载** | MediaPipe Hand Landmarker 检测开掌 + 掌心朝镜头；5 帧迟滞防抖（~165 ms @ 30 fps）；触发后机械臂走到可配置的 `unload_pose`（默认 `[0, -π/2, π/2, -π/2, 0, 0]` 垂直向下），夹爪同步打开 |
| **回 home pose** | 卸载完成后 1.5 s 关节空间线性插值回 home keyframe；夹爪保持打开 |
| **IDLE 待机** | 机械臂停在 home，等待恢复手势 |
| **握拳手势 → 重启** | IDLE 状态下检测握拳（可配 `gesture.enable_fist_resume: false` 关掉）→ 状态机回 RUNNING，重新画圆 |
| **Robotiq 2F-85 夹爪** | mujoco_menagerie 的 2F-85 模型用 `MjSpec.attach` 嫁接到 UR5e wrist；ctrl 0=open / 255=closed |
| Esc 急停 | 与手势独立的 OR 通道；任何状态按 Esc 冻结当前状态机推进，再按一次解除 |
| 状态条 | 6 色编码：RUNNING（绿）/ UNLOADING（橙）/ RETURNING HOME（蓝）/ IDLE — Show closed fist to resume（紫）/ E-STOP（红）/ NO CAMERA（灰） |
| 无摄像头降级 | 摄像头打不开或中途失联 → 状态条 NO CAMERA，仿真继续 |
| 主相机视角 | `MujocoRenderer` 默认 lookat=(0.3,0,0.4), distance=2.2 m, az=135°, elev=−25°，保证整臂 + 夹爪 + 工作空间球体完整入框 |

**不在 MVP 范围**（后续迭代）：Dynamic Safety Bubble、Authority Bar 平滑过渡、完整半透明 Ghost Arm 轮廓、MPC 预测、多手 / 多人场景。

---

## 技术栈

| 类别 | 选型 | 用途 |
|---|---|---|
| Python 运行时 | Python 3.12（`uv` 管理） | MediaPipe 在 macOS arm64 PyPI wheel 当前最高仅到 3.12 |
| 物理仿真 | MuJoCo ≥ 3.2（实测 3.8） | 高保真动力学，自带 `Renderer` 离屏渲染 + `MjvScene.geoms` 支持运行时注入额外几何 |
| 机械臂模型 | mujoco_menagerie / `universal_robots_ur5e` | 6 DOF UR5e，含 `attachment_site` |
| **夹爪模型** | mujoco_menagerie / `robotiq_2f85_v4` | 2F-85 夹爪，1 actuator (`fingers_actuator`, 0=open / 255=closed)，6 joints |
| **模型组装** | `mujoco.MjSpec` Python API | `assets/ur5e_gripper/build_scene.py` 把 UR5e + 2F-85 attach 成 `nq=12, nu=7` 的组合模型，并写出 `home` / `unload` 两个 keyframe |
| 手势识别 | MediaPipe Tasks Hand Landmarker（`mediapipe.tasks.python.vision.HandLandmarker`） | 21 点 3D 关键点 + 左右手分类；首次运行自动下载 `hand_landmarker.task` 到 `~/.cache/hmi-demo/` |
| 摄像头采集 | OpenCV (`cv2.VideoCapture`) | macOS AVFoundation 默认后端 |
| GUI | PyQt6 | QMainWindow + QThread + Qt signal 跨线程通信 |
| 数值 | NumPy < 2 | mediapipe 0.10.x 仍要求 numpy 1.x |
| 配置 | PyYAML | `configs/demo.yaml` |
| 测试 | pytest | 24 个单元测试覆盖 config / world / trajectory / IK / gesture |

---

## 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                  Qt Main Thread (event loop)                 │
│                                                              │
│  ┌──────────────────┐   QTimer 100Hz   ┌──────────────────┐  │
│  │ CircleTrajectory │ ───────────────> │  diff_ik_dls     │  │
│  │  .tick(dt)       │   x_target       │  (DLS 雅可比)    │  │
│  └──────────────────┘                  └────────┬─────────┘  │
│                                                 │ q_target[:6]│
│  ┌──────────────────┐                           │ +gripper ctrl
│  │ MotionState      │ ─── RUNNING/UNLOADING/    │            │
│  │  state machine   │     RETURNING_HOME/IDLE   │            │
│  └──────────────────┘                    ┌──────▼─────────┐  │
│                                          │  World.step()  │  │
│                                          │  (mj_step ×5)  │  │
│                                          └──────┬─────────┘  │
│  ┌──────────────────┐                           │            │
│  │ TrajectoryPreview│ ─── 20 球体 [SphereGeom]  │            │
│  │ (RUNNING only)   │                          │            │
│  └────────┬─────────┘                   ┌──────▼─────────┐  │
│           └────────── extra_geoms ────> │ Renderer.grab  │  │
│                                          │ (mjv_initGeom) │  │
│                                          └──────┬─────────┘  │
│                                                 │ QImage     │
│  ┌──────────────────────┐  cam QImage    ┌──────▼─────────┐  │
│  │ HMIWindow (PyQt6)    │ <───────────── │ QLabel.sim     │  │
│  │  + Esc keyPressEvent │                │ QLabel.cam     │  │
│  │  + 6-color StatusBar │                │ QStatusBar     │  │
│  └──────────▲───────────┘                └────────────────┘  │
│             │ Qt signal: gestureUpdated(str, QImage)         │
└─────────────┼────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────┐
│                 VisionThread (QThread, 30Hz)                 │
│                                                              │
│  cv2.VideoCapture.read()                                     │
│   → cv2.flip (selfie 镜像)                                   │
│   → MediaPipe HandLandmarker.detect_for_video                │
│   → is_open_hand + is_palm_facing_camera + is_closed_fist    │
│   → 两个独立 HysteresisFilter (palm / fist, each 5 frames)   │
│   → emit gestureUpdated("open_palm"/"closed_fist"/"", QImg)  │
└──────────────────────────────────────────────────────────────┘
```

**关键不变量**

- `mjData` 只被 Qt 主线程读写（无锁）
- VisionThread 只产数据，通过 Qt signal 把 `(gesture_name: str, QImage)` 投递到主线程 event queue；优先级 fist > palm
- `_latest_gesture` 是主线程一致的"邮箱"：信号槽（QueuedConnection）和 QTimer 都在主线程跑，无竞争
- `diff_ik_dls` 只控制前 6 个关节角；`data.ctrl[6]` 单独写夹爪开合值
- 退出顺序：`closeEvent` → `timer.stop()` → `vision_thread.request_stop()` + `wait(2000)` → `renderer.close()`；每步都有 `hasattr` 守护

---

## 状态机

```
            open_palm latched                       (auto, arm_progress>=1)
RUNNING ──────────────────────────> UNLOADING ─────────────────────────> RETURNING_HOME
   ▲                                  arm→unload_pose 1.5s                 arm→q_home 1.5s
   │                                  gripper close→open 0.5s              gripper stays open
   │                                                                              │
   │                                                                              │ (auto, progress>=1)
   │                                                                              │
   │                                                                              v
   │                            closed_fist latched +                            IDLE
   └──────────────────────────  enable_fist_resume  ──────────────────────  (await fist)
```

- **RUNNING**：`trajectory.tick(dt)` 推进圆周相位，`diff_ik_dls` 跟踪；夹爪保持 `close_ctrl`；轨迹预览 20 球体渲染
- **UNLOADING**：arm `lerp(q_at_release, unload_pose, s)`，s 在 `unload_duration_s` 秒内 0→1；gripper 独立进度 `min(1, s * unload_duration_s / gripper_release_duration_s)`，从 `close_ctrl` 渐变到 `open_ctrl`
- **RETURNING_HOME**：arm `lerp(unload_pose, q_home[:6], s)`，gripper 停在 `open_ctrl`
- **IDLE**：arm 钉在 `q_home[:6]`，gripper 在 `open_ctrl`；状态栏紫色 `IDLE — Show closed fist to resume`
- **Esc 急停**：独立 `estop_frozen` 标志，`_tick` 开头早返回 → 所有状态机推进暂停，状态栏红色 `E-STOP` 覆盖；再按 Esc 恢复

---

## 项目结构

```
hmi-manipulation-demo/
├── pyproject.toml                  # uv 项目，依赖锁定
├── .python-version                 # 3.12
├── configs/
│   └── demo.yaml                   # 全部可调参数
├── assets/
│   ├── mujoco_menagerie/           # shallow clone（gitignored，~2 GB）
│   └── ur5e_gripper/               # 本项目组合模型
│       ├── build_scene.py          # MjSpec 组装脚本（UR5e + 2F-85）
│       └── scene.xml               # 组装结果（绝对路径 mesh，机器相关）
├── tests/                          # 24 个 pytest 单元测试
└── src/hmi_demo/
    ├── __main__.py                 # python -m hmi_demo 入口
    ├── app.py                      # QApplication 装配 + 顶层错误对话框
    ├── config.py                   # 9 个 frozen dataclass + load_config + 形状校验
    ├── sim/
    │   ├── world.py                # MjModel + MjData 包装，home keyframe
    │   ├── trajectory.py           # CircleTrajectory（纯 NumPy）
    │   └── ik.py                   # diff_ik_dls 纯函数（DLS 雅可比）
    ├── vision/
    │   └── hand_thread.py          # VisionThread (cv2 + HandLandmarker + 双 gesture signal)
    ├── ui/
    │   ├── render.py               # MujocoRenderer (numpy → QImage) + SphereGeom + 默认相机
    │   └── hmi_window.py           # QMainWindow + 4-state MotionState + 100Hz QTimer
    └── utils/
        └── gesture.py              # is_open_hand / is_palm_facing_camera / is_closed_fist / HysteresisFilter
```

---

## 配置（`configs/demo.yaml`）

```yaml
sim:
  mjcf: assets/ur5e_gripper/scene.xml
  ee_site: attachment_site         # 末端 site 名（必须存在于 MJCF）
  home_keyframe: home              # MJCF keyframe 名，q_home 由此读取
  control_hz: 100                  # 外环控制频率 → QTimer 周期 10 ms
  substeps: 5                      # 每外环跑 5 次 mj_step（dt=0.002）

trajectory:
  center: [0.5, 0.0, 0.3]
  radius: 0.15
  omega: 0.6                       # rad/s, 2π/0.6 ≈ 10.5 s 一圈

trajectory_preview:                # 未来轨迹预览
  horizon_s: 2.0                   # 预览多远
  n_samples: 20                    # 多少个球
  sphere_radius: 0.012
  alpha: 0.5                       # 透明度
  color_near: [0.1, 0.9, 0.2]      # 近期色（绿）
  color_far:  [0.85, 0.75, 0.15]   # 远期色（淡黄）

ik:
  damping: 0.05                    # DLS λ
  kp: 0.5                          # 比例增益（>1 会引起振荡）

gripper:
  open_ctrl: 0                     # ctrl=0 → 完全张开
  close_ctrl: 255                  # ctrl=255 → 完全闭合

camera:
  device: 0
  width: 640
  height: 480
  fps: 30

gesture:
  hold_frames: 5                   # 迟滞防抖 latch/release 帧数
  enable_palm_check: true          # false 则任意"开掌"都触发，不要求掌心朝向
  enable_fist_resume: true         # false 则 IDLE 状态无法用握拳恢复

recovery:
  mode: return_home                # 仅当 mode=return_home 时走 UNLOADING/RETURNING_HOME 流程
  return_duration_s: 1.5           # RETURNING_HOME 阶段时长
  unload_pose: [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]   # UNLOADING 目标 6 关节角
  unload_duration_s: 1.5           # UNLOADING 阶段时长
  gripper_release_duration_s: 0.5  # 夹爪开合时长（与 unload 平行）

ui:
  sim_view_size: [640, 480]
  cam_view_size: [320, 240]
```

---

## 安装与运行

**前置条件**：macOS（Apple Silicon 推荐）+ `uv` + `git`。`uv` 会自动安装 Python 3.12。

```bash
git clone https://github.com/Jerrybery/hmi-manipulation-demo.git
cd hmi-manipulation-demo

git clone --depth=1 https://github.com/google-deepmind/mujoco_menagerie.git \
    assets/mujoco_menagerie

uv sync                                              # 装依赖
uv run python assets/ur5e_gripper/build_scene.py     # 组装 UR5e+夹爪 MJCF (本机绝对路径)
uv run python -m hmi_demo                            # 启动 demo
```

`build_scene.py` 把 mesh 绝对路径写进 `scene.xml`，所以**每台机器首次运行前需要跑一遍**。

**首次启动**：
- 自动下载 `hand_landmarker.task`（~7 MB）到 `~/.cache/hmi-demo/`
- macOS 弹摄像头权限。如未弹出，去 System Settings → Privacy & Security → Camera 给当前终端授权后重启 demo

**手动验收清单**（6 步完整流程）：

1. 启动 → 左侧仿真画面 UR5e + 夹爪画圆；**前方 20 个绿→黄半透明球体**勾出未来 2 s 轨迹；右侧摄像头预览 + 21 点骨架
2. 对镜头开掌正面 → ~165 ms 后状态条变橙 `UNLOADING`，机械臂离开圆周走向 unload pose（垂直向下），夹爪同步打开，轨迹球体消失
3. 卸载完成（自动）→ 状态条变蓝 `RETURNING HOME`，机械臂走回 home pose，夹爪保持打开
4. 到达 home（自动）→ 状态条变紫 `IDLE — Show closed fist to resume`
5. 对镜头握拳 → ~165 ms 后状态条变绿 `RUNNING`，机械臂重新开始画圆，球体重新出现
6. 任何时刻按 **Esc** → 状态条变红 `E-STOP`，再按一次解除

如果开掌不触发，把 `configs/demo.yaml` 的 `gesture.enable_palm_check` 改成 `false`，跳过掌心朝向检测。

---

## 测试

```bash
uv run pytest -v          # 24 个单元测试
```

| 测试文件 | 数量 | 范围 |
|---|---|---|
| `test_config.py` | 3 | YAML 加载 + recovery.mode 校验 + unload_pose 长度校验 |
| `test_world.py` | 2 | MJCF 加载（nq=12, nu=7） + 100 步无 NaN + 未知 site 报错 |
| `test_trajectory.py` | 5 | 起点 / 周期闭合 / `tick(0)` 不前进 / 恢复推进 / reset |
| `test_ik.py` | 2 | 差分 IK 收敛到 < 1 mm / 关节限位裁剪 |
| `test_gesture.py` | 12 | 开掌 + 掌心朝向 + 握拳 + 部分卷曲 + slack 区间 + 5 帧迟滞 |

视觉与 GUI 流程不做自动测试（cv2 + Qt 集成测试投入产出比低）；仅做 import smoke 和手动验收。

---

## 已知限制

- 摄像头分辨率/帧率只支持设备能商定的近似值（cap.set 失败时仅打印 warning）
- IK 仅约束位置 + 固定姿态朝下，不处理姿态目标（6 DOF 姿态规划）
- 轨迹预览基于运动学外推（解析圆周延拓），未考虑加速度突变 / 关节限位 / 动力学约束
- 手势词汇表只有 2 个（开掌 / 握拳），无法表达细粒度指令
- MediaPipe handedness 在 `cv2.flip` 后是图像相对的，VisionThread 内部已做反转还原
- `mediapipe>=0.10.21,<0.11` 锁版本：legacy `mp.solutions` 在 0.10.x 末期被移除，本项目用新的 `mediapipe.tasks.python.vision.HandLandmarker`
- 组合 MJCF (`assets/ur5e_gripper/scene.xml`) 内含绝对 mesh 路径，**机器迁移时需 rerun `build_scene.py`**
- 用户测试样本仅 10 人，长稳定性 / 强弱光 / 多用户并发未做基准

---

## 后续迭代（Roadmap）

- **完整 Ghost Arm**：从当前球体序列升级为半透明整条机械臂轮廓投影（每 0.5 s 一个完整 ghost）
- **Dynamic Safety Bubble**：以末端为中心、`R = R_base + k·|v_ee|` 的半透明球体
- **Authority Bar**：HMI 顶部条带，AI ↔ Human 控制权重二阶低通过渡 + 自然语言提示
- **MPC 预测**：从运动学外推升级到模型预测控制，考虑加速度与动力学约束
- **3D 感知**：可选 Intel RealSense 深度相机替代 MediaPipe 2D
- **人体姿态预测**：MediaPipe Pose 全身骨架预测人移动方向（提前 1 s 预判）
- **手势词汇扩展**：加 OK / 指向 / 数字等手势，扩展指令表达
- **多模态反馈**：状态高亮、音频播报（骨传导耳机）、Spatial AR (HoloLens)
- **真机对接**：UR5e RTDE 接口，从仿真落地真实产线
