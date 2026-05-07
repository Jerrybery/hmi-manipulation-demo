# hmi-manipulation-demo

单台 macOS 笔记本上跑通的"机械臂仿真 + 摄像头手势仲裁"最小闭环 demo：UR5e 在 MuJoCo 物理引擎里画圆，操作员对着摄像头做"开掌正面"手势即可让仿真减速冻结、再回到 home 姿态后重新启动；Esc 键作为独立急停。

用于验证"预测性意图可视化 + 人在回路手势仲裁"的整体软件架构（仿真线程 / 渲染 / 摄像头线程 / 手势 → 控制信号），为后续 Ghost Trajectory、Dynamic Safety Bubble、Authority Bar 等可视化层迭代打底。

---

## 演示功能（MVP）

| 功能 | 说明 |
|---|---|
| 圆周轨迹跟踪 | UR5e 末端在桌面上方画圆（默认半径 0.15 m，圆心 (0.5, 0, 0.3)，~10 s 一圈） |
| 差分 IK | DLS（damped least squares）雅可比伪逆，仅约束位置、姿态固定朝下 |
| 开掌手势暂停 | MediaPipe Hand Landmarker 检测开掌 + 掌心朝镜头；5 帧迟滞防抖（~165 ms @ 30 fps） |
| 回零恢复 | 放下手后关节空间线性插值 1.5 s 回 home keyframe，然后重置轨迹相位重新画圆；可配置为"原地继续" |
| Esc 急停 | 与手势独立的 OR 通道，按一次冻结、再按一次解冻 |
| 状态条 | RUNNING / PAUSED (gesture) / RETURN_HOME / E-STOP / NO CAMERA 五种颜色编码 |
| 无摄像头降级 | 摄像头打不开或中途失联 → 仿真继续，状态条提示 |

**不在 MVP 范围**（后续迭代）：Ghost Trajectory（半透明预测臂）、Dynamic Safety Bubble、Authority Bar 平滑过渡、多手 / 多人场景。

---

## 技术栈

| 类别 | 选型 | 用途 |
|---|---|---|
| Python 运行时 | Python 3.12（`uv` 管理） | MediaPipe 在 macOS arm64 PyPI wheel 当前最高仅到 3.12 |
| 物理仿真 | MuJoCo ≥ 3.2 | 高保真动力学，自带 `Renderer` 离屏渲染 |
| 机械臂模型 | mujoco_menagerie / `universal_robots_ur5e` | 6 DOF UR5e MJCF（含 `attachment_site` + `home` keyframe） |
| 手势识别 | MediaPipe Tasks Hand Landmarker（`mediapipe.tasks.python.vision.HandLandmarker`） | 21 点 3D 关键点 + 左右手分类；首次运行自动下载 `hand_landmarker.task` 到 `~/.cache/hmi-demo/` |
| 摄像头采集 | OpenCV (`cv2.VideoCapture`) | macOS AVFoundation 默认后端 |
| GUI | PyQt6 | QMainWindow + QThread + Qt signal 跨线程通信 |
| 数值 | NumPy < 2 | mediapipe 0.10.x 仍要求 numpy 1.x |
| 配置 | PyYAML | `configs/demo.yaml` |
| 测试 | pytest | 19 个单元测试覆盖 config / world / trajectory / IK / gesture |

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
│                                                 │ q_target   │
│                                          ┌──────▼─────────┐  │
│                                          │  World.step()  │  │
│                                          │  (mj_step ×5)  │  │
│                                          └──────┬─────────┘  │
│                                          ┌──────▼─────────┐  │
│                                          │ Renderer.grab  │  │
│                                          │ (offscreen)    │  │
│                                          └──────┬─────────┘  │
│                                                 │ QImage     │
│  ┌──────────────────────┐  cam QImage    ┌──────▼─────────┐  │
│  │ HMIWindow (PyQt6)    │ <───────────── │ QLabel.sim     │  │
│  │  + MotionState 状态机│                │ QLabel.cam     │  │
│  │  + Esc keyPressEvent │                │ QStatusBar     │  │
│  └──────────▲───────────┘                └────────────────┘  │
│             │ Qt signal: gestureUpdated(bool, QImage)        │
└─────────────┼────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────┐
│                 VisionThread (QThread, 30Hz)                 │
│                                                              │
│  cv2.VideoCapture.read()                                     │
│   → cv2.flip (selfie 镜像)                                   │
│   → MediaPipe HandLandmarker.detect_for_video                │
│   → is_open_hand + is_palm_facing_camera                     │
│   → HysteresisFilter (5 帧迟滞)                              │
│   → emit gestureUpdated(stable_bool, annotated_QImage)       │
└──────────────────────────────────────────────────────────────┘
```

**关键不变量**

- `mjData` 只被 Qt 主线程读写（无锁）
- VisionThread 只产数据，不直接写 `mjData`；通过 Qt signal 把 `(raised, QImage)` 投递到主线程 event queue
- 退出顺序：`closeEvent` → `timer.stop()` → `vision_thread.request_stop()` + `wait(2000)` → `renderer.close()` → `super().closeEvent()`
- 每个 cleanup 步骤都用 `hasattr` 守护，允许部分初始化失败时也能干净关闭

---

## 状态机

```
            举手 latch                   放手 latch (mode=return_home)
RUNNING ───────────────> FROZEN ───────────────────────────> RETURN_HOME
   ▲                       │                                       │
   │ 1.5s 关节空间插值到达  │ 放手 latch (mode=resume_in_place)     │
   │ home + reset t=0       └─────────────────────────────────┐    │
   └──────────────────────────────────────────────────────────┴────┘
```

- **RUNNING**：`trajectory.tick(dt)` 推进圆周相位，`diff_ik_dls` 跟踪
- **FROZEN**：`trajectory.tick(0.0)`，IK 目标静止，机械臂自然停在原位
- **RETURN_HOME**：`q_target = lerp(q_at_release, q_home, s)`，`s` 在 `recovery.return_duration_s` 秒内 0→1；到达 1.0 后 `trajectory.reset()` 并切回 RUNNING
- 冻结源：`effective_frozen = gesture_frozen OR estop_frozen`
  - `gesture_frozen` 由 VisionThread signal 写
  - `estop_frozen` 由 Esc `keyPressEvent` 切换
  - Esc 与手势完全独立，互不清除

---

## 项目结构

```
hmi-manipulation-demo/
├── pyproject.toml             # uv 项目，依赖锁定
├── .python-version            # 3.12
├── configs/
│   └── demo.yaml              # 全部可调参数
├── assets/
│   └── mujoco_menagerie/      # shallow clone（gitignored，~2 GB）
├── tests/                     # 19 个 pytest 单元测试
└── src/hmi_demo/
    ├── __main__.py            # python -m hmi_demo 入口
    ├── app.py                 # QApplication 装配 + 顶层错误对话框
    ├── config.py              # 7 个 frozen dataclass + load_config
    ├── sim/
    │   ├── world.py           # MjModel + MjData 包装，home keyframe
    │   ├── trajectory.py      # CircleTrajectory（纯 NumPy）
    │   └── ik.py              # diff_ik_dls 纯函数（DLS 雅可比）
    ├── vision/
    │   └── hand_thread.py     # VisionThread (cv2 + HandLandmarker + Qt signal)
    ├── ui/
    │   ├── render.py          # MujocoRenderer（numpy → QImage）
    │   └── hmi_window.py      # QMainWindow + MotionState + 100Hz QTimer
    └── utils/
        └── gesture.py         # is_palm_facing_camera + is_open_hand + HysteresisFilter
```

---

## 配置（`configs/demo.yaml`）

```yaml
sim:
  mjcf: assets/mujoco_menagerie/universal_robots_ur5e/scene.xml
  ee_site: attachment_site         # 末端 site 名（必须存在于 MJCF）
  home_keyframe: home              # MJCF 中 keyframe 名，q_home 由此读取
  control_hz: 100                  # 外环控制频率 → QTimer 周期 10 ms
  substeps: 5                      # 每外环跑 5 次 mj_step（dt=0.002）

trajectory:
  center: [0.5, 0.0, 0.3]
  radius: 0.15
  omega: 0.6                       # rad/s, 2π/0.6 ≈ 10.5 s 一圈

ik:
  damping: 0.05                    # DLS λ
  kp: 0.5                          # 比例增益（>1 会引起振荡，调高需谨慎）

camera:
  device: 0
  width: 640
  height: 480
  fps: 30

gesture:
  hold_frames: 5                   # 迟滞防抖 latch/release 帧数
  enable_palm_check: true          # 关掉则任意"开掌"都触发，不要求掌心朝向

recovery:
  mode: return_home                # 或 resume_in_place
  return_duration_s: 1.5

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

uv sync                              # 解析依赖、装到 .venv
uv run python -m hmi_demo            # 启动 demo
```

**首次启动**：
- 自动下载 `hand_landmarker.task`（~7 MB）到 `~/.cache/hmi-demo/`
- macOS 会请求摄像头权限。如未弹出，手动到 System Settings → Privacy & Security → Camera 给当前终端授权后重启 demo

**手动验收清单**：
1. 窗口 5 s 内显示，左侧 UR5e 平滑画圆，右侧摄像头预览 + 关键点叠加
2. 对着镜头开掌正面 → 200 ms 内状态条变黄 `PAUSED (gesture)`，机械臂停下
3. 放下手 → 状态条变蓝 `RETURN_HOME`，1.5 s 内回 home，再变绿重新画圆
4. Esc → 状态条变红 `E-STOP`，再按一次解除
5. 关窗口干净退出，`ps aux | grep hmi_demo` 无残留

如果第 2 步不工作（开掌不触发暂停），先把 `configs/demo.yaml` 的 `gesture.enable_palm_check` 改成 `false`，绕过掌心朝向检测。

---

## 测试

```bash
uv run pytest                        # 19 个单元测试
```

| 测试文件 | 数量 | 范围 |
|---|---|---|
| `test_config.py` | 2 | YAML 加载 + recovery.mode 校验 |
| `test_world.py` | 2 | MJCF 加载 + 100 步无 NaN + 未知 site 报错 |
| `test_trajectory.py` | 5 | 起点 / 周期回归 / `tick(0)` 不前进 / 恢复推进 / reset |
| `test_ik.py` | 2 | 收敛到 < 1 mm / 关节限位裁剪 |
| `test_gesture.py` | 8 | 开掌 + 掌心朝向 + 5 帧迟滞 |

视觉与 GUI 流程不做自动测试（cv2 + Qt 集成测试投入产出比低）；仅做 import smoke 和手动验收。

---

## 已知限制

- 摄像头分辨率/帧率只支持设备能商定的近似值（cap.set 失败时仅打印 warning，不中断）
- IK 仅约束位置 + 固定姿态朝下，不处理姿态目标
- MediaPipe handedness 在 `cv2.flip` 后是图像相对的，VisionThread 内部已做反转还原；如真实摄像头表现仍异常，关 `enable_palm_check` 即可
- `mediapipe>=0.10.21,<0.11` 锁版本依赖：legacy `mp.solutions` API 已在 0.10.x 末期被移除，本项目用新的 `mediapipe.tasks.python.vision.HandLandmarker`，模型文件首次运行自动下载

---

## 后续迭代

- **Ghost Trajectory**：场景内追加半透明 `mjvGeom` 渲染未来 0.5 s 关节轨迹，按速度/距离染色
- **Dynamic Safety Bubble**：以末端为中心、半径 = `f(|v_ee|)` 的半透明球体
- **Authority Bar**：HMI 顶部条带，控制权重在 AI ↔ Human 间二阶低通过渡
- **多模态反馈**：颜色高亮、状态文字滚动、摄像头框选当前主导手
