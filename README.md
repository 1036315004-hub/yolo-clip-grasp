# VLA-Arm-Project: Semantic Grasp Control System for Robotic Arm 

[English](#english-version) | [中文说明](#中文说明)
---

## English Version
### Project Summary: Semantic Grasp Control System for Robotic Arm Based on VLA Framework

#### 1. Core Project Capabilities
This project is developed based on the Elite EC63 SDK, building a full closed-loop system from natural language instructions to hardware low-level execution.
- **Algorithm Deployment on Physical Robot**: Successfully deployed on the Elite EC63 robotic arm and verified with more than 10 sets of basic grasp tasks in continuous tests.
- **Hardware-Software Coordination**: With the highly abstracted *SimulationAdapter* design, it enables one-click switch between the simulation environment (PyBullet) and physical robot control, realizing high reusability of algorithms and kinematics modules in both environments.
- **Performance Metrics**: The recognition and grasp success rate reaches 90% in the simulation environment; the closed-loop of *instruction understanding → target localization → motion control* is successfully implemented on the physical robot.

#### 2. Core Technical Framework (VLA)
The project adopts a deep integration scheme of **Vision-Language-Action (VLA)**:
- **Perception (V/L)**:
  Leverages YOLOv8 for real-time extraction of multiple candidate targets (Region Proposals) from environmental images for cross-modal alignment (CLIP w/ ViT-B-32). ViT-B-32 is loaded as the visual backbone network, and the CLIP framework maps candidate image patches and natural language instructions to a unified feature space, then calculates similarity rankings to achieve **zero-shot recognition** of target objects.
- **Execution (A)**:
  The EC control class is encapsulated based on the Elite SDK. After the core algorithm generates the target pose, the high-precision motion execution of the robotic arm is realized through inverse kinematics (IK) solution and Hand-Eye Matrix mapping.

#### 3. Repository Directory Structure
<pre>
```
yolo-clip-grasp/
└── VLA_arm_project/
    ├── src/
    │   ├── perception/       # Core of Vision and Semantic Understanding
    │   │   ├── VLM_YOLO_CLIP.py  # VLA core framework, integrating detection and semantic ranking
    │   │   ├── detector.py       # YOLOv8-based object detection encapsulation
    │   │   ├── ranker.py         # CLIP-based cross-modal semantic matcher
    │   │   └── camera_utils.py   # Camera pose calculation and coordinate transformation tools
    │   ├── robot/            # Robotic Control Layer
    │   │   ├── sim_adapter.py    # Simulation/Physical compatibility layer, supporting cross-end instruction conversion
    │   │   └── elite_sdk/        # Elite official SDK with low-level drivers (e.g., _ec.py, _move.py)
    │   └── utils/
    │       └── hand_eye_calib.py # Hand-eye calibration tool, core step for physical robot deployment
    ├── scripts/              # Business Logic and Test Scripts
    │   ├── run_data_recorder.py  # PyBullet-based simulation data recorder
    │   ├── collect_data.py   # Closed-loop grasp strategy with multi-stage scanning and fine alignment
    │   └── test_vla_sim.py   # Entrance for full-process verification of semantic grasping
    ├── assets/               # Robotic arm URDF models and simulation environment resources
    └── checkpoints/          # Pretrained weight files for YOLO and CLIP
   ```
</pre>
#### 4. Outcome Summary
- **Engineering Capability**: Achieved real machine deployment from 0 to 1. The code design adopted the strategy pattern (controlled by the `sim` parameter), enabling seamless switching between the perception module and the kinematics module in simulation and hardware.
- **Complex System Integration**: Independently completed hand-eye calibration, coordinate system conversion, and SDK secondary development, optimizing parameter design and reducing common motion jitter and perception error compensation issues during real machine deployment.

---

## 中文说明
### 项目总结：基于 VLA 框架的机械臂语义抓取控制系统

#### 1. 项目核心能力
本项目基于 Elite（艾利特）EC63 SDK 开发，构建了从自然语言指令到硬件底层执行的全链路闭环。
- 算法实机部署：已成功在 EC63 机械臂上完成部署，并连续通过 10 组以上基础抓取任务验证。
- 软硬件协同：通过高度抽象的 SimulationAdapter 设计，实现了仿真环境（PyBullet）与实机控制的一键切换，算法与运动学模块在两端高度复用。
- 性能指标：仿真环境识别及抓取成功率达 90%；实机环境成功打通“指令理解→目标定位→运动控制”闭环。

#### 2. 核心技术框架 (VLA)
项目采用了 “视觉-语言-动作” (Vision-Language-Action) 深度集成方案：
- 感知（V/L）：(YOLOv8)：实时从环境图像中提取多个候选目标（Region Proposals），跨模态对齐 (CLIP w/ ViT-B-32)。加载 ViT-B-32 作为视觉骨干网络，通过 CLIP 框架将候选图块与自然语言指令映射到统一特征空间，计算相似度排名，从而实现对目标物体的零样本识别。
- 执行（A）：基于 Elite SDK 封装了 EC 控制类。核心算法生成目标位姿后，通过逆运动学（IK）求解和手眼标定矩阵（Hand-Eye Matrix）映射，实现机械臂的高精度动作执行。

#### 3. 仓库目录说明
<pre>
```
   yolo-clip-grasp/
└── VLA_arm_project/
    ├── src/
    │   ├── perception/       # 视觉与语义理解核心
    │   │   ├── VLM_YOLO_CLIP.py  # VLA 核心框架，集成检测与语义排名
    │   │   ├── detector.py       # 基于 YOLOv8 的物体检测封装
    │   │   ├── ranker.py         # 基于 CLIP 的跨模态语义匹配器
    │   │   └── camera_utils.py   # 相机位姿计算与坐标转换工具
    │   ├── robot/            # 机器人控制层
    │   │   ├── sim_adapter.py    # 仿真/实机兼容层，支持指令跨端转换
    │   │   └── elite_sdk/        # 艾利特官方 SDK，包含 _ec.py, _move.py 等底层驱动
    │   └── utils/
    │       └── hand_eye_calib.py # 手眼标定工具，实机部署的核心环节
    ├── scripts/              # 业务逻辑与测试脚本
    │   ├── run_data_recorder.py   # 基于PyBullet的仿真数据记录器
    │   ├── collect_data.py   # 包含多阶段扫描、精细对准的闭环抓取策略
    │   └── test_vla_sim.py   # 语义抓取全流程验证入口
    ├── assets/               # 机械臂 URDF 模型与仿真环境资源
    └── checkpoints/          # YOLO 与 CLIP 的预训练权重文件
  ```
</pre>
            
  
#### 4. 成果总结
- **工程落地能力**：完成了从0到1的实机部署，代码采用策略模式（由`sim`参数控制），实现了感知模块与运动学模块在仿真和硬件端的无缝切换。
- **复杂系统集成**：独立完成手眼标定、坐标系转换及SDK二次开发，优化了参数设计，减少了实机部署中常见的运动抖动和感知误差补偿问题。
