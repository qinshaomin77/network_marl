# Hierarchical Traffic Signal Control with Deep Reinforcement Learning

基于分层深度强化学习的城市交通信号控制系统，在 5×5 网格路网上实现排放感知的自适应信号配时优化。

---

## 项目概述

本项目采用**双层分层架构**，将交通信号控制问题分解为：

- **下层（Local）**：每个交叉口独立的信号配时决策，使用 **A2C（Advantage Actor-Critic）** 算法，通过 GAT（Graph Attention Network）编码车道级特征与邻居策略信息。
- **上层（Network）**：全网协调层，使用 **PPO（Proximal Policy Optimization）** 算法，通过边级别（edge-level）GAT 学习动态调整各交叉口的排放-效率权重分配。

系统在 SUMO 微观交通仿真平台上运行，支持基于排放因子表的逐车排放估算，兼顾交通效率与环境指标。

---

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    上层 (Upper PPO)                    │
│  边级 GAT 编码器 → 全网权重分配 [w_em, w_eff] per TLS  │
│  每 K=36 步决策一次                                    │
└────────────────────────┬─────────────────────────────┘
                         │  权重矩阵 [N_tls, 2]
┌────────────────────────▼─────────────────────────────┐
│                    下层 (Lower A2C)                    │
│  车道级 GAT + LSTM + 邻居策略编码 → 各路口相位选择      │
│  每 ΔT=5s 决策一次                                    │
└────────────────────────┬─────────────────────────────┘
                         │  动作（相位切换）
┌────────────────────────▼─────────────────────────────┐
│                 SUMO 仿真环境 (5×5 Grid)               │
│  25 个信号控制交叉口 · 3 车道 · 混合车流（轿车+货车）   │
└──────────────────────────────────────────────────────┘
```

---

## 文件结构

| 文件 | 说明 |
|------|------|
| `main.py` | 训练主入口，包含完整的训练循环、checkpoint 管理与 TensorBoard 日志 |
| `config.py` | 全局超参数配置（网络结构、学习率、奖励系数、仿真参数等） |
| `env.py` | Gymnasium 风格的 SUMO 交通仿真环境，封装上下层观测构建与奖励计算 |
| `model.py` | 统一模型接口，整合上下层 agent 供 `main.py` 调用 |
| `node_model.py` | 下层 A2C agent：局部 GAT 编码器、邻居策略编码器、Actor/Critic 网络 |
| `network_model.py` | 上层 PPO agent：边级 GAT 编码器、协调器网络、价值网络 |
| `model_layers.py` | 共享神经网络模块（RelationGATLayer、LSTMCore、MLPBlock） |
| `buffer.py` | 经验回放缓冲区：A2CBuffer（下层 n-step）与 PPOBuffer（上层 GAE） |
| `utils.py` | SUMO 命令构建工具与单交叉口状态采集器（Node 类） |
| `xml_mapping.py` | SUMO 路网 XML 静态解析，构建车道映射、相位信息、邻接图与边级图 |
| `emission_lookup.py` | 排放因子查询器，基于 CSV 排放因子表按车型-速度-加速度查询排放 |
| `logger.py` | CSV 批量日志记录器，涵盖网络级/路口级/边级/训练指标 |
| `bulid_network_grid.py` | 5×5 网格路网与路由文件生成脚本（SUMO 场景构建） |

---

## 核心设计细节

### 下层观测空间

每个交叉口的观测包含：

- **车道级节点特征** `[N_lanes, 4]`：wave（车辆数归一化）、speed_lack（速度不足率）、truck_ratio（货车占比）、queue_ratio（排队占比）
- **局部图结构**：同组邻接矩阵 `A_same` 与异组邻接矩阵 `A_diff`（基于信号相位分组）
- **邻居策略** `[4, A_max]`：东南西北四个邻居的动作概率分布
- **动作掩码**：有效相位的 mask

### 上层观测空间

全网边级别观测：

- **边特征** `[E, 3]`：车辆密度归一化、货车比例、排队归一化
- **边级图结构**：上游邻接 `A_edge_up` 与下游邻接 `A_edge_down`
- **边-路口映射矩阵** `edge_to_tls [N_tls, E]`

### 奖励设计

- **下层奖励**：加权混合排放成本与效率成本（排队+速度不足），包含空间平滑的邻居奖励项
- **上层奖励**：全网归一化排队总和 + 排放总和 + 热点惩罚，三项加权

### 排放计算

通过 `emission_lookup.py` 加载外部排放因子 CSV 表，按车型（sedan/truck）、速度、加速度查询 NOx 等污染物的瞬时排放因子（g/s），每个仿真步累计排放量。

---

## 环境依赖

- Python ≥ 3.8
- PyTorch
- NumPy / Pandas
- Gymnasium
- SUMO（需安装并配置 `SUMO_HOME` 环境变量）
- TensorBoard（可选）

---

## 快速开始

### 1. 生成路网文件

```bash
python bulid_network_grid.py
```

此脚本将在 `./network_grid/` 下生成 5×5 网格路网所需的全部 SUMO 配置文件（`.net.xml`、`.rou.xml`、`.add.xml` 等），并自动调用 `netconvert`。

### 2. 准备排放因子表

确保 `./emission_data/emission_factor_table.csv` 存在，包含以下字段：

```
vehicle_type, pollutant, speed_kmh, accel_ms2, EmissionFactor_gs
```

### 3. 启动训练

```bash
python main.py --seed 42 --episodes 80 --tensorboard
```

主要命令行参数：

| 参数 | 说明 |
|------|------|
| `--seed` | 全局随机种子（默认 42） |
| `--episodes` | 训练回合数（默认读取 config 中的 80） |
| `--gui` | 启用 SUMO GUI 可视化 |
| `--tensorboard` | 启用 TensorBoard 日志 |
| `--resume <path>` | 从 checkpoint 恢复训练 |
| `--deterministic-upper` | 上层采用确定性策略 |
| `--deterministic-lower` | 下层采用确定性策略 |

### 4. 查看结果

训练产出保存在 `./results/<run_name>/` 下：

- `logs/` — 各类 CSV 指标日志与 TensorBoard 事件文件
- `models/` — 最佳模型与周期性 checkpoint
- `simulations/` — SUMO 输出（tripinfo、summary 等）

---

## 关键超参数（config.py）

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 仿真 | `DELTA_T` | 5 | 下层决策间隔（秒） |
| 仿真 | `YELLOW_TIME` | 3 | 黄灯持续时间（秒） |
| 仿真 | `SIMULATION_STEPS` | 3600 | 每回合仿真步数 |
| 下层 | `LOCAL_GAT_HIDDEN` | 64 | GAT 隐藏维度 |
| 下层 | `LSTM_DIM` | 64 | LSTM 隐状态维度 |
| 下层 | `N_STEP` | 12 | n-step return 窗口 |
| 下层 | `LOWER_LR_ACTOR` | 1.5e-4 | Actor 学习率 |
| 上层 | `UPPER_K` | 36 | 上层决策间隔（下层步数） |
| 上层 | `UPPER_CLIP_RATIO` | 0.2 | PPO 裁剪比 |
| 上层 | `UPPER_ROLLOUT_SIZE` | 20 | 上层 rollout 长度 |
| 奖励 | `UPPER_BETA_QUEUE` | 0.40 | 上层排队惩罚权重 |
| 奖励 | `UPPER_BETA_EMISSION` | 0.40 | 上层排放惩罚权重 |
| 奖励 | `UPPER_BETA_HOTSPOT` | 0.20 | 上层热点惩罚权重 |

---

## 日志输出说明

训练过程中会生成以下 CSV 日志文件：

| 文件 | 粒度 | 内容 |
|------|------|------|
| `network_step_metrics.csv` | 每步/全网 | 全网奖励、排放/排队总和、权重分布 |
| `tls_step_metrics.csv` | 每步/每路口 | 各路口的观测统计、动作、奖励、权重 |
| `edge_step_metrics.csv` | 每步/每边 | 边级车辆数、排队、排放、热点标记 |
| `upper_step_metrics.csv` | 每上层步 | 上层决策前后状态快照、权重统计 |
| `training_metrics_lower.csv` | 每次更新 | 下层 policy/value/entropy loss、梯度范数 |
| `training_metrics_upper.csv` | 每次更新 | 上层 PPO 训练指标 |
| `episode_summary.csv` | 每回合 | 回合级汇总统计 |
| `tls_node_step_features_raw.csv` | 每步/每车道 | 原始车道级特征（wave、queue、speed 等） |
