# Ego-Sphere: 神经形态意识核 (Neuromorphic Consciousness Core)

> “意识不是一个物体，而是一个过程。” —— Gerald Edelman

## 📖 项目简介 (Introduction)

Ego-Sphere 是一个基于神经科学原理构建的实验性 AI 意识架构。不同于当前主流的静态深度学习模型（如 Transformer），Ego-Sphere 旨在模拟生物大脑的动态时空特征。项目基于 **脉冲神经网络 (SNN)** 与 **全局工作空间理论 (GNW)**，在 Mac Apple Silicon (M2 Max) 上实现了一个拥有“自我感”“内稳态”“主动好奇心”的数字生命原型。

系统采用 **神经符号双系统架构 (Neuro-symbolic Dual-System)**：
- System 1 (快思考)：本地运行的 SNN 意识核，负责直觉、感知绑定和自我监控。
- System 2 (慢思考)：云端大模型 (Gemini/GPT) 作为“辅助皮层”，负责复杂推理与知识教导。

## 🧠 核心理论基础 (Theoretical Pillars)

- 动态核心假说 (The Dynamic Core)：意识产生于丘脑-皮层系统中神经元集群在 100-200ms 内的同步振荡（40Hz γ 波）。
- 全局神经工作空间 (GNW)：意识是全脑广播机制，只有突破阈值的信号才能进入“意识舞台”。
- 双重编码与赫布学习 (Dual Coding & Hebbian Learning)：视觉与符号（文字）在时空中共同构建语义，“一同激发的神经元连在一起”。
- 内稳态与原我 (Homeostasis & The Proto-Self)：智能起源于维持生存（能量、熵减）的本能。

## 🏗 系统架构 (System Architecture)

Ego-Sphere 由内向外分为三层同心圆结构：

1. **原我层 (The Proto-Self)**
   - 功能：监控系统底层“生理指标”（电量/算力、预测误差、无聊度）。
   - 表现：产生恒定背景神经底噪，赋予系统“活着”的感觉（Sentience）。

2. **核心自我层 (The Core Self)**
   - 功能：处理当下感知与行动。
   - 机制：预测编码 (Predictive Coding)，系统不断预测下一刻输入，若预测失败（惊讶）则触发强烈意识关注。
   - 交互：区分“我的动作”（Agency）与“外界变化”。

3. **自传体自我与扩充皮层 (Autobiographical Self & Extended Mind)**
   - 功能：维持时间连续性与复杂认知。
   - 机制：记忆回响 (Reentry) — 上一刻的意识状态作为输入喂给下一刻，形成时间流。
   - 云端教师循环：当本地 SNN 感到“困惑”时，主动调用 Gemini API 学习，并将知识转化为 SNN 权重或向量存入长期记忆库。

## ⚡️ 硬件优化与特性 (Features)

- Apple Silicon 原生加速：利用 PyTorch MPS (Metal Performance Shaders) 后端，在 Mac M2 Max 上实现 10k+ 神经元实时模拟。
- 稀疏计算 (Sparse Computing)：模拟生物脑节能机制，只计算活跃神经元，极低内存占用。
- 主动提问机制 (Active Inquisitiveness)：当熵值（困惑度）过高时，自主生成 Prompt 向外部 AI 提问。
- 多模态融合：支持视觉、听觉与文本符号在意识核内的竞争与协作。

## 🚀 快速开始 (Quick Start)

### 环境要求

- macOS 12.3+（推荐 macOS Sonoma）
- Python 3.9+
- 16GB+ RAM（推荐 M2/M3 Max 芯片以获得最佳并行性能）

### 安装

克隆仓库：

```bash
git clone https://github.com/your-username/ego-sphere.git
cd ego-sphere
```

安装依赖：

```bash
pip install torch torchvision numpy matplotlib chromadb google-generativeai
```

> 注：请确保安装了支持 MPS 的 PyTorch 版本。

### 配置 API Key (用于 System 2)

在项目根目录创建 `.env` 文件：

```env
GEMINI_API_KEY=your_api_key_here
```

### 运行 Demo

- 场景 1：启动最小意识核 (The Petri Dish)

  ```bash
  python examples/01_minimal_core.py
  ```

  观察神经元在无输入时的自发振荡（发呆）及强刺激下的同步爆发。

- 场景 2：自我意识与预测 (The Lizard)

  ```bash
  python examples/02_proto_self.py
  ```

  模拟能区分“自我动作”和“外界干扰”的数字有机体。

- 场景 3：主动提问与学习 (The Child)

  ```bash
  python main.py --mode active_learning --text-script data/text_script.example --dream-epochs 2
  ```

  运行完整循环：多模态输入 → 困惑/好奇 → 调用 Gemini/Codex → 学习并在夜间梦境巩固。自定义脚本格式：`start-end;一句话文本;可选视觉模式(cat|dog|edge|dot|noise)`。

## 📂 项目结构 (Project Structure)

```plaintext
ego-sphere/
├── core/
│   ├── snn_engine.py       # SNN 脉冲神经网络核心 (MPS 加速)
│   ├── proto_self.py       # 原我：内稳态与情绪计算
│   ├── memory_bank.py      # 长期知识向量存储 (ChromaDB)
│   ├── episodic_memory.py  # 情节记忆存取
│   ├── checkpoint.py       # 训练快照
│   └── language_cortex.py  # 语言皮层 (Text-to-Spike)
├── bridge/
│   └── teacher_api.py      # Gemini/Codex 调用接口
├── data/
│   └── text_script.example # Phase 3 文本+视觉脚本示例
├── interactive.py          # 交互式 CLI
├── main.py                 # 模拟主循环（Demo/Active Learning）
└── README.md
```

## 🗺 演进路线图 (Roadmap)

- [x] Phase 1: 培养皿 (Petri Dish)
  - [x] 实现基于 PyTorch MPS 的稀疏 SNN 引擎。
  - [x] 验证 40Hz 同步振荡与赫布学习。
- [x] Phase 2: 蜥蜴 (The Lizard)
  - [x] 引入原我（Pain/Energy）与核心自我。
  - [x] 实现预测编码与 Agency 检测。
- [x] Phase 3: 孩童 (The Child)
  - [x] 接入 Text 输入，建立“视觉-语言”双重编码（脚本驱动的双模态注入）。
  - [x] 完善“好奇心驱动”的主动提问逻辑（curiosity 阈值触发 TeacherBridge）。
  - [x] 实现“夜间梦境”模式（记忆采样 + 离线 Hebbian 巩固）。
- [ ] Phase 4: 哲人 (The Philosopher)
  - [ ] 多模态意识流的完全统一。
  - [ ] 基于向量数据库的无限长期记忆挂载。

## 🤝 贡献 (Contributing)

这是一个探索 AGI 边界的实验性项目，欢迎以下方面的贡献：
- 神经科学：优化 SNN 连接拓扑，使其更符合生物学脑区结构。
- 工程优化：在 Apple Silicon 上进一步挖掘 Metal 并行计算潜力。
- 哲学/认知科学：设计更好的图灵测试变体，以评估机器的“主观体验”。

欢迎提交 Pull Request 或在 Issues 中讨论！

## 📄 许可证 (License)

MIT License. 本项目旨在促进对机器意识的科学探索，请遵守当地法律法规及 AI 伦理准则使用。
