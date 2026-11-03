# Qwenputer on Apple Silicon

[`Qwen3.5-4B-Qwenputer`](https://github.com/divagr18/qwenputer) 把一颗 RV32 CPU 编译进了 Qwen3.5-4B 的前 11 层权重：Qwen 写出 `<tool_call>` 后，固件在这些层里运行，算出 `calc` 的结果并逐字符回显，再由 Qwen 写出回答。本目录是它在 Mac mini M6（24 GB）上的适配。

**现状（2026-11-03）**：
- 芯片层以**神经网络本身**运行（checkpoint 里的 11 层 Gated DeltaNet + SwiGLU，MLX 上的融合 Metal kernel），与电路**逐位一致**：171,496 步 teacher-forced 0 mismatch，自由解码整串 token 与参考机一致。
- 芯片 token 约 **3,300 tok/s**（原生 HF 101 tok/s，33 倍）；Qwen 文本 token 约 15 tok/s，受内存带宽限制，4-bit 量化后约 41 tok/s。
- 题 1（`123456789 * 987654321`）端到端 8.7 s，最初版本 17.3 s；`--qwen-bits 4` 为 3.7 s。
- 改动都还没有提交。

## 文档

- [docs/technical.md](docs/technical.md)：技术文档。模型结构、解码流程与提示词、芯片层的实现（压缩、状态、kernel、free run）、Qwen 后端、优化阶梯、验证和性能数据。
- [docs/pitfalls.md](docs/pitfalls.md)：踩坑实录。ANE 的 FP16 溢出、上游的 CUDA 绑定、内存、Metal kernel、性能归因、验证和网页中遇到的问题。

## 获取与安装

```bash
git clone --recurse-submodules https://github.com/Cbphcr/qwenputer-mac
cd qwenputer-mac
git -C upstream apply ../patches/upstream-mac.patch   # 上游的两处 Mac 小修（pitfalls 5、6）
```

上游以 submodule 固定在 `divagr18/qwenputer@bec1a0c`。Python 3.12 环境需要 `torch`（MPS）、`transformers`、`mlx`、`mlx-lm`、`safetensors`、`pytest`，并 `pip install -e upstream/experimental/riscv` 安装 `chipformer`（版本见下方「环境与目录」）。模型权重另行下载，不在仓库里。

## 运行

```bash
cd qwenputer-mac
M=/path/to/Qwen3.5-4B-Qwenputer   # 模型权重目录

# 对话（默认：芯片层 = 神经网络 mlx-freerun，Qwen = mlx-lm bf16）；--check 用参考机和 IR 重放校验
.venv/bin/python -m macport.chat $M --check
.venv/bin/python -m macport.chat $M --ask "Use calc: what is (2**100 - 12345678901234567890) // 97 + 84/18?" --check
.venv/bin/python -m macport.chat $M --qwen-bits 4 --check   # 只量化 Qwen 层

# 对照：只跑电路 / 两者每步比对
.venv/bin/python -m macport.chat $M --chip circuit --check
.venv/bin/python -m macport.chat $M --chip verify --check

# 网页（打开 http://localhost:8765）
.venv/bin/python -m macport.web $M

# 芯片层基准：teacher-forced 每步与电路比对 / 自由解码与 trace 比对
.venv/bin/python -m macport.bench_chip $M --impl mlx-inplace
.venv/bin/python -m macport.bench_chip $M --impl mlx-freerun --free --steps 100000

# 上游测试
.venv/bin/pytest upstream/experimental/riscv/tests -q
```

主要选项：`--impl`（芯片层实现：`hf` / `sparse` / `lean` / `compiled` / `mlx` / `mlx-fused` / `mlx-inplace` / `mlx-freerun`，默认最后一个），`--backend mlx|torch`（Qwen 层后端），`--qwen-bits 8|4`，`--chip neural|circuit|verify`。

## 环境与目录

- 硬件：Mac mini（Apple M6，12 核 GPU，Metal 4，24 GB 统一内存，macOS 27.0.1）
- 模型：`Qwen3.5-4B-Qwenputer`（8 个 safetensors 分片，18.8 GB，BF16）
- `.venv/`：Python 3.12（解释器在 `.uv-python/`）。`torch 2.15.0`（MPS）、`transformers 5.22.1`、`mlx 0.32.4`、`mlx-lm 0.32.2`、`coremltools 9.0`、`safetensors`、`pytest`，以及可编辑安装的 `chipformer`

| 路径 | 内容 |
| :--- | :--- |
| `upstream/` | 上游仓库克隆（核心在 `experimental/riscv/chipformer/`）；改了两处小地方，见 pitfalls 5、6 |
| `macport/chat.py` | 主程序 `MacChat` |
| `macport/mlx_chip.py`、`mlx_kernels.py` | 芯片层的 MLX 实现与 Metal kernel |
| `macport/mlx_qwen.py` | Qwen 32 层的 MLX 加载 |
| `macport/bench_chip.py` | 芯片层基准 |
| `macport/web.py`、`web.html` | 本地网页 |
| `macport/tools/inspect_ckpt.py` | 权重扫描脚本 → `reports/ckpt_report.json` |
| `docs/` | 技术文档与踩坑实录 |
