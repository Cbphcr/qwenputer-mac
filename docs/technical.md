# Qwenputer Mac 适配：技术文档

本文说明 `Qwen3.5-4B-Qwenputer` 在 Apple Silicon（Mac mini M6 / 24 GB）上的实现：模型结构、芯片层如何作为神经网络本身运行、各级优化、精确性验证与性能数据。过程中遇到的问题与解决办法见 [pitfalls.md](pitfalls.md)。

## 1. 模型结构

`Qwen3.5-4B-Qwenputer` 把一颗 **32 位 RISC-V CPU（RV32IM + Zba + bignum）** 编译进了 Transformer 权重：

| 属性 | 原版 Qwen3.5-4B | Qwenputer（合并后） | 说明 |
| :--- | :--- | :--- | :--- |
| `num_hidden_layers` | 32 | **43** = 11 芯片层 + 32 Qwen 层 | 芯片层都是 `linear_attention`（Gated DeltaNet + SwiGLU），权重由编译器生成，不是训练出来的 |
| `hidden_size` | 2560 | **3968** = 2560 文本维 + 1408 芯片维 | `d0 = 2560`；芯片层只用 `2560..3967` |
| `vocab_size` | 248,320 | **478,656** | `v0 = 248320`；`< v0` 是文本 token，`>= v0` 是 CPU 执行轨迹（chip token） |
| 磁盘（BF16） | ~8 GB | **18.8 GB** | 拓宽的矩阵绝大多数是精确的 0：11 层芯片层只有 1,046,109 个非零权重 |
| 机器状态 | — | Gated DeltaNet 循环状态 | ROM、RAM、32 个寄存器和 PC 都存在芯片层的 recurrent state 里 |

`chipformer.json` 给出关键维度：`code_dims = 2754..2774`（21 维输出码），`mu_dim = 2776`，`tau_dim = 2777`，`chip_layers = 11`，`chip_dims_offset = 2560`。

**层间耦合**（`reports/ckpt_report.json` 全量扫描）：
- 芯片层的最后一层（exit 层）之后，芯片维只剩 23 个 `keep_cols` 非零：21 个 code 维、`tau`、`mu`。
- Qwen 的 32 层从不写芯片维；在芯片维里只读 `mu_dim`。
- 合并时 Qwen 层的 RMSNorm 权重按 $\sqrt{3968/2560}$ 缩放过，所以残差必须保持 3968 宽，每个 RMSNorm 才看到与合并模型相同的向量。

## 2. 解码流程

沿用上游 `RvChat`（`upstream/experimental/riscv/chipformer/rv/chat.py`）的语义：

1. **预填**：prompt 逐 token 走芯片层，同时整段过 Qwen 层；然后 `core.snapshot` 载入固件（如 `calc-bridge.elf`，停在首个 exit ecall），`hf.inject` 把机器快照写进芯片层的 DeltaNet 状态。
2. **解码**：
   - **芯片 token**（`id >= v0`）只过 11 层芯片层，Qwen 的缓存和 RoPE 位置看不到它。
   - **文本 token**先过芯片层，再过 Qwen 32 层。
   - 每个位置的下一个 token：芯片层输出码（code 维）有值时由电路决定，否则取 Qwen 的 argmax。
3. Qwen 写出 `<tool_call>…</tool_call>` 后，电路接管：先跑约 1,100 个芯片 token 解析表达式，再跑约 1,000–13,000 个算出结果，把结果逐字符回显成文本 token（`<tool_response>…</tool_response>`），最后交还给 Qwen 写最终回答。

### 提示词

`prompt_ids` 用 Qwen3.5 自带的 chat template，传入一个 `calc` 工具定义（`rv/chat.py` 的 `CALC_TOOL`），`enable_thinking=False`。系统提示词是模板标准的工具段：`# Tools`、`<tools>{calc 的 JSON schema}</tools>`、`<tool_call><function=…><parameter=…>` 的调用格式和一段 `<IMPORTANT>` 提醒。加上用户问题，第一题共 405 个 token。工具没有外部执行者，调用的执行完全在模型内部。

## 3. 芯片层：作为神经网络本身运行

默认实现（`--chip neural`）直接跑 checkpoint 里的 11 层，不转换回电路，也不还原成稠密矩阵。

### 3.1 加载与压缩

逐层（`macport/chat.py` 的 `load_layers`、`LeanChip`）：
1. 在 CPU 上建 HF decoder 层，载入该层权重。
2. `fast.prune_heads` 剪掉电路没用到的 value head（上游）。
3. 每个 Linear 只保留非零项（`NZLinear`：行、列、值）：11 层合计 1.85 亿 → **约 102 万**。
4. 合并投影：q、k、v、z、b、a 合成一个输入映射，gate 和 up 合成一个。输出全零的子层跳过（第 2 层没有 MLP，第 10 层没有注意力）。
5. 转成 CSR 交给 MLX（`macport/mlx_chip.py`），torch 那份释放。

### 3.2 状态

每个注意力层两份循环状态：conv 状态 `[C, K]`（bf16），DeltaNet 状态 `S`：`[头数, 128, 128]`（fp32）。预填后 `hf.inject` 把机器快照写进 `S`，之后机器状态由网络自己读写。电路的 key 是 one-hot，相当于按地址访问的存储。

### 3.3 一个 token 的计算

```
token → embedding（文本行或芯片行，补零到 3968 维）
  每层：
   ① norm_csr        RMSNorm·(1+w) 融合合并后的 in-proj（CSR）→ q k v z b a
   ② gdn_inplace     每个 value head 一个 threadgroup：
                     causal conv1d + SiLU → q、k 做 l2norm → β = sigmoid(b)，
                     衰减 = exp(A·softplus(a + dt_bias))
                     delta rule：kv = Sᵀk，δ = (v − kv)·β，S += k δᵀ，o = Sᵀq（原地）
                     → gated RMSNorm × silu(z)
   ③ csr_res         out-proj + 残差
   ④ norm_csr_swiglu RMSNorm + gate/up + SwiGLU
   ⑤ csr_res         down-proj + 残差
→ 残差 h（3968 维）
```

五个都是 `macport/mlx_kernels.py` 里手写的 Metal kernel（`mx.fast.metal_kernel`），一步约 50 个。每个浮点运算保持 HF 的顺序和 dtype：模块输出 bf16 的地方舍入到 bf16，norm、conv、门控和 delta rule 内部用 fp32。

**线性层**是稀疏矩阵乘向量：每个输出行只遍历它的非零项，长行分给 2–8 个 lane（`simd_shuffle_xor` 归约），特别长的输出行（512–979 个非零）用整个 simdgroup。

**delta rule 的激活稀疏**：key 是 one-hot，query 最多 2 个非零（`qhot`），不衰减的头 `exp(g)` 精确等于 1.0。所以不衰减的头每步只碰几行：kernel 先用 simd ballot 按升序挑出 k 或 q 非零的行，只处理这些行，一次性独立加载。跳过的行只贡献精确的 0。会衰减的头整头更新。

### 3.4 输出与 free run

只读 `h` 的 23 个 keep 维。21 个 code 维各取「是否 > 0」得到比特，经 `core.decode_code` 得到下一个 token（可能是芯片 token、文本 token 或「无」）。

`mlx-freerun`（默认，`FreeRunMlxChip`）把连续芯片 token 的循环放到 GPU 上（参考上游 `FreeRun`）：
- `code_lut` 预先把 `decode_code` 展开成 2^21 项查表；GPU 上由 code 维直接查出下一个 token，再 embedding 进下一步。
- 一串步骤是一张惰性图，按块提交（2 → 32 步倍增），主机读上一块时下一块已在执行。
- 出现第一个不归芯片管的 token 时 `live` 标志落下，之后已排队的步变为空操作：GDN kernel 提前返回、conv 状态原样传递。主机读到这个 token，按文本 token 的普通流程继续。

### 3.5 精确性的依据

- 权重是 bf16，bf16 × bf16 的乘积在 fp32 中精确；电路的部分和都可精确表示，所以稀疏求和的任意顺序都等于稠密 GEMM。
- 舍入位置和 dtype 与 HF 一致（BF16 恰好截掉 `1/sqrt(128)` 等带来的 1-ulp 噪声，见 pitfalls）。
- 跳过的状态行只贡献精确的 0。
- RMSNorm 的归约是唯一被重排的不精确运算，已用下面的验证覆盖。

### 3.6 对照：电路本身

- `--chip circuit`：只跑电路执行器 `fastexec.Executor`（IR 的算术，CPU fp32）。文本 token 的残差 = embedding 前 `d0` 维 + 电路给出的 keep 维。
- `--chip verify`：两者同时跑，每个位置比对 keep 维。

## 4. Qwen 32 层、embedding 与 head

- **Qwen 层**：默认 mlx-lm 的 `qwen3_5.TextModel`（`macport/mlx_qwen.py`），以芯片层输出残差作 `input_embeddings`。每个 Linear 裁成非零矩形块（读 `[0, mu]`，写 `[0, 2560)`）。`--qwen-bits 8|4` 只量化这些块（`mx.quantize`），芯片层永远是原始 bf16。`--backend torch` 改用 MPS 上的 torch 模块。
- **embedding**：文本行 `[:248320]` 只取列 hull `[0, 2908)`，芯片行只取其列 hull。
- **tied lm_head**：文本行 × 列 `[0, 2560) ∪ 23 个 keep 维`（量化时 keep 列仍保持 bf16 精确）。

## 5. 实现与代码

| 文件 | 内容 |
| :--- | :--- |
| `macport/chat.py` | 主程序 `MacChat`：加载、`--chip` 三种模式、`generate`（含 free run 接入）、命令行 |
| `macport/mlx_chip.py` | 芯片层 MLX 实现：`MlxChip`、`FusedMlxChip`、`InplaceMlxChip`、`FreeRunMlxChip`、`code_lut` |
| `macport/mlx_kernels.py` | Metal kernel：`norm_csr(_swiglu)`、`gdn`、`gdn_inplace`、`csr_res(_simd)` |
| `macport/mlx_qwen.py` | Qwen 32 层的 MLX 加载（`BlockLinear` / `QBlockLinear`）与 head |
| `macport/bench_chip.py` | 芯片层基准：teacher-forced 每步与电路比对 keep 维；`--free` 自由解码与 trace 比对 |
| `macport/web.py`、`web.html` | 本地网页：SSE 流式显示 Qwen 文本、芯片运行段和电路回显 |
| `macport/tools/inspect_ckpt.py` | 流式扫描 18.8 GB 权重（`reports/ckpt_report.json`） |

上游改了两处小地方（见 pitfalls）：`rv/fastexec.py` 的 `check()`，`q35/fast.py` 的 `compact()`。

### 网页

`.venv/bin/python -m macport.web <模型路径> [--port 8765]`，标准库 `http.server` + SSE。
- 模型在主线程运行（MLX 的 stream 属于创建它的线程），HTTP 线程把请求放进队列、把事件流回页面。
- `generate` 的回调 `on_token(id, owned)` 区分电路和 Qwen 给出的 token。页面上 Qwen 文本为白色，电路回显为绿色，芯片运行段为橙色胶囊（≤4 个 token 的短段缩成小点）。
- 每个问题是一次独立对话（重新载入固件），一次只跑一个。可选结束后用 IR + 参考机重放校验。

## 6. 芯片层优化阶梯

c-crc32 trace，batch 1，每一级都与电路逐位一致（mismatched steps 0）。ms/step 为 `--no-check`（bench 带电路比对时约 1.0 ms/step）。

| `--impl` | ms/step | tok/s | 做了什么 / 参考 |
| :--- | ---: | ---: | :--- |
| `hf` | 9.94 | 101 | 原生 HF 模块（MPS bf16）+ 上游 `fast.compact` 非零块、`fast.prune_heads`。上游 L4 eager 为 11–12.5 ms |
| `sparse` | 7.07 | 141 | 线性层只读非零权重：gather × w → scatter-add（fp32）。1.85 亿 → 102 万权重；思路同上游 executor 的 sparse 路径 |
| `lean` | 5.77 | 173 | 自写单 token decode，与 HF `causal_conv1d_update` / `torch_recurrent_gated_delta_rule` 同序同 dtype；合并投影；跳过全零子层 |
| `compiled` | 2.63（纯步 1.99） | 380 | `torch.compile`（inductor Metal 后端） |
| `mlx` | 2.71（纯步 1.56） | 370 | 同一数学用 MLX 写，`mx.compile` 整步（代替 CUDA Graph） |
| `mlx-fused` | 纯步 0.79 | — | 每层 5 个融合 Metal kernel，对应上游 Triton executor v2 的「一子层一 kernel」 |
| `mlx-inplace` v1 | 0.63 | 1599 | DeltaNet 状态原地更新，只碰 k/q 非零的行：零误差的激活稀疏（Deja Vu / PowerInfer 是带近似的版本） |
| `mlx-inplace` v2 | 0.48 | 2079 | GDN kernel 先用 simd ballot 压缩出活跃行，状态行一次性加载：21 → 11 µs/kernel |
| `mlx-inplace` v3 | 0.45 | 2201 | gate/up 的长行（最长 82 个非零）分给 2–8 个 lane |
| **`mlx-freerun`** | **0.28** | **~3500** | 连续芯片 token 在 GPU 上自回归，每块才同步一次主机 |

**时间去向**（v3）：GPU 上纯依赖链 0.305 ms/步；空 kernel 的依赖启动只有约 1.5–2 µs/个（50 个约 0.1 ms），其余是 kernel 执行。free run 去掉了每步约 0.15 ms 的主机往返。

**负结果：每层 2 个 kernel**。把 out-proj + 残差融进下一个 RMSNorm + in-proj（每个 threadgroup 自己重算上一层的小输出矩阵和整条残差，用 threadgroup 原子加，免全局同步），kernel 从 50 降到 21 且仍逐位精确，但依赖链从 0.461 变慢到 0.495 ms：瓶颈是执行而不是启动，重算和更少的线程得不偿失，已删除。

## 7. 验证

- **teacher-forced**：5 个 RV32 程序完整 trace（c-crc32 4,166 / c-sieve 44,156 / c-gmpmul 22,671 / rv32ui-add 502 / c-qsort 前 100,001 步），共 **171,496 步，每步 23 个 keep 维与电路一致，0 mismatch**（v3 kernel 重跑过）。
- **free run 自由解码**：整串 token 与参考机 trace 一致：c-crc32 4,165 / c-sieve 44,155 / c-qsort 100,000 / rv32ui-add 501。c-gmpmul 前 21,725 个一致，第一处不同是 putchar 回显，电路在同一位置也预测同一字符（trace 不记录回显）。
- **端到端**：`--check`（参考机 lock + IR replay）全部 ok；`--chip verify` 在第二题 14,377 个位置 0 mismatch。
- **上游测试**：`pytest upstream/experimental/riscv/tests`：231 passed、22 skipped；`test_rv_fuzz.py::test_fuzz_free_running_cpu` 的失败是上游既有问题（见 pitfalls）。

## 8. 端到端性能

Mac mini M6 / 24 GB，`--check` 全部 lock None、replay 0 错。题 1：`Use calc: what is 123456789 * 987654321?`（114 文本 + 2,430 芯片 token）。

| 配置 | 用时 | text tok/s | chip tok/s |
| :--- | ---: | ---: | ---: |
| 最初版本：电路（CPU）+ torch Qwen | 17.3 s | 12 | 309 |
| `--chip verify`（HF 模块 + 电路） | 42.3 s | 14 | 72 |
| torch 后端 + `mlx-inplace` | 10.2 s | 13 | 1548 |
| MLX Qwen bf16 + `mlx-inplace` v1 | 8.9 s | 15.6 | 1528 |
| **默认：MLX Qwen bf16 + `mlx-freerun`** | **8.7 s** | 14.4 | **3038** |
| `--qwen-bits 8` | 5.3 s | 25 | 3156 |
| `--qwen-bits 4` | 3.7 s | 41 | 3197 |

题 2：`(2**100 - 12345678901234567890) // 97 + 84/18` = `39205688666470630595736143441/3`（327 文本 + 13,625 芯片 token）：最初 70 s → 默认 25.7 s（芯片 3,291 tok/s）。峰值 RSS 约 11–14 GB。

**文本 token 受带宽限制**：每个文本 token 要读约 7.2 GB 的 Qwen bf16 权重 + 1.28 GB head，按约 120 GB/s 就是约 15 tok/s，torch 和 MLX 都在这个水平。只有减少字节（量化）能再快。4-bit 下 Qwen 的措辞会变（如加千分位逗号），电路结果不变。端到端里芯片部分已只占约 1 s。

## 9. 可选后续

- 芯片层约 0.28 ms/token 由 kernel 执行主导：`norm_csr` 每个 threadgroup 都重做 3968 维 RMSNorm 归约，可改为每步算一次；CSR 的 `val` 存 bf16、`col` 存 int16 可把每步约 8 MB 的读取减半。
- 持久化的单 kernel 整步需要跨 threadgroup 的全局屏障，Metal 不保证所有 threadgroup 同时驻留，有死锁风险，暂未做。
- 端到端的大头是 Qwen 文本 token，芯片层再快对总时间影响已很小。
