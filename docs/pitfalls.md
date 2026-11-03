# Qwenputer Mac 适配：踩坑实录

适配过程中遇到的问题、原因和解决办法。实现本身见 [technical.md](technical.md)。

## 一、硬件与数值

### 1. 芯片层不能放到 ANE（神经网络引擎）上
- **现象**：用 `macport/tools/inspect_ckpt.py` 扫描全部权重（`reports/ckpt_report.json`），芯片层的 1,046,109 个非零权重里有 **854,873 个（81.7%）超过 FP16 上限 65504**。电路用极端量级的权重实现阈值门、温度计编码（$8 \cdot 2^m$）和 $2^{36}$ 折叠。
  - 最大值：`layers.1.mlp.gate_proj` 8.5e37，`layers.0.mlp.gate_proj` 5.2e33，`layers.0/1.linear_attn.in_proj_qkv` 2.3e18，`model.norm.weight[2777]`（tau）1.15e18；`embed_tokens[:, 2560:]` 最大 1.1e12，共 2,967 万个元素超过 FP16 上限。
- **原因**：ANE 只支持 FP16 / INT8，超过 65504 直接变成 `inf`。
- **结论**：芯片层只能用 GPU（Metal，BF16/FP32）。Qwen 的 32 层虽然权重在 ±19 以内，但与芯片层逐 token 交错执行，还有循环状态和 KV 缓存，拆到 CoreML/ANE 的跨引擎同步开销远大于收益。全部放在 GPU 上。

### 2. GEMM 的累加精度
- **背景**：上游在 `q35/fast.py:32` 设置 `allow_bf16_reduced_precision_reduction = False`。芯片层依赖精确的 2 的幂抵消（如 `128 - 33280 + 65536 = 32384`），如果 GEMM 用 BF16 做部分和，`128 - 33280` 会先舍入成 `-33280`，最后得到错误的 `32256`。
- **Mac 上的情况**：实测 MPS 和 MLX 的 bf16 矩阵乘都用 FP32 累加器，`[1,1,1] @ [128, -33280, 65536]ᵀ` 精确返回 32384。自写的 Metal kernel 也一律用 fp32 累加。

### 3. 为什么 BF16 反而比 FP32 更「精确」
GatedDeltaNet 对 query 乘 `1/sqrt(128)`（无理数），chunk kernel 里还有 `exp(cumsum)`。FP32 下会留下约 1 ulp 的尾数噪声（如 `0.25` 变成 `0.2500611`），而模块输出时强转 BF16 恰好把它截掉，恢复成精确的 2 的幂（上游 `docs/rv32.md:607`）。所以实现必须在 HF 舍入到 bf16 的同一位置舍入，不能「更精确」地全程 fp32。

## 二、上游代码对 CUDA 的绑定

### 4. `RvChat` 在非 CUDA 设备上退化
`rv/chat.py:122-124` 硬编码 `self.fast = fast and self.dev == "cuda"`、`self.executor = executor if self.dev == "cuda" else False`。在 Mac 上会退化成 CPU 上的稠密全栈（18.8 GB）。**解决**：不改 `RvChat`，在 `macport/chat.py` 里继承它（`MacChat`），重写加载、芯片步和 `generate`。

### 5. MPS 不支持 float64
`rv/fastexec.py` 的 `check()` 里 `ex.step(...)[0].double().cpu()` 在 MPS 上报 `Cannot convert a MPS Tensor to float64`。**修复**：改成 `.cpu().double()`（上游补丁）。

### 6. `torch.cuda.empty_cache()`
`q35/fast.py` 的 `compact()` 无条件调用它。**修复**：加 `if torch.cuda.is_available()` 守卫（上游补丁）。

### 7. `Qwen3_5TextRotaryEmbedding` 不接受 `device` 参数
**解决**：在 `with torch.device(device):` 里构造。

## 三、内存与加载

### 8. MPS OOM（驱动内存 18 GB）
- **原因**：替换线性层时遍历 `named_modules()`，列表里还包含将被替换的 `nn.Linear`，它们的稠密权重也被载入并搬上了 GPU。
- **解决**：逐层在 CPU 上加载 → 剪头 → 压缩成非零项 → 再 `.to(device)`；搬运前先 `.contiguous()` 避免临时副本，加载后 `torch.mps.empty_cache()`。

### 9. `copy.copy(nn.Module)` 共享 `_buffers`
合并多个 NZ 映射时用了浅拷贝，新旧 module 共享 `_buffers` 字典，覆盖了原层的索引，之后 `hf.inject` 报索引越界。**解决**：新建一个 `NZLinear` 并重新注册 buffer。

### 10. tied lm_head 不能只读前 2560 列
文本行的 embedding 在 code 维和 `tau_dim` 上也非零（code 维上 260 个可打印字符行非零，tau 上每行都非零）。只读 `[0, 2560)` 会算错 logits。**解决**：head 列取 `[0, 2560) ∪ 23 个 keep 维`；量化 head 时这 23 列单独保持 bf16。

## 四、Metal / MLX kernel

### 11. 写 Metal kernel 的几个限制
- Metal 没有 `log1p`：用精确等价式 `w = 1+u; w == 1 ? u : log(w)·u/(w−1)`（softplus 需要）。
- MLX 把小数组以 `constant` 地址空间传入，与 `device` 指针类型不匹配：helper 写成模板。
- bf16 类型是 `bfloat16_t`，转换用 `static_cast`。
- `mx.fast.metal_kernel` 的模板参数不接受 float：eps、`1/sqrt(dk)` 等常量直接替换进源码字符串，按值缓存 kernel。

### 12. 原地更新状态
- 原地 kernel 要 `ensure_row_contiguous=False`，否则 MLX 可能先复制一份，写进副本。
- 状态必须是已实体化的独立数组：`mx.zeros` 可能是广播视图，写入会出错。用 `mx.array(np.array(s))` 实体化。

### 13. DeltaNet kernel 里「跳过的行」并不免费
`mlx-inplace` v1 已经只读写活跃行，但逐 kernel 计时发现它仍要约 21 µs：两个 128 次的循环在每次迭代都检查 `ks[d]`、`qs[d]`，活跃行的设备内存读取还是串行等待。**解决**：先用 simd ballot 按升序压缩出活跃行列表（与原循环同序，结果逐位相同），再一次性独立加载这些行：降到约 11 µs。

### 14. 长行由单线程串行求和
gate/up 映射最长一行有 82 个非零项，一个线程串行求和成了瓶颈。**解决**：每行分给 2–8 个 lane，用 `simd_shuffle_xor` 归约。各项乘积精确、部分和可精确表示，所以求和顺序不影响结果。

### 15. 负结果：减少 kernel 数反而变慢
先入为主地以为每步 50 个 kernel 是「启动延迟」瓶颈，于是做了每层 2 个 kernel 的版本（threadgroup 自己重算上一层的输出和残差），kernel 减到 21 个，仍逐位精确，但更慢（0.461 → 0.495 ms）。实测空 kernel 的依赖启动只要约 1.5–2 µs，真正的瓶颈是 kernel 执行。**教训**：先逐 kernel 计时，再决定优化方向。

## 五、性能归因

### 16. 文本 token 慢的原因判断错了一次
起初以为文本 token 慢是因为 HF 的 fallback（非 CUDA 下没有融合 kernel）。实际是**内存带宽**：每个文本 token 要读约 7.2 GB Qwen 权重 + 1.28 GB head，按约 120 GB/s 就是约 15 tok/s，换成 mlx-lm 的融合 kernel 也一样。只有量化（`--qwen-bits 8|4`）能再快。

## 六、验证

### 17. free run 在 gmpmul 上「不一致」其实是 trace 的差异
自由解码 c-gmpmul 时，第 21,725 个 token 网络输出了文本 'm'，而 `ref.Machine.run` 的 trace 是 `BYTE(109)` 后接 `F0(109)`。原因是 trace 不记录 putchar 的回显（chat 模式下电路会把输出字节回显成 Q 文本 token）。用电路 teacher-forced 跑到同一位置，它同样预测 'm'（token 76）。网络与电路一致。

### 18. 上游既有的测试失败
`test_rv_fuzz.py::test_fuzz_free_running_cpu` 报 `KeyError: 'Q'`（`rv/spec.py:111`）。撤掉本次两处补丁后同样失败，属上游问题。其余 231 passed、22 skipped。

## 七、网页

### 19. MLX 的 stream 绑定线程
网页初版在 HTTP 处理线程里调用模型，报 `RuntimeError: There is no Stream(gpu, 0) in current thread`。MLX 的 stream 属于创建它的线程（加载模型的主线程）。**解决**：模型留在主线程循环处理任务队列，HTTP 服务放到后台线程，通过队列传请求和事件。

## 八、杂项

- 在 scratchpad 里跑临时脚本要加 `PYTHONPATH=.`，否则找不到 `macport`。
- shell 循环里把 `"--qwen-bits 8"` 写成一个带引号的字符串会变成单个参数，要拆开写。
