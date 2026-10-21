# MRPNN 训练复现路线

依据：本地 MRPNN 推理代码、附带权重、数据生成代码、论文与补充材料。
这是可执行的复现建议，不代表已经实现或恢复作者完整训练工程。

截至 2026-10-21，已完成原权重 PyTorch/CUDA 逐阶段对齐，以及
共享特征提取和 Monte Carlo 监督数据生成。实现、标签定义及实测见
[training/DATA_GENERATION.md](training/DATA_GENERATION.md)。
运行 `training/generate.ps1` 可重新生成与检查小批数据；
已完成 600 步小批物理监督微调、checkpoint 和 CUDA 权重头导出，
见 [training/PHYSICAL_FIT.md](training/PHYSICAL_FIT.md)；
运行 `training/fit.ps1` 可重复拟合与导出核验。下文保留完整阶段路线。

## 首个目标：已有权重的 PyTorch 推理与 CUDA 对齐

从 `core/NNWeight.cuh` 提取权重；按 `core/radiancePredict.cu`
实现前向网络。矩阵乘的权重排列为 `Weight[j + i * From]`，
对应 PyTorch Linear 的 `[out_features, in_features]`。
先用 FP32，关闭 AMP/TF32，降低数值核验中的额外变量。

现有代码的网络输入：

- 192 个 stencil 采样点：160 个低频点和 32 个沿光线的高频点。
- 每点包含密度、透射率、相函数三个特征通道。
- 密度变换为 `log(1 + alpha * density / 64)`。
- 相函数特征也包含 log 变换，并使用累计相函数 LUT。
- 全局参数包括 g、视线与光线夹角 gamma，以及单通道 albedo 的四次方。

网络含 SE 模块、多尺度残差特征块。三类特征分别产生 32D 低频
和 8D 高频向量，合计 120D，再与 8D 参数特征拼成 128D。
后续为 128→128→64→32→16、六组双层 16D 残差块、16→1。
每个颜色通道运行共享预测逻辑；最终输出不是直接 MLP 输出，
还包含 `max(exp(X)-1,0) * albedo^4`。

需展开 SE/SERES/ADD3/REP3 宏，逐层核对连接关系。
不能只依据论文示意图猜网络。

验证方法：从真实体积采样导出特征和 CUDA 各阶段输出，
比较 PyTorch 输出，检查平均误差、最大误差及暗区误差。
允许浮点累加顺序产生小误差，容差应结合实际数值范围确定。

## 第二个目标：生成与部署一致的监督数据

当前 `dataGen.cu` 使用 DisneyDescriptor：
10 层 × 5×5×9 密度值，加 Gamma 和 Radiance；
g 固定为 0.857、散射率固定为 1。
因此它不能直接作为当前 MRPNN 三特征网络的数据生成器。

可复用其散射点采样和 `VolumeRender::GetSamples` 的基础设施，
但必须按当前 `RadiancePredict` 实现替换描述子。
最好让训练数据生成和推理共用一个特征提取函数，
包括坐标系、mipmap、边界投影、透射率缩放和 LUT。

监督标签来自 Monte Carlo 多次散射估计，而不是既有 MRPNN 预测。
首先检查 GetSamples 内部估计量：
是否含直接光、是否含局部相函数/散射系数、单位光强和 pi 缩放，
并核对训练标签是否需除以 albedo^4 和进行 log1p。
避免重复计算部署端单独处理的直接光。

训练参数需覆盖体积、采样位置、视线方向、光方向、alpha、g、albedo。
原 dataGen 中的固定 g / albedo 不足以恢复可调材质能力。
按体积拆分训练与验证，避免同一体积随机点拆分导致泛化指标虚高。

## 第三个目标：分阶段训练并闭环验证

1. 用真实体积的已有模型输出做小规模拟合，只验证训练和导出链路。
   这是蒸馏测试，不应标成物理监督复现。
2. 用小规模路径追踪标签测试过拟合、有限值、loss 下降和输出尺度。
3. 先固定材质参数训练，再扩大参数范围与体积种类。
4. 用验证体积及前光/侧光/逆光场景评估，与路径追踪比较。
5. 导出新权重，载入原 CUDA 渲染器，再次核对前向数值和整图。

训练初始实现可采用 Adam 和压缩域误差，
但在未核实原论文训练章节前，这些只能标注为工程选择；
不能声称作者使用了某个学习率、batch size 或 loss。
论文原始设置与工程选择应分别记录。

## 时间与设备

RTX 5080 16GB 适合这类小型 MLP 训练，但应实测显存和吞吐。
主要不确定成本是生成高质量 Monte Carlo 标签及核对其物理定义。
先测 1,000 个标签在两档采样数下的耗时和方差，
再估算大规模数据成本，不直接承诺训练总时长。

## 参考

- https://www.cad.zju.edu.cn/home/jin/sig20231/MRPNN.pdf
- https://www.cad.zju.edu.cn/home/jin/sig20231/Supplementary.pdf
- `core/radiancePredict.cu`
- `core/NNWeight.cuh`
- `core/render.cuh`
- `core/volume.cu`
- `dataGen.cu`
