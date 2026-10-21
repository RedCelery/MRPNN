# MRPNN 特征与物理监督数据

实现入口：`training/generate.ps1`、`training/dataset.cu`、
`training/generate_dataset.py`。本阶段完成数据生成，不训练新权重。

## 运行与产物

在仓库根目录运行：

```powershell
.\training\generate.ps1
.\training\verify.ps1
```

生成器使用本地 CUDA 12.8 / RTX 5080 的 `build-5080` 配置。
`MRPNN_PYTHON` 可指定具有 NumPy、PyTorch 的 Python。
默认输出 `training/datasets/smoke`，包含三个真实参数组各 1,000 点，
每组分别计算 512 / 4096 条路径，还有四组诊断数据。
两档使用相同样本和特征、独立标签随机流，不能当成 6,000 个独立位置。

每个 NPZ 保存：

- `density/transmittance/phase`：各 `[N,192]`，已变换的网络输入。
- `g/gamma/alpha`：`[N,1]`；`albedo`：`[N,3]`，网络的有效参数。
- `pos/view/light/basis_y/basis_z`：坐标、传播方向和采样基底。
- `label/variance/standard_error`：RGB 间接光均值、单路径样本方差、均值标准误。
- `target_log`：`log1p(label / albedo^4)`；零 albedo 使用
  `target_log_mask=false`，训练时应屏蔽该压缩域目标。
- `direct/direct_variance/direct_reference`：直接光诊断值。
- `original_prediction`：原权重预测，仅用于比较，不参与标签生成。
- `path_caps/tracking_caps/sample_failures`：生成质量诊断。

NPZ 不含物理监督以外的光照来源；没有 HDRI/环境光。
`generation-report.json` 保存参数、seed、耗时、误差、检查结果和文件 SHA-256。
生成器失败时会返回非零退出码，已写出的部分文件不能视为完整有效批次。

## 标签定义与源码证据

`core/render.cuh::NNPredict`：

1. 用 delta tracking 从相机传播方向找到第一个真实散射点。
2. `RadiancePredict(..., scatter_rate / 1.001)` 输出后续散射的条件贡献。
3. 另加 `Tr * HenyeyGreenstein(dot(view,light),g)` 直接光。
4. 将两者之和乘 `lightColor * scatter_rate`。

因此网络监督不包括当前点的直接光、不包括当前点的外部散射率、
不包括相机到当前点的透射率；它们由渲染器的采样/组合处理。
`core/render.cuh::GetSample` 同样先在当前点采样 HG 方向，再从后续
碰撞点累加光照，每次后续碰撞乘 `1.001 * effective_albedo`。

记 `a` 为网络输入的有效 albedo，物理后续碰撞权重为 `w=1.001a`：

```text
Y_c = E[ sum(k>=1) w_c^k * T(x_k, light) * p_repo(dot(d_(k-1), light),g) ]
current-point conditional radiance = (Y_c + direct) * w_c * lightColor_c
```

这里首个方向是当前点的 HG 抽样方向；`x_k` 从之后的自由飞行得到。
没有额外乘密度：自由飞行和真实碰撞抽样已包含消光分布。
RGB 共用几何随机路径，通道方差分别保存，通道之间相关。

`core/vector.cu::HenyeyGreenstein` 返回
`(1-g²)/(4*(1+g²-2g*cos)^(3/2))`，源码将 `/pi` 注释掉。
HG 方向抽样仍是归一化概率分布。因此保存的辐射量是相同物理
单位光源约定下标准 HG 辐射量的 π 倍；不能再次乘/除 π 后直接
与原权重和原部署端混用。

`albedo^4` 是网络输出参数化，不是四次散射起点。
标签包括所有后续碰撞；压缩域目标单独保存，物理 RGB 标签保留原值。
生成器限制 `a<=1/1.001`，避免物理散射权重超过 1。

## 共享特征与估计量

`core/radiancePredict.cu::ExtractMRPNNFeatures` 从原函数原样提取：
192 点 stencil、旋转基底、MipDensity、TR mip、HG LUT 和所有变换。
运行时和数据生成器调用同一设备函数。TR mip 本身保留原推理的
缩放（`alpha / pow(1.73,mip+1)`）和 128 步积分，这是网络输入约定，
不能拿它当真实阴影标签。标签使用原始 `_DensityVolume` 的非负密度。

当前点分布：从半径 2 球面发射朝向体积内部随机目标的射线，
delta tracking 取得第一处真实碰撞，失败重试。基底绕视向随机旋转。
与运行时相同的碰撞机制，但相机分布是工程采样设计，不是作者数据集复原。
alpha/g/light 在每个批次固定，以复用 HG/TR 预计算；目前三组只是 smoke 数据。

新标签路径使用 delta tracking + 完整 ratio tracking，没有旧 `Tr` 的
`0.001` 阈值截断，也不使用原网络输出。32 次后采用基于最大通道
throughput 的俄罗斯轮盘赌并补偿存活路径；该方法无偏（无限路径极限）。
有限 bounce cap 的实际触达次数被保存，正常数据只在零触达时通过。
零触达是本批数据的证据，不保证其他参数没有截断偏差。
密度为负时按零处理；没有采用原代码负值距离场的跳步优化。

## 验证与解释限制

- 空密度、alpha=0、albedo=0：间接光必须严格为零。
- 直接光：随机透射估计对比 4096 步独立中点积分；
  容差为 6 个标准误加 0.001 积分预算。
- 仅一次后续碰撞、g=0、常密度：对比独立 Python 角度/自由飞行
  Gauss-Legendre 积分；6 个标准误加 0.001 边界预算。
  常密度 CUDA texture 在边缘存在半体素插值斜坡，预算明确覆盖该差异。
  该诊断有意限制为一次碰撞，cap 非零在该文件中合法。
- 512→4096 spp：特征逐值一致、比较两档均值差与理论合成标准误，
  检查标准误比值；Monte Carlo 容差是统计检查，不是严格证明。
- 原网络兼容性：重构前后 64 点探针二进制 SHA-256 一致；
  `verify.ps1` 的 4×256 点 CPU/GPU 逐阶段与反向梯度检查通过。

尚未恢复作者的训练数据分布、损失和训练调度，也未完成独立全阶
路径追踪器交叉验证。这里证实的是特征一致、标签边界/尺度正确、
已测试物理极限和二阶结果，以及采样噪声符合所测统计行为。
后续训练应按体积分组验证；同一批两档 spp 不能拆到训练/验证两侧。

## 本机实测（2026-10-21，RTX 5080）

最终批次使用 2048 bounce cap，轮盘赌存活率上限 0.99，
正式数据无路径上限触达。3,000 个位置 × 两档 spp 共 6 个 NPZ，
另有 128 个诊断位置。独立二阶积分最大绝对差 `0.00084282`。
`verification.json` 和 `generation-report.json` 均为 `passed=true`。

| 每组 1,000 点 | 512 spp 标签耗时 | 4096 spp 标签耗时 |
|---|---:|---:|
| Cloud g=0, alpha=0.5 | 约 39 ms | 约 293 ms |
| Cloud g=0.857, alpha=6 | 约 1.79 s | 约 13.75 s |
| Bunny g=0.5, alpha=1 | 约 37 ms | 约 277 ms |

数字是带设备同步的标签 kernel 墙钟耗时，不包括体积加载、特征/LUT
生成和 NPZ 保存；每次运行略有波动，以 JSON 最新实测为准。
普通通道两档 RMS 标准误比例约 0.35（理论 0.354）。
albedo=0.999 通道具有长尾：Cloud 的比例约 0.247，Bunny 约 0.640；
因此检查要求本批不确定性下降、均值差匹配合成标准误，而不将
所有通道都按理想平方根比例判定。4096 spp 仍不是高精度参考答案。
