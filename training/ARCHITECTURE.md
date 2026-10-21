# MRPNN 代码架构与 PyTorch 前向复现

## 已完成的里程碑

恢复可微 PyTorch 模型，导入原版 146 个权重数组，共 49,673 个参数。
验证使用 PyTorch 2.11.0+cu128，FP32，关闭 TF32；对照为本机
CUDA 12.8 / RTX 5080 的 MRPNN，关闭 fast math 和可选 Tensor Core。

四组真实体积探针，每组 256 个位置／视线样本：
Cloud-03 的 `(g, alpha)` 为 `(0, 0.5)`、`(0.5, 2)`、`(0.857, 6)`，
兔子为 `(0.5, 1)`。有效 albedo 为 `[0.8, 0.9, 0.999]`，
使三个共享颜色预测分支都执行。CPU 和 CUDA PyTorch 均通过。

| 检查阶段 | 全部案例和设备的最大绝对误差 |
|---|---:|
| 120D 特征融合向量 | 7.87e-6 |
| 全局 SE 门控 | 6.85e-7 |
| 16D 降维向量 | 1.16e-6 |
| 六个残差块输出 | 1.16e-6 |
| log 域预测 | 3.58e-7 |
| 最终辐射亮度 | 7.15e-7 |

每个阶段使用 `atol=1e-4, rtol=1e-4`；
所有数值有限，所有 146 个参数数组均得到有限梯度，
当前小批次中 130 个数组有非零梯度。ReLU/门控使部分参数在特定批次上
梯度为零，并不意味着不可训练。

此结果验证网络前向，不代表训练流程已恢复、整图数值已对齐，
也没有单独验证 Python 特征提取——Python 直接读取 CUDA 导出的特征。
探针采用有效位置、视线和正交坐标系，但不是完整 delta-tracking
采样分布。原部署入口会将用户 albedo 除以 1.001，
PyTorch 接收除法之后的有效 albedo。

## 渲染器调用链

`main.cu` 创建 VolumeRender 和 Camera，进入 `GUI.hpp::RunGUI`。

1. `volume.cu / volume.hpp` 管理密度体积、mipmap、
   透射率场和相函数 LUT，也封装 Render/GetSamples。
2. `GUI.hpp` 负责 OpenGL 显示、CUDA 互操作、鼠标输入，
   使用独立 Direct3D9/ImGui 面板修改参数。
3. `render.cuh` 定义 CUDA 渲染内核和路径追踪／神经预测逻辑：
   找到散射采样点、建立坐标系、调用网络，
   再与直接光和背景贡献组合。
4. `radiancePredict.cu` 同时包含三特征提取和网络计算。
   它从体积及辅助场采样，形成密度、透射率、相函数三组输入。
5. `NNWeight.cuh` 保存已训练权重；`tensor.cuh` 提供矩阵乘，
   普通实现用 warp shuffle，可选实现用 Tensor Core。
6. `renderResources.cu/.cuh` 保存 CUDA 纹理对象和 mipmap 采样。
   `vector.cu/.cuh` 包含向量、相函数、stencil 等数学逻辑。

Camera 主要管理视角及离线渲染接口；实际交互式渲染由
GUI 调用 VolumeRender 完成。`test.cu` 是原项目比较入口，
`dataGen.cu` 是旧 Disney/RPNN 描述子数据生成器，
不是当前 MRPNN 网络的完整训练数据管线。

## 网络如何组织

三组 `[B,192]` 输入分别对应密度、透射率、相函数。
192 点拆成 160 个低频点和 32 个高频点：

- 低频层大小：8、8、16、16、16、32、32、32。
- 高频：4 层，每层 8 个点。
- 每层对三类特征计算 mean/max，与 g/gamma 拼成 8D，
  经无偏置的 SE 网络 8→8→3 产生 sigmoid 门控。
- 三类特征各有自己的 FC；通过跨层残差／拼接传递信息。
  最终各得到一个 32D 低频向量和一个 8D 高频向量，
  合计 `3×32 + 3×8 = 120D`。
- 各层统计值共 72D，加 g、gamma、albedo^4 得到 75D。
  SE 75→16→6 控制上述六个向量。
- 参数 `[g,gamma,albedo^4]` 经 3→8，与 120D 拼为 128D。
- 主干 128→128→64→32→16，然后六个双层 16D 残差块，
  最后 16→1 并使用 ReLU。
- 每个 RGB 通道以对应 albedo 调用同一套参数，
  将预测转换为 `max(exp(logit)-1,0) * albedo^4`。

`REP3` 不只是一个普通残差：它移动当前变换后的半段，
保留另一半原始特征，在后续层变宽时形成拼接。
`model.py` 明确实现此行为，避免仅照论文图重写。
权重索引 `Weight[j+i*From]` 对应 PyTorch `[out,in]`。

## 新增训练侧代码

| 文件 | 用途 |
|---|---|
| `model.py` | trainable `nn.Module`，解析权重，提供前向和阶段输出 |
| `original_weights.pt` | 原版权重的 PyTorch state_dict |
| `probe.cu` | 从真实体积导出 CUDA 特征及网络阶段结果 |
| `../core/network_trace.cuh` | 探针二进制布局 |
| `verify.py` | 运行四种探针、CPU/GPU 对齐及反向传播核验 |
| `verify.ps1` | 本机复测入口 |
| `verification.json` | 每个案例、设备、阶段的原始误差结果 |

CUDA trace 由 `MRPNN_TRACE` 控制，只在 NetworkProbe 目标中启用；
正常 VolumeRender 目标不创建或写 trace。
NetworkProbe 每 warp 32 个线程完整参与 shuffle，padding lane 也运行网络。

复测：

```powershell
.\training\verify.ps1
```

默认复用 `D:\Hong\FacePoke\.venv\Scripts\python.exe`；
可用 `MRPNN_PYTHON` 指向另一套安装 PyTorch/NumPy 的解释器。
探针临时二进制及日志位于系统临时目录，报告和权重在 training 中。

模型使用示例：

```python
from training.model import MRPNN
model = MRPNN().cuda()
# CUDA 导出的预处理特征 [B,192]，参数 g/gamma [B,1]，albedo [B,3]
radiance = model(density, transmittance, phase, g, gamma, effective_albedo)
loss = (radiance - target).square().mean()
loss.backward()
```

下一阶段：让数据生成器共用当前的三特征提取逻辑，
核对 Monte Carlo 标签的直接光／缩放定义，再做小规模训练。
