# 小批物理监督拟合与 CUDA 权重导出

本阶段以生成器的 Monte Carlo 间接光为监督，微调原权重，验证 loss、
checkpoint、CUDA 导出和前向对齐。没有恢复作者训练配方，也没有验证整图。

## 可复现入口

仓库根目录运行 `.\training\fit.ps1`。
默认 Adam、学习率 0.0003、600 步、batch 256、FP32、关闭 TF32、
seed 20261021。需要 PyTorch、NumPy、matplotlib，可用 `MRPNN_PYTHON` 指定解释器。
训练和导出校验失败时脚本返回非零退出码。

输入是 `training/datasets/smoke` 的三个 4096 spp 批次：
两组 Cloud 共 2,000 点训练；Bunny 的 1,000 点整个体积留出。
两档 spp 没有混入训练/验证。脚本检查生成报告、文件 SHA-256、路径上限、
有效压缩域目标。原网络预测不参与监督。

优化目标：

```text
MSE(logits, log1p(MonteCarloIndirectRadiance / effective_albedo^4))
```

该 loss/优化器是工程选择，不声称与作者相同。
原输出头为 ReLU，logits 非负，等于部署输出的压缩域表示。
训练不改变网络连接方式；用全部原参数微调，检查所有梯度有限，
并以范数 1 裁剪。没有基于留出集选择 checkpoint，保存的是第 600 步。

## 本次实测

RTX 5080、PyTorch 2.11.0+cu128，训练循环及定期评价用时约 39.08 秒，
PyTorch 峰值 allocated 显存约 38.25 MiB；不含编译、磁盘加载、导出和 CUDA 核验。
该显存数字不是整个进程或显卡总占用。

| 指标 | 原权重 | 600 步 |
|---|---:|---:|
| Cloud 训练 log MSE | 0.00064132 | 0.00018609 |
| Cloud 训练物理 RMSE | 0.028441 | 0.014356 |
| Bunny 留出 log MSE | 0.00071385 | 0.00060459 |
| Bunny 留出物理 RMSE | 0.030195 | 0.028863 |

训练 loss 降低约 71.0%，训练 RMSE 降低约 49.5%。
130/146 个参数数组在本次样本中出现非零梯度且发生更新；
所有 146 个数组梯度有限。ReLU 对本批数据关闭的分支可以产生零梯度。
checkpoint 重新加载后的同设备输出逐位一致。

留出误差初期下降、后期回升；末步比原权重略好，但不证明广泛泛化。
尤其 albedo=0.999 通道的标签 RMS 标准误约为 Cloud 0.0190、Bunny 0.0267，
已接近部分测得误差。这里的 RMSE 是对有限 spp 标签，不是对精确物理解。
下一步需要增加独立体积、材质/光方向覆盖和更高质量标签，再训练验证。

## 导出与 CUDA 证据

产物目录：`training/runs/physical-smoke`。

- `weights.pt`：PyTorch state_dict，可直接加载。
- `checkpoint.pt`：模型、Adam 状态、步数、CPU/CUDA RNG 状态。
- `NNWeight.cuh`：146 个数组、49,673 个 FP32 参数，矩阵仍为 `[out,in]` 平铺。
- `fit-report.json`：配置、初末误差、loss 历史、梯度检查、产物哈希。
- `loss.svg`：训练和按体积留出的 loss 曲线。
- `export-verification.json`：CUDA 与 CPU/GPU PyTorch 的逐阶段误差。
- `replay.bin`：三组各 32 个真实监督样本的几何/材质参数，用于 CUDA 回放。

9 位有效数字保证 FP32 数值往返；Python 重新解析导出头并逐位核对 checkpoint。
导出保留原源码常量/全局存储划分：130 个小数组位于 constant memory，
16 个较大数组位于 device global readonly storage，避免 64 KiB 常量内存上限。

`core/radiancePredict.cu` 接受可选编译宏 `MRPNN_WEIGHT_HEADER`。
`FittedNetworkProbe` 编译目标使用导出头；原 `NetworkProbe` 与渲染器仍使用原权重。
导出头无需覆盖原文件即可核验，同一个 `RadiancePredict` 实现计算新权重。

回放 96 个真实监督位置：

1. CUDA 根据保存的 pos/view/light/基底/alpha/g/albedo 重新采样特征。
2. 重新生成的特征对比 NPZ 输入，容差 `1e-5`。
3. CUDA 导出融合向量、门控、降维、六个残差块、logits 和物理输出。
4. 同一 checkpoint 在 CPU/GPU PyTorch 对比各阶段，容差 `atol=rtol=1e-4`。
5. 原权重 `verify.ps1` 再跑四组 256 点回归。

本次 CUDA 导出核验 `passed=true`，新权重最终物理输出与 GPU PyTorch
最大绝对差 `4.77e-7`；三组全部中间阶段通过。原权重回归亦全部通过。

导出验证通过仅表示新权重可在原 CUDA 网络中正确执行。
没有部署新的 GUI 渲染权重、没有量化，也没有宣称全场景视觉质量改善。
