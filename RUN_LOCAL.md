# MRPNN 本地运行

项目：`D:\Hong\MRPNN`。
来源：https://github.com/What-a-stupid-username/MRPNN
下载提交：`d833df008c7f3bc30f974a4dff6f19484cfee593`。

## 启动

双击 `start.bat`，默认运行 Cloud-03。
在本目录的 PowerShell 中切换示例：

```powershell
.\start.bat
.\start.bat TestCase\BUNNY
```

参数是体积文件路径，不包含 `.bin` 后缀。
Cloud-03 和兔子密度数据复用自本机 MRBNN 示例，已经复制到
`TestCase`，无需依赖 MRBNN 的模型权重。
MRPNN 使用自己附带的预训练权重。
兔子只使用密度形状，不读取 MRBNN 的彩色 albedo。
原 README 的 Cloud0 / 其他演示数据仍可从 OneDrive 下载。

## 操作

- 鼠标左键拖动：旋转相机。
- 滚轮：缩放。
- Esc：关闭渲染窗口。
- 参数面板中的 `Predict` 默认开启，使用神经网络渲染。
- `Inspector` 提供 G、Alpha、Scatter rate、光照方向和颜色等参数。
- 关闭 Predict 可切换路径追踪；本次仅验证默认 MRPNN 模式。

## 本机适配与验证

- RTX 5080，CUDA 12.8，目标架构 `120`，Visual Studio 2022。
- 已成功编译 Release：`build-5080\Release\VolumeRender.exe`。
- 已检查 Cloud-03 实际渲染画面；窗口持续响应。
- 尚未验证与 CUDA 11 原版或路径追踪参考图的数值一致性。

原版使用 CUDA 11 已废弃的 texture / surface references。
本地将其迁移为 CUDA texture / surface objects，
保留原有线性采样、归一化坐标、边界模式与坐标顺序；
替换对象前同步 GPU，检查创建与绑定错误。
同时修正 BRDF 中五处标量 `saturate` 调用为已有 `saturate_`，
增加启动时指定体积文件和缺失文件检测。

GUI 依赖改为 CMake FetchContent，使用 GLEW 2.2.0 和 GLFW 3.4。
未启用 RPNN 对照模式或可选 Tensor Core 路径；
关闭 fast math，以保守配置运行现代 CUDA。

## 增量编译

```powershell
cmake --build build-5080 --config Release --target VolumeRender -j 8
```

本次构建缓存引用了临时目录中的 GLEW 源码及本机 MRBNN 的 GLFW 源码；
当前可执行文件静态链接这些依赖，启动不依赖其源码目录。
若重新配置构建，可让 CMake 按项目配置下载依赖：

```powershell
cmake -S . -B build-fresh -G "Visual Studio 17 2022" -A x64 -T "cuda=12.8" -DGUI=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DDISABLE_FAST_MATH=ON "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
cmake --build build-fresh --config Release --target VolumeRender -j 8
```

使用新目录编译后，相应调整 `start.bat` 中的可执行文件路径。
源码许可详见仓库 `LICENSE`。
