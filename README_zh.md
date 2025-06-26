# MeshShaderDemo
## 项目简介

MeshShaderDemo 是一个基于 Vulkan 的演示项目，展示了 Mesh Shader 的基本用法。集成了 meshoptimizer 进行网格优化，项目包含完整的着色器编译与运行流程，适合学习和参考 Mesh Shader 相关开发。
![hello_meshlet](https://github.com/user-attachments/assets/66cd0264-7ed4-426c-b327-4ce01fc932b7)
## 依赖

本项目依赖以下组件（部分已包含在 `env/` 目录下，无需单独安装）：

- [Vulkan SDK](https://vulkan.lunarg.com/)（需自行安装，并设置环境变量 `VULKAN_SDK`）
- [GLFW 3.5](https://www.glfw.org/)（已包含在 `env/Include/GLFW` 和 `env/libs/glfw3.lib`）
- [GLM](https://github.com/g-truc/glm)（已包含在 `env/Include/glm`）
- [meshoptimizer](https://github.com/zeux/meshoptimizer)（已包含在 `src/meshoptimizer`）


## 构建方式

1. **安装 Vulkan SDK**  
   请从 [Vulkan 官网](https://vulkan.lunarg.com/)下载安装，并确保环境变量 `VULKAN_SDK` 已正确设置。

2. **生成工程文件并编译**  
   推荐使用 CMake 3.15 及以上版本。命令如下：

   ```sh
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

   编译完成后，`MeshShaderDemo` 可执行文件会生成在 `build/Release/` 或 `build/Debug/` 目录下。

## 着色器编译

项目的着色器源码位于 `shaders/` 目录。启动前会自动调用一次编译，你也可以使用 `shaders/compile.bat` 脚本自动编译所有着色器（需已安装 Vulkan SDK 并设置 `VK_SDK_PATH` 环境变量）：

```sh
cd shaders
compile.bat
```

该脚本会自动调用 Vulkan SDK 的 `glslangValidator.exe`，将 `.vert`、`.frag`、`.mesh` 等着色器源码编译为 `.spv` 二进制文件。

## 目录结构简述

- `src/`：主程序源码及 meshoptimizer
- `env/Include/`：第三方头文件
- `env/libs/`：第三方库文件
- `env/bin/`：运行所需 DLL
- `shaders/`：着色器源码与编译脚本

## 其他说明

- 仅支持 Windows 平台。

## todo
- 使用加载模型而非手动构建mesh的方式
- 集成UI
---

如需更详细的说明或遇到问题，欢迎反馈！ 
