# MeshShaderDemo
[zh (Chinese README)](./README_zh.md)

## Introduction

MeshShaderDemo is a Vulkan-based demo project that uses the VK_EXT_mesh_shader extension to demonstrate the usage of MeshShader. The project includes both a TaskShader and a MeshShader to showcase the complete dispatch process and parameter passing between the two shaders. It integrates meshoptimizer for mesh optimization and includes a complete shader compilation and execution workflow, making it suitable for learning and reference in Mesh Shader development.

![anim](https://github.com/user-attachments/assets/a4314427-6535-4cf6-9935-e7f26705518d)


## Dependencies

This project depends on the following components (most are included in the `env/` directory, no extra installation required):

- [Vulkan SDK](https://vulkan.lunarg.com/) (must be installed manually, and the `VULKAN_SDK` environment variable must be set)
- [GLFW 3.5](https://www.glfw.org/) (included in `env/Include/GLFW` and `env/libs/glfw3.lib`)
- [GLM](https://github.com/g-truc/glm) (included in `env/Include/glm`)
- [meshoptimizer](https://github.com/zeux/meshoptimizer) (included in `src/meshoptimizer`)

## Build Instructions

1. **Install Vulkan SDK**  
   Download and install from the [Vulkan official website](https://vulkan.lunarg.com/), and make sure the `VULKAN_SDK` environment variable is set correctly.

2. **Generate project files and build**  
   CMake 3.15 or above is recommended. Run the following commands:

   ```sh
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

   After building, the `MeshShaderDemo` executable will be generated in the `build/Release/` or `build/Debug/` directory.

## Shader Compilation

The shader source files are located in the `shaders/` directory. Shaders will be automatically compiled once before startup. You can also manually compile all shaders using the `shaders/compile.bat` script (requires Vulkan SDK and the `VK_SDK_PATH` environment variable):

```sh
cd shaders
compile.bat
```

The script will call `glslangValidator.exe` from the Vulkan SDK to compile `.vert`, `.frag`, `.mesh`, etc. shader sources into `.spv` binaries.

## Directory Structure

- `src/`: Main source code and meshoptimizer
- `env/Include/`: Third-party headers
- `env/libs/`: Third-party libraries
- `env/bin/`: Required DLLs
- `shaders/`: Shader sources and compilation scripts

## Other Notes

- Windows only.

## todo
- Use model loading instead of manual mesh construction
- Integrate UI

---

For more details or if you encounter any issues, feel free to ask! 
