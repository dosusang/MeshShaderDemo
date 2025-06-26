@echo off
setlocal enabledelayedexpansion

rem 设置 Vulkan SDK 的 glslangValidator 路径
set "GLSLC_PATH=%VK_SDK_PATH%\Bin\glslangValidator.exe"

rem 设置要遍历的文件夹路径（当前脚本所在目录）
set "FOLDER_PATH=%~dp0"

rem 编译规则列表：格式为 文件后缀|编译参数|输出后缀
for %%T in (
    "vert|-V|_vert.spv"
    "frag|-V|_frag.spv"
    "mesh|-V --target-env vulkan1.2|_mesh.spv"
) do (
    for /f "tokens=1,2,3 delims=|" %%a in (%%T) do (
        for /r "%FOLDER_PATH%" %%f in (*%%a) do (
            set "FILENAME=%%~nf"
            set "FILEDIR=%%~dpf"
            echo 编译 %%a Shader: %%f
            "%GLSLC_PATH%" %%b -o "!FILEDIR!!FILENAME!%%c" "%%f"
        )
    )
)
