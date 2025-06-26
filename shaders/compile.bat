@echo off
setlocal enabledelayedexpansion

rem ���� Vulkan SDK �� glslangValidator ·��
set "GLSLC_PATH=%VK_SDK_PATH%\Bin\glslangValidator.exe"

rem ����Ҫ�������ļ���·������ǰ�ű�����Ŀ¼��
set "FOLDER_PATH=%~dp0"

rem ��������б���ʽΪ �ļ���׺|�������|�����׺
for %%T in (
    "vert|-V|_vert.spv"
    "frag|-V|_frag.spv"
    "mesh|-V --target-env vulkan1.2|_mesh.spv"
) do (
    for /f "tokens=1,2,3 delims=|" %%a in (%%T) do (
        for /r "%FOLDER_PATH%" %%f in (*%%a) do (
            set "FILENAME=%%~nf"
            set "FILEDIR=%%~dpf"
            echo ���� %%a Shader: %%f
            "%GLSLC_PATH%" %%b -o "!FILEDIR!!FILENAME!%%c" "%%f"
        )
    )
)
