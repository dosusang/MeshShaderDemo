cmake_minimum_required(VERSION 3.15)
project(MeshShaderDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan SDK 路径（可根据实际安装位置调整）
if(NOT DEFINED ENV{VULKAN_SDK})
    message(FATAL_ERROR "Please set the VULKAN_SDK environment variable!")
endif()
set(VULKAN_SDK $ENV{VULKAN_SDK})

# 源文件
set(SRC
    src/hello_meshlet.cpp
)

file(GLOB MESHOPT_SRC
    src/meshoptimizer/*.cpp
)

# 头文件搜索路径
include_directories(
    env/Include
    ${VULKAN_SDK}/Include
)

# 链接库搜索路径
link_directories(
    ${CMAKE_SOURCE_DIR}/env/libs
    ${VULKAN_SDK}/Lib
)

# 可执行文件
add_executable(MeshShaderDemo ${SRC} ${MESHOPT_SRC})

# meshoptimizer 过滤器
source_group("meshoptimizer" FILES ${MESHOPT_SRC})

# 链接静态库
target_link_libraries(MeshShaderDemo
    glfw3.lib
    vulkan-1.lib
)

# 拷贝 DLL 到输出目录（仅 Windows）
if (WIN32)
    add_custom_command(TARGET MeshShaderDemo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/env/bin/assimp-vc143-mt.dll
            $<TARGET_FILE_DIR:MeshShaderDemo>
    )
endif()

# 方便 IDE 识别头文件
file(GLOB_RECURSE HEADERS
    env/Include/**/*.h
    env/Include/**/*.hpp
)
source_group("Header Files" FILES ${HEADERS})

# 方便 IDE 识别着色器
file(GLOB SHADERS
    shaders/*.spv
    shaders/*.vert
    shaders/*.frag
    shaders/*.mesh
)
source_group("Shaders" FILES ${SHADERS})