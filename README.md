# Kpllminfer

LLM 推理加速引擎，基于 ARMv9 CPU（鲲鹏平台，支持 SVE/SVE2/SME/BF16）的异步运行时架构，通过计算与通信重叠优化 Transformer 模型推理性能。

## 环境依赖

### 系统要求

- **架构**: ARM aarch64 (鲲鹏平台)
- **编译器**: 华为 Bisheng HPCKit 26.0.RC1（提供 OpenMP 运行时和 clang 编译器）
- **内存分配**: `libmemkind`（HBM 感知内存分配）
- **Python**: 3.10+

### Python 依赖

```bash
uv pip install -r requirements.txt
```

## 环境变量

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `ASYNC_RUNTIME_ENABLE` | 否 | `0` | 设置为 `1` 启用异步运行时模式（AsyncRunMode TorchDispatchMode），将算子调度到后台线程执行 |
| `OMPI_COMM_WORLD_RANK` | 否 | `0` | MPI 进程 rank，用于 CPU 亲和性绑定。多进程部署时由 MPI 运行时自动设置 |

### 运行时环境变量

```bash
# 启用异步运行时
export ASYNC_RUNTIME_ENABLE=1

# 多进程场景下由 MPI 自动注入，无需手动设置
# export OMPI_COMM_WORLD_RANK=0
```

## 构建

项目使用 `build.py` 编译 C++ 原生扩展，依赖 Bisheng 工具链和 `kutacc` 加速库。

```bash
# 构建全部扩展（compute ops + comm ops + async runtime）
python build.py all

# 仅构建计算算子
python build.py compute

# 仅构建通信算子
python build.py comm

# 仅构建异步运行时
python build.py cpp_async_runtime
```

### 构建产物

编译产物输出到 `kernels/` 目录：

| 产物 | 说明 |
|------|------|
| `async_compute_op.so` | 自定义 PyTorch 计算算子（量化、激活函数、归一化、RoPE 等） |
| `async_comm_op.so` | 自定义 PyTorch 通信算子 |
| `cpp_async_runtime.so` | pybind11 异步运行时模块 |

### 构建 HBM 分配器

```bash
cd kp_allocator && python setup.py
```

## 使用示例

```python
import torch
import async_rt

# 异步运行时默认不启用，通过环境变量控制
# export ASYNC_RUNTIME_ENABLE=1

if async_rt.is_asyncrt_enabled():
    print("异步运行时已启用")
```

## 项目结构

```
Kpllminfer/
├── async_rt/            # 异步运行时（Python + C++ pybind11）
│   ├── __init__.py
│   ├── runtime.py       # AsyncRunMode TorchDispatchMode 分发器
│   ├── async_ops.py     # 异步算子封装
│   ├── cpp_async_runtime.cpp  # C++ 运行时核心
│   └── csrc/
│       ├── compute_ops/ # 自定义计算核 (quant, rms_norm, rope, silu 等)
│       ├── comm_ops/    # 集合通信算子
│       └── common/      # 公共头文件
├── kp_allocator/        # HBM 感知内存分配器
├── third_party/kutacc/  # 华为加速库 (int8/bf16 GEMM, FlashMLA, MoE)
├── tests/               # 正确性与性能测试
├── build.py             # 构建脚本
└── kernels/             # 编译产物目录
```
