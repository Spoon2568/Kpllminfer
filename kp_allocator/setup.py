from pathlib import Path
from torch.utils.cpp_extension import load

def build_my_allocator(here: Path):
    # 假设 custom_allocate.cpp 与此脚本在同一目录下
    # 如果在其他目录（例如 csrc/），请相应修改路径：here / "csrc" / "custom_allocate.cpp"
    src_file = here / "custom_allocator.cpp"
    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")

    build_dir = here / "kp_allocator" / "kp_allocator"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling my_allocator ...")

    # 使用 load 进行 JIT 编译
    module = load(
        name="kp_allocator",
        sources=[str(src_file)],
        extra_cflags=[
            "-O3",
            "-g",
            "-std=c++17",
            "-fopenmp",
            # 保留针对 ARM 架构和 SVE 的向量化指令优化
            "-march=armv9+sve+sve2+sme+bf16",
        ],
        extra_ldflags=[
            "-lmemkind",
        ],
        verbose=True,
        is_python_module=True,  # 关键点：使用了 pybind11，这里必须设为 True
        is_standalone=False,
        build_directory=str(build_dir),
    )

    print("Compilation finished.")
    return module

def find_latest_so(build_dir: Path):
    so_candidates = sorted(
        build_dir.glob("*.so"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return so_candidates[0] if so_candidates else None

def main():
    here = Path(__file__).resolve().parent

    # 1. 编译并加载自定义分配器算子
    module = build_my_allocator(here)

    # 此时可以直接在 Python 中使用 module，例如 module.my_allocator_func(...)

    # 2. 如果你需要将 .so 文件提取出来备用
    print("\nBuild summary:")
    allocator_so = find_latest_so(here / "kp_allocator" / "kp_allocator")

    if allocator_so is not None:
        target_path = here / "kp_allocator" / allocator_so.name
        # 使用 write_bytes 复制文件
        target_path.write_bytes(allocator_so.read_bytes())
        print(f"  my_allocator .so copied to: {target_path}")
    else:
        print("  my_allocator .so not found, skipping copy.")

if __name__ == "__main__":
    main()