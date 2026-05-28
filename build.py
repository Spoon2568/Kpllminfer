import argparse
from pathlib import Path

from torch.utils.cpp_extension import _write_ninja_file_and_build_library, load

OMP_LIB_DIR = Path(
    "/home/share/fengguangnan/sibow/software/HPCKit/26.0.RC1/compiler/bisheng/lib/aarch64-unknown-linux-gnu"
)
CLANG_RT_BUILTINS = Path(
    "/home/share/fengguangnan/sibow/software/HPCKit/26.0.RC1/compiler/bisheng/lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a"
)
KUTACC_DIR_NAME = "third_party/kutacc"
KUTACC_COMPUTE_SOURCE_RELS = [
    "src/utils/check.cpp",
    "src/utils/parallel.cpp",
    "src/utils/quant.cpp",
]
KUTACC_COMM_SOURCE_RELS = [
    "src/utils/check.cpp",
    "src/utils/parallel.cpp",
]


def common_ldflags():
    return [
        "-fopenmp",
        f"-L{OMP_LIB_DIR}",
        "-lomp",
        f"-Wl,-rpath,{OMP_LIB_DIR}",
        str(CLANG_RT_BUILTINS),
    ]


def resolve_kutacc_paths(here: Path):
    kutacc_dir = here / KUTACC_DIR_NAME
    kutacc_include_dir = kutacc_dir / "include"
    kutacc_src_dir = kutacc_dir / "src"
    if not kutacc_include_dir.exists():
        raise FileNotFoundError(
            f"kutacc include directory not found: {kutacc_include_dir}"
        )
    if not kutacc_src_dir.exists():
        raise FileNotFoundError(f"kutacc source directory not found: {kutacc_src_dir}")
    return kutacc_dir, kutacc_include_dir, kutacc_src_dir


def resolve_kutacc_sources(kutacc_dir: Path, source_rels):
    sources = [kutacc_dir / rel for rel in source_rels]
    missing_sources = [str(p) for p in sources if not p.exists()]
    if missing_sources:
        raise FileNotFoundError(f"kutacc source files not found: {missing_sources}")
    return sources


def build_torch_library(
    name: str,
    source_dir: Path,
    extra_sources,
    include_paths,
    build_dir: Path,
):
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory not found: {source_dir}")

    cc_files = sorted(str(p) for p in source_dir.rglob("*.cc"))
    cc_files.extend(str(p) for p in extra_sources)
    if not cc_files:
        raise FileNotFoundError(f"No .cc files found in: {source_dir}")

    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"Step: compiling {name} ...")

    _write_ninja_file_and_build_library(
        name=name,
        sources=cc_files,
        extra_include_paths=[str(p) for p in include_paths],
        extra_cflags=[
            "-O3",
            "-g",
            "-std=c++17",
            "-fopenmp",
            "-march=armv9+sve+sve2+sme+bf16",
            "-DBGEMM",
            "-DUSE_OMP_PARALLEL",
            "-Wno-undefined-arm-za",
            "-Wno-undefined-arm-streaming",
        ],
        extra_cuda_cflags=[],
        extra_sycl_cflags=[],
        extra_ldflags=common_ldflags(),
        build_directory=str(build_dir),
        verbose=True,
        with_cuda=False,
        with_sycl=False,
        is_standalone=False,
    )

    print(f"Step finished: {name} compiled.")
    copy_latest_so(build_dir, build_dir.parent, name)


def build_compute_op(here: Path):
    csrc_dir = here / "async_rt" / "csrc"
    _, kutacc_include_dir, kutacc_src_dir = resolve_kutacc_paths(here)
    kutacc_sources = resolve_kutacc_sources(
        here / KUTACC_DIR_NAME, KUTACC_COMPUTE_SOURCE_RELS
    )
    build_torch_library(
        name="async_compute_op",
        source_dir=csrc_dir / "compute_ops",
        extra_sources=kutacc_sources,
        include_paths=[
            csrc_dir / "compute_ops",
            csrc_dir / "common",
            kutacc_include_dir,
            kutacc_src_dir,
        ],
        build_dir=here / "kernels" / "async_compute_op",
    )


def build_comm_op(here: Path):
    csrc_dir = here / "async_rt" / "csrc"
    _, kutacc_include_dir, kutacc_src_dir = resolve_kutacc_paths(here)
    kutacc_sources = resolve_kutacc_sources(
        here / KUTACC_DIR_NAME, KUTACC_COMM_SOURCE_RELS
    )
    build_torch_library(
        name="async_comm_op",
        source_dir=csrc_dir / "comm_ops",
        extra_sources=kutacc_sources,
        include_paths=[
            csrc_dir / "comm_ops",
            csrc_dir / "common",
            kutacc_include_dir,
            kutacc_src_dir,
        ],
        build_dir=here / "kernels" / "async_comm_op",
    )


def build_cpp_async_runtime(here: Path):
    runtime_src = here / "async_rt" / "cpp_async_runtime.cpp"
    if not runtime_src.exists():
        raise FileNotFoundError(f"source file not found: {runtime_src}")

    build_dir = here / "kernels" / "cpp_async_runtime"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Step: compiling cpp_async_runtime ...")

    load(
        name="cpp_async_runtime",
        sources=[str(runtime_src)],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-fopenmp",
        ],
        extra_ldflags=common_ldflags(),
        verbose=True,
        is_python_module=True,
        is_standalone=False,
        build_directory=str(build_dir),
    )

    print("Step finished: cpp_async_runtime compiled.")
    copy_latest_so(build_dir, here / "kernels", "cpp_async_runtime")


def find_latest_so(build_dir: Path):
    so_candidates = sorted(
        build_dir.glob("*.so"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return so_candidates[0] if so_candidates else None


def copy_latest_so(build_dir: Path, target_dir: Path, name: str):
    so_path = find_latest_so(build_dir)
    if so_path is None:
        print(f"  {name} .so not found, skipping copy.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / so_path.name
    target_path.write_bytes(so_path.read_bytes())
    print(f"  {name} .so copied to: {target_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Kpllminfer native extensions.")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=("all", "compute", "comm", "async_op", "cpp_async_runtime"),
        help="Build target. Default: all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent

    if args.target in ("all", "compute", "async_op"):
        build_compute_op(here)

    if args.target in ("all", "comm"):
        build_comm_op(here)

    if args.target in ("all", "cpp_async_runtime"):
        build_cpp_async_runtime(here)


if __name__ == "__main__":
    main()
