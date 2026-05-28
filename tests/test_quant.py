#!/usr/bin/env python3
import argparse
import faulthandler
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def quant_ref(input_bf16: torch.Tensor):
    x = input_bf16.float()
    absmax = x.abs().amax(dim=1)
    scale = absmax / 127.0
    inv_scale = torch.where(absmax > 0, 127.0 / absmax, torch.zeros_like(absmax))
    out = torch.round(x * inv_scale[:, None]).clamp(-128, 127).to(torch.int8)
    return out, scale.float()


def load_async_compute(so_path: Path) -> None:
    if not so_path.exists():
        raise FileNotFoundError(f"async_compute shared library not found: {so_path}")
    torch.ops.load_library(str(so_path))


def run_quant_op(input_bf16: torch.Tensor):
    output = torch.empty(input_bf16.shape, dtype=torch.int8, device=input_bf16.device)
    scale = torch.empty((input_bf16.shape[0],), dtype=torch.float32, device=input_bf16.device)
    torch.ops.async_compute.quant_out(input_bf16, output, scale)
    return output, scale


def make_input(m: int, k: int, input_scale: float):
    return (torch.randn(m, k, dtype=torch.float32) * input_scale).to(torch.bfloat16)


def build_functional_cases(args):
    print("[INFO] preparing PyTorch reference before loading async_compute", flush=True)
    if not args.no_seed:
        torch.manual_seed(args.seed)

    shapes = [(1, 16), (4, 128), (17, 257), (128, 1024)]
    if args.include_large_functional:
        shapes.append((args.m, args.k))

    cases = []
    for m, k in shapes:
        name = f"random[{m},{k}]"
        print(f"[PREP] {name}: input", flush=True)
        x = make_input(m, k, args.input_scale)
        print(f"[PREP] {name}: reference", flush=True)
        expect_out, expect_scale = quant_ref(x)
        cases.append((name, x, expect_out, expect_scale))

    name = "zero_rows"
    print(f"[PREP] {name}: reference", flush=True)
    x = torch.zeros(8, 128, dtype=torch.bfloat16)
    expect_out, expect_scale = quant_ref(x)
    cases.append((name, x, expect_out, expect_scale))

    name = "non_contiguous_input"
    print(f"[PREP] {name}: reference", flush=True)
    x_base = make_input(64, 32, args.input_scale)
    x = x_base.t()
    assert not x.is_contiguous()
    expect_out, expect_scale = quant_ref(x)
    cases.append((name, x, expect_out, expect_scale))
    return cases


def check_case(name, x, expect_out, expect_scale, output_atol, scale_atol):
    print(f"[RUN] {name}: quant_out", flush=True)
    start = time.perf_counter()
    actual_out, actual_scale = run_quant_op(x)
    elapsed_ms = (time.perf_counter() - start) * 1e3

    out_diff = (actual_out.to(torch.int16) - expect_out.to(torch.int16)).abs()
    max_out_diff = int(out_diff.max().item()) if out_diff.numel() else 0
    max_scale_diff = float((actual_scale - expect_scale).abs().max().item()) if actual_scale.numel() else 0.0

    if max_out_diff > output_atol:
        idx = int(out_diff.argmax().item())
        row = idx // x.shape[1]
        col = idx % x.shape[1]
        raise AssertionError(
            f"{name}: output mismatch at ({row}, {col}), "
            f"actual={int(actual_out[row, col])}, expect={int(expect_out[row, col])}, "
            f"max_diff={max_out_diff}, atol={output_atol}"
        )
    if max_scale_diff > scale_atol:
        idx = int((actual_scale - expect_scale).abs().argmax().item())
        raise AssertionError(
            f"{name}: scale mismatch at row {idx}, "
            f"actual={float(actual_scale[idx])}, expect={float(expect_scale[idx])}, "
            f"max_diff={max_scale_diff}, atol={scale_atol}"
        )

    print(
        f"[PASS] {name}: elapsed={elapsed_ms:.3f} ms, "
        f"max_out_diff={max_out_diff}, max_scale_diff={max_scale_diff:.6g}",
        flush=True,
    )


def bench_one(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {"mean": statistics.mean(samples), "median": statistics.median(samples), "min": min(samples)}


def perf_tests(args):
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    x = make_input(args.m, args.k, args.input_scale).contiguous()
    out = torch.empty((args.m, args.k), dtype=torch.int8)
    scale = torch.empty((args.m,), dtype=torch.float32)

    def op_call():
        torch.ops.async_compute.quant_out(x, out, scale)

    def ref_call():
        quant_ref(x)

    op_time = bench_one(op_call, args.warmup, args.iters)
    ref_time = bench_one(ref_call, args.warmup, args.iters)

    bytes_per_iter = x.numel() * 2 + out.numel() * 1 + scale.numel() * 4

    op_sec = op_time["median"]
    ref_sec = ref_time["median"]
    op_gbps = bytes_per_iter / op_sec / 1e9
    ref_gbps = bytes_per_iter / ref_sec / 1e9
    speedup = ref_sec / op_sec

    print("\n[PERF]", flush=True)
    print(f"shape=[{args.m}, {args.k}], dtype=bf16 -> int8, threads={torch.get_num_threads()}", flush=True)
    print(
        f"async_compute.quant_out : median={op_sec * 1e3:.3f} ms, "
        f"min={op_time['min'] * 1e3:.3f} ms, bandwidth≈{op_gbps:.2f} GB/s",
        flush=True,
    )
    print(
        f"torch reference         : median={ref_sec * 1e3:.3f} ms, "
        f"min={ref_time['min'] * 1e3:.3f} ms, bandwidth≈{ref_gbps:.2f} GB/s",
        flush=True,
    )
    print(f"speedup: {speedup:.2f}x", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Test async_compute.quant_out correctness and performance.")
    parser.add_argument("--so", type=Path, default=Path("kernels/async_compute_op.so"), help="Path to async_compute.so")
    parser.add_argument("--m", type=int, default=4096, help="Rows/tokens")
    parser.add_argument("--k", type=int, default=7168, help="Columns/hidden size")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads; 0 keeps current setting")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-seed", action="store_true", default=True, help="Skip torch.manual_seed during diagnostics")
    parser.add_argument("--input-scale", type=float, default=3.0)
    parser.add_argument("--output-atol", type=int, default=1, help="Allowed int8 absolute error")
    parser.add_argument("--scale-atol", type=float, default=1e-2, help="Allowed fp32 scale absolute error")
    parser.add_argument("--skip-functional", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--include-large-functional", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"[ARGS] skip_functional={args.skip_functional}, skip_perf={args.skip_perf}, "
        f"include_large_functional={args.include_large_functional}, m={args.m}, k={args.k}",
        flush=True,
    )

    cases = None if args.skip_functional else build_functional_cases(args)

    load_async_compute(args.so)
    print(f"Loaded: {args.so.resolve()}", flush=True)
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)

    if cases is not None:
        print("[INFO] running quant_out functional tests", flush=True)
        for case in cases:
            check_case(*case, args.output_atol, args.scale_atol)
    else:
        print("[INFO] functional tests skipped", flush=True)

    if not args.skip_perf:
        perf_tests(args)
    else:
        print("[INFO] perf tests skipped", flush=True)

    print("[DONE] test_quant.py finished", flush=True)


if __name__ == "__main__":
    try:
        faulthandler.enable()
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise
