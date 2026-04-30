from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Callable

import torch

_LOCAL_TMP = os.path.abspath(".tmp")
os.makedirs(_LOCAL_TMP, exist_ok=True)
os.environ.setdefault("TMP", _LOCAL_TMP)
os.environ.setdefault("TEMP", _LOCAL_TMP)
os.environ.setdefault("TMPDIR", _LOCAL_TMP)
tempfile.tempdir = _LOCAL_TMP
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from flash_kmeans import batch_kmeans_Euclid as active_batch_kmeans_Euclid
from flash_kmeans.torch_fallback import batch_kmeans_Euclid_torch_native

try:
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid as triton_batch_kmeans_Euclid
except Exception:
    triton_batch_kmeans_Euclid = None


BackendFn = Callable[[torch.Tensor, int, int, float, torch.Tensor | None], tuple]


@dataclass
class Backend:
    name: str
    fn: BackendFn


def _make_triton_backend(use_heuristic: bool) -> Backend | None:
    if triton_batch_kmeans_Euclid is None:
        return None

    def _run(x: torch.Tensor, n_clusters: int, max_iters: int, tol: float, init_centroids: torch.Tensor | None):
        return triton_batch_kmeans_Euclid(
            x,
            n_clusters,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=False,
            use_heuristic=use_heuristic,
        )

    suffix = "heuristic" if use_heuristic else "autotune"
    return Backend(name=f"triton_{suffix}", fn=_run)


def _make_active_backend() -> Backend:
    def _run(x: torch.Tensor, n_clusters: int, max_iters: int, tol: float, init_centroids: torch.Tensor | None):
        return active_batch_kmeans_Euclid(
            x,
            n_clusters,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=False,
        )

    return Backend(name=f"active::{active_batch_kmeans_Euclid.__module__}", fn=_run)


def _make_torch_backend() -> Backend:
    def _run(x: torch.Tensor, n_clusters: int, max_iters: int, tol: float, init_centroids: torch.Tensor | None):
        return batch_kmeans_Euclid_torch_native(
            x,
            n_clusters,
            max_iters=max_iters,
            tol=tol,
            init_centroids=init_centroids,
            verbose=False,
        )

    return Backend(name="torch_native", fn=_run)


def _benchmark_backend(
    backend: Backend,
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int,
    tol: float,
    init_centroids: torch.Tensor,
    warmup: int,
    rounds: int,
) -> dict:
    for _ in range(warmup):
        backend.fn(x, n_clusters, max_iters, tol, init_centroids.clone())

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(rounds):
        backend.fn(x, n_clusters, max_iters, tol, init_centroids.clone())
    end_evt.record()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(end_evt) / rounds
    return {
        "backend": backend.name,
        "time_ms": total_ms,
        "time_per_iter_ms": total_ms / max_iters,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark flash-kmeans backends without optional third-party dependencies.")
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--num-points", "-n", type=int, default=32768)
    parser.add_argument("--dim", "-d", type=int, default=128)
    parser.add_argument("--num-clusters", "-k", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--output-file", type=str, default="")
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    torch.manual_seed(0)
    x = torch.randn(args.batch_size, args.num_points, args.dim, device="cuda", dtype=dtype)

    init_indices = torch.randperm(args.num_points, device=x.device)[: args.num_clusters]
    init_centroids = x[:, init_indices, :].contiguous()

    backends = [_make_torch_backend(), _make_active_backend()]
    for backend in (_make_triton_backend(True), _make_triton_backend(False)):
        if backend is not None:
            backends.append(backend)

    seen = set()
    unique_backends = []
    for backend in backends:
        if backend.name in seen:
            continue
        seen.add(backend.name)
        unique_backends.append(backend)

    results = []
    for backend in unique_backends:
        result = _benchmark_backend(
            backend,
            x,
            args.num_clusters,
            args.max_iters,
            args.tol,
            init_centroids,
            args.warmup,
            args.rounds,
        )
        result.update(
            {
                "batch_size": args.batch_size,
                "num_points": args.num_points,
                "dim": args.dim,
                "num_clusters": args.num_clusters,
                "dtype": args.dtype,
                "max_iters": args.max_iters,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
        results.append(result)
        print(f"{result['backend']}: {result['time_ms']:.3f} ms total, {result['time_per_iter_ms']:.3f} ms/iter")

    if args.output_file:
        with open(args.output_file, "a", encoding="utf-8") as output:
            for result in results:
                output.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
