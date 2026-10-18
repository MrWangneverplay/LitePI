"""Leakage budget for statistical masking.

For a mask with statistical parameter sigma, N hybrid elements per query and Q
encrypted queries, the total statistical distance is bounded by

    eps <= N * Q * 2^-sigma

This script prints the maximum number of encrypted queries Q_max for a given
target eps, and the eps reached for a few representative session lengths.

Usage:
    python analysis/leakage_budget.py
    python analysis/leakage_budget.py --sigma 30 --n 1328 --eps 1e-3
"""

from __future__ import annotations

import argparse


def epsilon(n_elems: int, q: int, sigma: int) -> float:
    """Total statistical distance after q queries."""
    return n_elems * q * (2.0 ** (-sigma))


def q_max(n_elems: int, sigma: int, eps: float) -> int:
    """Largest q with epsilon(n_elems, q, sigma) <= eps."""
    return int(eps * (2 ** sigma) / n_elems)


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.2e}"
    return f"{v:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistical masking leakage budget")
    ap.add_argument("--sigma", type=int, default=30,
                    help="statistical parameter (default: profile B, 30)")
    ap.add_argument("--n", type=int, default=1328,
                    help="hybrid elements per query (default: CIFAR layer, 1328)")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="target total statistical distance (default: 1e-3)")
    ap.add_argument("--table", action="store_true",
                    help="print the full sigma x N table instead of one number")
    args = ap.parse_args()

    if not args.table:
        qm = q_max(args.n, args.sigma, args.eps)
        print(f"sigma={args.sigma}  N={args.n}  eps_target={_fmt(args.eps)}")
        print(f"Q_max = {qm:,} encrypted queries")
        for q in (1, qm // 10, qm, qm * 10):
            if q >= 1:
                print(f"  Q={q:>10,}  eps={_fmt(epsilon(args.n, q, args.sigma))}")
        return

    sigmas = (14, 20, 30)
    ns = (256, 1200, 1328)
    eps_targets = (1e-3, 1e-2)
    print("Q_max = eps_target * 2^sigma / N   (encrypted queries)\n")
    for eps in eps_targets:
        print(f"eps_target = {_fmt(eps)}")
        header = "  sigma |" + "".join(f"  N={n:<6}" for n in ns)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for sigma in sigmas:
            row = f"  {sigma:>5} |"
            for n in ns:
                row += f"  {q_max(n, sigma, eps):>8,}"
            print(row)
        print()
    print("Note: Q counts encrypted inference queries, not plaintext test-set images.")
    print("Q_max=0 means even one query exceeds eps_target at that sigma/N.")
    print("A multi-layer hybrid network multiplies N by the number of hybrid layers.")


if __name__ == "__main__":
    main()
