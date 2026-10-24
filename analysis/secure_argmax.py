"""Two-party secure argmax for LitePI output mode B.

Each party holds an additive share of the fixed-point logits.  A sequential
tournament performs C-1 secure comparisons, and each comparison secretly selects
both the current maximum score and its index.  The client combines only the
final index shares, so it learns one class index and not the logits.  Gate cost
is O(C*b), matching 协议设计 §9.
"""

from __future__ import annotations

import socket
import threading

from secure_relu_net import (
    Chan,
    Ctx,
    d_drelu_batch,
    d_xorbit_mult_batch,
)


def _quantize(value: float, k: int, f: int) -> int:
    return int(round(float(value) * (2.0 ** f))) % (1 << k)


def _select_shared(ctx: Ctx, a_vec, b_vec, w_vec, k: int) -> list[int]:
    """Return shares of b + w*(a-b), where a, b and w are all shared."""
    n = len(a_vec)
    if len(b_vec) != n or len(w_vec) != n:
        raise ValueError("secure-select vectors have incompatible sizes")
    mod = 1 << k
    d_vec = [(a_vec[i] - b_vec[i]) % mod for i in range(n)]
    zeros = [0] * n
    if ctx.party == 0:
        t1 = d_xorbit_mult_batch(ctx, d_vec, w_vec, w_vec, k, self_party=0)
        t2 = d_xorbit_mult_batch(ctx, zeros, w_vec, w_vec, k, self_party=1)
    else:
        t1 = d_xorbit_mult_batch(ctx, zeros, w_vec, w_vec, k, self_party=0)
        t2 = d_xorbit_mult_batch(ctx, d_vec, w_vec, w_vec, k, self_party=1)
    return [(b_vec[i] + t1[i] + t2[i]) % mod for i in range(n)]


def run_party_argmax(ctx: Ctx, shares, k: int, f: int,
                     k_compare: int) -> int:
    """Return this party's additive ring share of the winning class index."""
    q = [_quantize(v, k, f) for v in shares]
    n = len(q)
    if n < 2:
        return 0
    mask = 1 << k
    champion_score = q[0]
    champion_index = 0
    for i in range(1, n):
        # Shared bit w=1 iff current champion strictly exceeds challenger i.
        w = d_drelu_batch(
            ctx,
            [(champion_score - q[i] - 1) % mask],
            k_compare,
        )[0]
        challenger_index = i if ctx.party == 0 else 0
        champion_score, champion_index = _select_shared(
            ctx,
            [champion_score, champion_index],
            [q[i], challenger_index],
            [w, w],
            k,
        )
    return int(champion_index)


def combine_index_share(client_share: int, server_share: int, k: int) -> int:
    return int((int(client_share) + int(server_share)) % (1 << int(k)))


def self_test() -> int:
    k, f, k_compare = 44, 8, 14
    failures = 0
    for winner in range(6):
        truth = [-3.5, -1.0, 0.25, 2.75, -2.0, 1.5]
        truth[winner] += 10.0
        server_shares = [
            value / 3.0 - 0.125 for value in truth
        ]
        client_shares = [
            value - share for value, share in zip(truth, server_shares)
        ]
        srv = socket.create_server(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        holder: dict = {}

        def server_side() -> None:
            conn, _ = srv.accept()
            with conn:
                ctx = Ctx(0, Chan(conn))
                holder["share"] = run_party_argmax(
                    ctx, server_shares, k, f, k_compare,
                )

        thread = threading.Thread(target=server_side, daemon=True)
        thread.start()
        with socket.create_connection(("127.0.0.1", port)) as conn:
            client_share = run_party_argmax(
                Ctx(1, Chan(conn)), client_shares, k, f, k_compare,
            )
        thread.join(timeout=10)
        srv.close()
        got = combine_index_share(client_share, holder["share"], k)
        ok = got == winner
        failures += not ok
        print(
            f"{'PASS' if ok else 'FAIL'}  6-class winner={winner} got={got}"
        )
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
