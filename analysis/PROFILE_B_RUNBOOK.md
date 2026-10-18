# Profile B runbook (2026-10-18)

## What changed

- `params_b.py` — single source of truth: sigma=30, k_ring=44, k_compare=14,
  f=8, N=16384, scale 2^15.
- `hybrid_mpc.py` — uses profile B; input mask sigma=30.
- `mpc_relu_fxp.py` — default k_ring=44 / k_compare=14 / f=8; output mask uses
  `secrets` (CSPRNG), not numpy PCG64.
- `mpc_relu_secure.py` — slim carry-chain DReLU: 58 -> 27 base OTs for
  k_compare=14 (56 -> 25 for the sign alone).
- `secure_relu_net.py` — networked slim DReLU; roles fixed
  (party0=server, party1=client); client draws the output mask.
- `gmw_core.py` — base 1-of-2 OT counter for verification.
- `leakage_budget.py` — eps = N*Q*2^-sigma and the Q_max table.
- `bench_pi.py` — client-side HE benchmark for the Raspberry Pi.
- `train_deep_conv.py` — RGB + BatchNorm + multi-seed + JSON output.
- `model/cifar10_cnn_relu.py` — seed before model creation; AdamW.
- Eight analysis scripts — repo-relative `DEFAULT_ROOT` (no hard-coded `D:\`).

## Run on your machine

## Verified results (2026-10-18, desktop)

- `mpc_relu_secure.py --self-test --bits 14`: ALL PASS; slim DReLU
  `old_sign=56 new_sign=25`; the full slim ReLU is 27 with the two
  bit-select OTs (old full ReLU: 58).
- `mpc_relu_fxp.py --self-test`: ALL PASS; `k_ring=44, k_compare=14, f=8,
  sigma=30`; server-share TV `<= 9.31e-10`.
- `secure_relu_net.py --net-self-test`: ALL PASS; `k=44, k_compare=14, f=8,
  sigma=30`; z = 3.0, -2.0, 5.5, -0.5 all exact.
- `hybrid_mpc.py --self-test --real-relu`: ALL PASS with the 23-bit chain
  `[60, 23x6, 60]` and scale 2^23; `share->HE recovery` abs_err=3.09e-4
  (tol 1.17e-2), `HE->share` 1.70e-3, `share->HE` 1.11e-3.

### 1. MPC self-tests (no TenSEAL)

```powershell
.venv\Scripts\python.exe analysis\mpc_relu_secure.py --self-test --bits 14
.venv\Scripts\python.exe analysis\mpc_relu_fxp.py --self-test
.venv\Scripts\python.exe analysis\secure_relu_net.py --net-self-test
```

Expected: `slim DReLU uses fewer base OTs [old=56 new=25]` and
`(k_ring=44, k_compare=14, f=8, sigma=30) ALL PASS`.

### 2. HE self-test (needs TenSEAL)

```powershell
.venv\Scripts\python.exe analysis\hybrid_mpc.py --self-test --real-relu
```

Expected: all PASS with `sigma=30`. If `share->HE recovery` fails, lower SIGMA
or adjust the scale / outer primes in `params_b.py`.

### 3. Leakage table

```powershell
.venv\Scripts\python.exe analysis\leakage_budget.py --table
```

This prints Q_max = eps * 2^sigma / N for sigma in {14,20,30} and
N in {256,1200,1328}. Q counts encrypted inference queries, not plaintext
test-set images.

### 4. CIFAR ladder (GPU server)

```powershell
.venv\Scripts\python.exe analysis\train_deep_conv.py --conv-depth 2,3,4 `
  --activation relu,p2 --rgb --bn --epochs 100 --num-seeds 3
```

This writes one JSON per (activation, depth, seed) under
`results/raw_results/E1/cifar10/deep/`.

### 5. Raspberry Pi benchmark (after the Pi arrives)

```bash
.venv/bin/python analysis/bench_pi.py --vector-size 1328 --repeat 5 \
  --out results/pi_cifar_B.json
```

Report keygen, encrypt, decrypt, one hybrid-layer round, public/full context
size, and peak RSS. Run the same command on the desktop for the ratio.

## Caveats

- The CKKS parameter set is a starting point, not a validated final answer:
  run the lattice estimator and the HE self-test before putting it in the paper.
- The bounded-Q theorem must go into the security section:
  `eps <= N * Q * 2^-sigma`, with Q defined as encrypted inference queries.
- The paper should not claim unlimited/lifetime queries under profile B.

## Troubleshooting

- `RuntimeError: failed to find enough qualifying primes` — SEAL requires
  `p = 1 mod 2N`. For N=16384 the number of qualifying primes is 1 at 18
  bits, 2 at 20, 4 at 21, 5 at 22, and 19 at 23. Use 23-bit level primes;
  six multiplications need seven levels, so the current chain has seven
  23-bit primes (`[60, 23x7, 60]`) and scale 2^23.
