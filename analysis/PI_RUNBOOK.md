# LitePI on Raspberry Pi — runbook

This folder is a **deployment demo**, not a new experiment result.  It puts the
real two-party protocol on two machines:

```
   Raspberry Pi (client / party1)            Desktop (server / party0)
   -----------------------------             -------------------------
   CKKS secret key                           HE model weights
   Encrypt input  ------------------------>  Linear layers on ciphertext
   Decrypt z+r    <------------------------  Enc(z+r), fresh mask r
   MPC ReLU share  <===== 27 base OTs ====>  MPC ReLU share (s_s = -r)
   Encrypt c_c    ------------------------>  Encrypt c_s, add ciphertexts
   Decrypt logits <------------------------  Enc(logits)
```

The model is a small random-weight MLP.  Its job is to exercise every step of
the five-step hybrid flow over a real socket; it is **not** a CIFAR/MNIST
accuracy result.

The Pi code uses the explicitly labelled `pi-demo-sigma30` profile in
`params_pi.py`: `k_ring=44`, `k_compare=14`, `f=8`, `sigma=30`, CKKS scale
2^24 and chain `[60,24,60]`.  This is a runnable deployment profile with the
paper's 30-bit statistical margin.  It uses only one multiplication between
hybrid ReLUs and must not be reported as the paper's six-layer profile-B
precision result.

`pi_server.py --weights model.npz` loads real weights (`W0,b0,W1,b1,...`);
without it the weights come from `--seed`.  The client never needs the weights
or the file.

## 1. Files

| File | Runs on | Purpose |
|---|---|---|
| `pi_proto.py` | both | framing, byte counters, TenSEAL/stub backend |
| `params_pi.py` | both | the separate Pi demo profile (sigma=30, one-mult chain) |
| `pi_server.py` | desktop | HE linear layers + MPC party0 |
| `pi_client.py` | Pi | keygen, encrypt, decrypt, MPC party1, re-encrypt |
| `pi_bench.py` | desktop | repeated loopback benchmark, JSON output |
| `pi_preflight.py` | Pi | environment, HE round trip, MPC smoke test |

The client imports `params_pi`, `secure_relu_net` and the OT stack from the same
folder.  The low-level MPC defaults still come from `params_b.py`; the Pi
entry points pass the `params_pi.py` ring parameters explicitly.

## 2. Prepare the Pi

Use **Raspberry Pi OS 64-bit (Bookworm)**.  A Pi 5 with 8 GB is the realistic
target: the deployment uses `N=16384`, which is heavy for a Pi Zero.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgomp1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2" cryptography
```

`cryptography` is not optional in practice: without it `ot_core_fast.py` falls
back to a pure-Python curve that is orders of magnitude slower.

## 3. Installing TenSEAL on ARM

TenSEAL wheels are not guaranteed for `aarch64`.  Check first; do not assume:

```bash
python -m pip install tenseal==0.3.18
python -c "import tenseal; print(tenseal.__version__)"
```

If pip reports "no matching distribution", try in this order:

1. **Conda-forge (Miniforge aarch64)**

   ```bash
   # only if a tenseal build exists for linux-aarch64
   conda install -c conda-forge tenseal
   ```

2. **Build from source** (SEAL 3.7.x + TenSEAL 0.3.18).  Expect 1–3 hours on a
   Pi 4/5 and at least 4 GB free disk.  Install `cmake`, `build-essential`,
   `libzstd-dev`, `libprotobuf-dev`, then build SEAL with
   `-DSEAL_USE_ZSTD=ON` and TenSEAL against it.  This path is only worth it if
   nothing else works.

3. **Do not** measure the Pi with a desktop-only fallback and label it "Pi".
   If TenSEAL cannot run, report the deployment as blocked and use the desktop
   numbers separately.

Run the preflight once TenSEAL imports:

```bash
python pi_preflight.py
```

Expected output includes the Pi-demo keygen time, an encrypt/decrypt round trip,
both MPC self-tests, a real-CKKS hybrid loopback, and `PREFLIGHT OK`.
`--skip-mpc` checks HE only.

## 4. Put the files on the Pi

The six Python files already live in `analysis/` in the repository.  Clone or
pull the repository on the Pi; do not copy only the client file, because it
depends on the shared MPC and parameter modules.

## 5. Smoke test on one machine (no network, no Pi)

This validates the message order, masking and share arithmetic:

```bash
# protocol wiring only -- stub HE is NOT private
python pi_client.py --loopback --backend stub --dims 32,16,8,4

# real CKKS on a desktop with TenSEAL
python pi_client.py --loopback --backend tenseal --dims 32,16,8,4 --verbose
```

Expected: `hybrid_layers=2`, 24 ReLU elements, 648 base OTs, and
`logits err ... PASS`.  The tolerance is `5e-2`; the MPC fixed point is 1/256
per element, and the final linear layer propagates a few LSBs.  The one-time
public CKKS context is about 36 MB with `[60,24,60]`; the server must not add a
large finalize share as plaintext.  It encrypts its own share and adds the two
ciphertexts.

The client and server now call `run_party_batch`, so the 648 base OTs are
sent as ~82 channel messages per hybrid layer instead of scaling the message
count with the element count.  The scalar `run_party` path is kept only for
the legacy self-test.

## 6. Two-machine run

Desktop:

```powershell
python analysis\pi_server.py --host 0.0.0.0 --port 9101 --dims 64,32,10 --backend tenseal
```

The server exits after one query by default; add `--serve` to keep it listening
for repeated connections and persistent sessions.

Pi:

```bash
python pi_client.py --host 192.168.1.20 --port 9101 --backend tenseal \
  --out results/pi_hybrid.json
```

For a persistent session with one key generation and one context upload:

```bash
python pi_client.py --host 192.168.1.20 --port 9101 --backend tenseal \
  --input-seed 0 --repeat 3 --reuse-context \
  --out results/pi_hybrid_warm.json --verbose
```

The first query includes key generation and context upload. Later queries report
`reused_context=true` and only transfer the input/MPC ciphertexts.

Open TCP 9101 on the desktop firewall for the LAN.  For a first connectivity
check you can use `--backend stub` on both sides, but that traffic contains
plaintext activations — use it only on a trusted link, then switch back.

## 7. Benchmark

Desktop loopback benchmark (same protocol, both parties in one process):

```bash
python pi_bench.py --backend tenseal --dims 64,32,10 --repeat 3 \
  --out results/pi_hybrid_B.json
```

On the Pi, the meaningful end-to-end numbers are the client-side ones from
`pi_client.py --out ...`: keygen, encrypt, decrypt per layer, MPC time, bytes
sent/received.  `bench_pi.py` is a separate client-only HE microbenchmark and
uses profile B parameters; label it separately from the Pi-demo end-to-end
result.

Measured on Pi 5 with `64,32,10` and TenSEAL 0.3.18:

- cold start: 10.452 ± 0.092 s;
- setup: keygen 0.374 s, context upload 7.220 s;
- warm queries: 1.742–1.973 s each;
- post-setup upload: about 0.806 MB per query.

What to report:

- keygen seconds and public/full context size (one-time cost),
- encrypt/decrypt milliseconds,
- MPC seconds and base-OT count for the client,
- bytes up/down,
- device, OS, Python/TenSEAL versions, and whether the CPU throttled.

What not to report:

- accuracy from this random model,
- `--backend stub` numbers as a privacy result (they are a wiring simulation),
- Pi numbers without noting thermal throttling/power supply.

Scaling intuition: the client still pays `2*k_compare-1 = 27` base OTs per
activation element.  A CIFAR-sized layer with 1328 elements therefore costs
~35,800 OTs on the Pi.  With `run_party_batch`, those OTs are packed into
~82 channel messages / ~41 round trips per layer, so the Pi's CPU time (not
the round-trip count) dominates.  Measure it; do not extrapolate a paper
number from the desktop.

For reference, the desktop measured in this workspace (bundled Python 3.12,
`cryptography` 50.0.2) spends ~22 ms per activation element on the MPC; a
32-element hidden layer took 0.69 s.  Expect the Pi to be substantially slower.

## 8. Security notes

- The client never sends its secret key; `public_context_bytes()` serializes
  only public + Galois + relin keys.
- Each ReLU uses a fresh OS-CSPRNG mask.  Never reuse `r` across layers or
  queries; that is the M1 rule and reusing it leaks `z`.
- The Pi demo keeps sigma=30, but its one-multiplication CKKS chain is valid
  only because every hidden layer ends in a hybrid ReLU.  It assumes every
  pre-ReLU value is below 32 (the 14-bit comparison window).  A real model must
  clip or rescale values before the mask is drawn.
- The TCP channel is unauthenticated.  Run it over an SSH tunnel or a trusted
  LAN for the demo; a deployment needs mutual authentication.
- The threat model remains semi-honest, single-server, two non-colluding
  parties.  The stub backend provides no security at all.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named tenseal` | No ARM wheel; see section 3 |
| keygen takes minutes / OOM | `N=16384` is heavy; close other apps, use a Pi 5 8 GB |
| `backend mismatch` | Start both sides with the same `--backend` |
| `input dimension ... does not match` | Client input size must equal the server's first dim |
| `expected b'CTXT', got ...` | One side crashed mid-MPC; check the server console first |
| hangs after `SPEC` | Client and server disagree on backend or the port is blocked |
| `logits err ... FAIL` | Check `params_pi.py`; do not silently switch back to profile B |

