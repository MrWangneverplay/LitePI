# LitePI Strict Protocol Compliance

The full experiment protocol is the authoritative target:

- Python 3.11 runtime
- pybind11 C++ backend
- SCI/CrypTFlow2 additive sharing with silent OT/VOLE
- 128-bit CKKS parameter validation with the SEAL/TenSEAL estimator
- online hybrid ReLU interaction counted as one round trip per hybrid layer

The Python P-256 OT implementation in this repository is a reference
implementation.  It is useful for correctness and protocol debugging, but it
does not satisfy the paper's final deployment contract.

Run:

```bash
python analysis/check_protocol_compliance.py
python analysis/check_protocol_compliance.py --json
```

Until the gate passes, Pi measurements must be labelled:

```text
reference-only / not paper-deployable
```

## Known gaps

| Requirement | Current repository |
|---|---|
| Python 3.11 | Pi currently runs Python 3.13 |
| C++ pybind11 backend | Not implemented |
| SCI/CrypTFlow2 silent OT/VOLE | Not integrated; custom P-256 Simplest OT is used |
| N >= 65536 and estimator artifact | Current Pi demo uses N=16384 and no estimator artifact |
| r_h = 1 per hybrid layer | Current batched OT path uses multiple channel phases/round trips |
| Client cost target | §12 says `s=00` is 2 rounds and each hybrid adds 1 round; current Python path uses 82 messages/41 RTTs per hybrid layer |

M1/M3 are now implemented in `analysis/mask_session.py`: masks are derived from
the server-only session seed and `(query_id, layer_id)`, and a query-counter
gap or aborted session fails closed. Run:

```bash
python analysis/check_mask_session.py
```

The same protocol now has three working schedule configurations in
`pi_server.py`: `s=0` uses HE-P2 and has two online rounds, `s=1` uses
Hybrid-ReLU and adds one round, and mixed schedules use the same trace. Run:

```bash
python analysis/check_protocol_modes.py
```

The canonical server path now loads the three `train_hybrid.py` geometries and
uses the protocol-defined im2col/conv/FC trace:

```bash
python analysis/check_canonical_contract.py
```

M2 now uses per-layer training statistics. Export and load them with:

```bash
python analysis/export_mask_domains.py --dataset cifar10 --schedule 10
python analysis/pi_server.py --dataset cifar10 --schedule 10
```

Output mode B is implemented with a sequential secure tournament:

```bash
python analysis/check_output_modes.py
python analysis/secure_argmax.py
```

Schedule obfuscation from §7.2 is available through
`--obfuscate-schedule`; its regression test is included in
`check_protocol_modes.py`.

These gaps must be closed before quoting the current timing as the final
paper result.
