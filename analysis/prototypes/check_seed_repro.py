"""Check that the multi-seed paths of the three ReLU twins really honour `seed`.

It runs each script's REAL main() with `--num-seeds N`, but with the data loaders,
the training loop and the JSON writer replaced by stubs, so no dataset, GPU or
training is needed.  The stub of train_one() only records a fingerprint of the
model it is handed (a checksum of the initial weights) after doing what the real
train_one() does first, torch.manual_seed(seed).

Three properties are checked per script:
  A. distinct   - the N seeds give N different initial models;
  B. repeatable - running the same command twice gives identical fingerprints;
  C. independent- the model for seed s does not depend on which seeds ran before
                  it: the fingerprint of seed 74 in `--seed 73 --num-seeds 3` equals
                  the one in `--seed 74 --num-seeds 2`.
Before the fix the model was built BEFORE torch.manual_seed(seed), so B and C fail
(the first model comes from unseeded RNG state, later ones from the previous seed's).

Run:
    python analysis/prototypes/check_seed_repro.py
    python analysis/prototypes/check_seed_repro.py --model-dir <dir with older copies of the three scripts>
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

SCRIPTS = ["cifar10_cnn_relu.py", "mnist_cnn_relu.py", "uci_har_cnn_relu.py"]


class _FakeLoader:
    dataset = []

    def __len__(self):
        return 0


def fingerprints(path: Path, seed: int, num_seeds: int) -> list[float]:
    """Run main() of the script at `path`; return the initial-weight checksum per seed."""
    spec = importlib.util.spec_from_file_location(path.stem + "_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    seen: list[float] = []

    def fake_train_one(model, train_loader, test_loader, device, epochs, lr, s):
        torch.manual_seed(s)                       # what the real train_one does first
        seen.append(round(float(sum(p.detach().double().sum() for p in model.parameters())), 10))
        return {}, {"train_acc_final": 0.0, "test_acc_final": 0.5,
                    "test_acc_best": 0.5, "train_time_s": 0.0}

    mod.load_data = lambda *a, **k: (_FakeLoader(), _FakeLoader())
    mod.train_one = fake_train_one
    mod.save_record = lambda *a, **k: Path("/dev/null")
    old_argv = sys.argv
    sys.argv = [path.name, "--seed", str(seed), "--num-seeds", str(num_seeds)]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parent.parent.parent / "model"))
    args = ap.parse_args()
    model_dir = Path(args.model_dir)

    bad = 0
    for name in SCRIPTS:
        path = model_dir / name
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            run1 = fingerprints(path, 73, 3)
            run2 = fingerprints(path, 73, 3)
            sub = fingerprints(path, 74, 2)
        a = len(set(run1)) == 3
        b = run1 == run2
        c = run1[1:] == sub
        bad += not (a and b and c)
        print(f"{name:22s} seeds 73,74,75 -> {run1}")
        print(f"{'':22s} A distinct={'OK' if a else 'FAIL'}  B repeatable={'OK' if b else 'FAIL'}"
              f"  C order-independent={'OK' if c else 'FAIL'}")
    print("\nRESULT:", "PASS" if bad == 0 else f"FAIL ({bad} script(s))")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
