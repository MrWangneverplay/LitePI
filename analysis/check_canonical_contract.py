"""Static contract check for 实验方案 §3.2.1 and §5.2 geometries."""

from pathlib import Path

from canonical_net import CANONICAL_SPECS, choose_checkpoint


def _next_pow2(value: int) -> int:
    return 1 << max(1, (value - 1).bit_length())


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    expected = {
        "mnist": (256, 64, 4096, 17_298),
        "cifar10": (1200, 128, 3200, 155_330),
        "uci_har": (756, 128, 2016, 98_006),
    }
    for name, spec in CANONICAL_SPECS.items():
        n1, n2, slots, params = expected[name]
        kernel = (3, 9) if name == "uci_har" else spec.im2col_kernel
        flat = spec.conv_channels * spec.windows
        encoded = spec.windows * _next_pow2(kernel[0] * kernel[1])
        parameter_count = (
            spec.conv_channels * (kernel[0] * kernel[1] + 1)
            + spec.conv_flat * spec.hidden + spec.hidden
            + spec.hidden * spec.classes + spec.classes
        )
        assert (spec.conv_flat, spec.hidden, encoded, parameter_count) == (
            n1, n2, slots, params
        ), (name, spec, encoded, parameter_count)
        assert flat == spec.conv_flat
        for schedule in ((0, 0), (0, 1), (1, 0), (1, 1)):
            assert choose_checkpoint(root, name, schedule, 73).exists()
        print(
            f"PASS  {name}: Conv -> {n1} -> {n2} -> {spec.classes}, "
            f"slots={slots}, params={params}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
