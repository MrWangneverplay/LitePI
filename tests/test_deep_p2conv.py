"""独立 B1 无 SEAL 测试；不加载模型、不运行 runner/MPC 或 native binding。"""

from __future__ import annotations

import ast
import copy
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

from deep_hybrid import (
    DepthBudgetExceeded,
    P2BeforeConvUnsupported,
    SCHEDULE_LAYERS,
    im2col_ct,
    im2col_rows,
    parse_schedule,
    schedule_depth,
    schedule_ops,
    to_blocks,
    validate_schedule,
)
from planner import Layer, required_depth


class FakeCiphertext:
    def __init__(self, values, depth=0):
        self.values = np.asarray(values, dtype=np.float64).copy()
        self.depth = depth

    @property
    def size(self):
        return self.values.size


class FakeCtx:
    """只模拟真实 binding 的 COO、连续分块和一层 rescale，不估计旋转。"""

    def __init__(self, slots):
        self.slots = slots
        self.calls = []

    def slot_capacity(self):
        return self.slots

    def linear_sparse(self, blocks, in_idx, out_idx, values, out_len, threads=1):
        if (in_idx.dtype != np.int64 or out_idx.dtype != np.int64
                or values.dtype != np.float64):
            raise AssertionError("COO dtype 与真实绑定不一致")
        if not (len(in_idx) == len(out_idx) == len(values)) or not len(values):
            raise ValueError("linear_sparse: bad COO arrays")
        source = np.zeros(len(blocks) * self.slots)
        for index, block in enumerate(blocks):
            source[index * self.slots:index * self.slots + block.size] = block.values
        if (np.any(in_idx < 0) or np.any(in_idx >= source.size)
                or np.any(out_idx < 0) or np.any(out_idx >= out_len)):
            raise ValueError("linear_sparse: index out of range")
        if len({block.depth for block in blocks}) != 1:
            raise AssertionError("输入块深度不一致")
        self.calls.append((in_idx.copy(), out_idx.copy(), values.copy(), out_len, threads))
        result = np.zeros(out_len)
        np.add.at(result, out_idx, source[in_idx] * values)
        mapped = set((out_idx[values != 0] // self.slots).tolist())
        output = []
        for start in range(0, out_len, self.slots):
            if start // self.slots not in mapped:
                raise ValueError("linear_sparse: an output block is all zero")
            output.append(FakeCiphertext(result[start:start + self.slots],
                                         blocks[0].depth + 1))
        return output


def explicit_ops():
    # 仅为本测试明确给出的 op 元数据，不宣称任何现有模型具有此结构。
    ops = []
    for name in SCHEDULE_LAYERS:
        ops.extend([{"op": "fc" if name == "fc1" else "conv", "name": name},
                    {"op": "act"}])
        if name in ("conv1", "conv4"):
            ops.append({"op": "pool", "k": 2})
    ops.append({"op": "fc", "name": "logits"})
    return ops


def deepnet_protocol_depth(ops, bits):
    # 提取真实的纯 Python protocol_ops，避免导入 DeepNet 时加载 SCI/HE。
    tree = ast.parse((ANALYSIS / "deep_net.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DeepNet")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "protocol_ops")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ANALYSIS / "deep_net.py"),
                 "exec"), namespace)
    ops = copy.deepcopy(ops)
    for index, op in enumerate(op for op in ops if op["op"] == "act"):
        op["k"] = index
    network = SimpleNamespace(ops=ops, schedule=bits)
    trace = namespace["protocol_ops"](network)
    depth = maximum = 0
    for kind, value in trace:
        if kind == "fresh":
            depth = 0
        elif kind == "rescale":
            depth += value
            maximum = max(maximum, depth)
    return maximum


class ScheduleTests(unittest.TestCase):
    def test_strict_five_bits(self):
        for bits in itertools.product((0, 1), repeat=5):
            self.assertEqual(parse_schedule("".join(map(str, bits))), bits)
            self.assertEqual(parse_schedule(list(bits)), bits)
        self.assertEqual(parse_schedule([np.int64(0)] * 5), (0,) * 5)
        for bad in (None, 0, "", "0000", "000000", "s00000", " 00000", "00000\n",
                    "00002", "０１０１０", [0, 1], [0] * 6, [False] * 5, [0.0] * 5,
                    ["0"] * 5, [0, 0, 0, 0, -1], iter([0] * 5)):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                parse_schedule(bad)

    def test_explicit_metadata_and_no_mutation(self):
        ops = explicit_ops()
        original = copy.deepcopy(ops)
        layers = schedule_ops("10101", ops)
        self.assertEqual([layer.name for layer in layers], list(SCHEDULE_LAYERS))
        self.assertEqual([layer.bit for layer in layers], [1, 0, 1, 0, 1])
        self.assertEqual(layers[0].linear_index, 0)
        self.assertEqual(layers[0].activation_index, 1)
        self.assertEqual(layers[0].pool_indices, (2,))
        self.assertEqual(layers[0].next_linear_index, 3)
        self.assertEqual(layers[0].next_linear_name, "conv2")
        self.assertEqual(layers[0].next_linear_op, "conv")
        self.assertEqual(layers[3].next_linear_op, "fc")
        self.assertEqual(layers[4].next_linear_name, "logits")
        self.assertEqual(ops, original)

    def test_reject_missing_or_different_model_metadata(self):
        bad_ops = [[], [{"op": "conv"}], explicit_ops()[:-1],
                   explicit_ops() + [{"op": "act"}]]
        unnamed = explicit_ops()
        del unnamed[0]["name"]
        bad_ops.append(unnamed)
        wrong_name = explicit_ops()
        wrong_name[3]["name"] = "conv3"
        bad_ops.append(wrong_name)
        wrong_kind = explicit_ops()
        wrong_kind[3]["op"] = "fc"
        bad_ops.append(wrong_kind)
        missing_act = explicit_ops()
        missing_act[1] = {"op": "pool", "k": 2}
        bad_ops.append(missing_act)
        unknown = explicit_ops()
        unknown[2] = {"op": "flatten"}
        bad_ops.append(unknown)
        for ops in bad_ops:
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                schedule_ops("11111", ops)

    def test_all_schedules_match_required_depth_and_deepnet_trace(self):
        ops = explicit_ops()
        planner_layers = [Layer(name, elements=1) for name in SCHEDULE_LAYERS]
        for bits in itertools.product((0, 1), repeat=5):
            with self.subTest(bits=bits):
                need = schedule_depth(bits, ops)
                self.assertEqual(need, required_depth(bits, planner_layers))
                self.assertEqual(need, deepnet_protocol_depth(ops, bits))
                validate_schedule(bits, ops, need, p2_before_conv=True)
                with self.assertRaises(DepthBudgetExceeded):
                    validate_schedule(bits, ops, need - 1, p2_before_conv=True)

    def test_depth_budget_six_rejects_all_p2_before_capability_check(self):
        self.assertEqual(schedule_depth("00000", explicit_ops()), 11)
        for enabled in (False, True):
            with self.assertRaisesRegex(DepthBudgetExceeded, "11.*6"):
                validate_schedule("00000", explicit_ops(), p2_before_conv=enabled)

    def test_default_capability_only_applies_to_new_validator(self):
        ops = explicit_ops()
        with self.assertRaisesRegex(P2BeforeConvUnsupported, "conv1.*conv2"):
            validate_schedule("01111", ops)
        validate_schedule("01111", ops, p2_before_conv=True)
        validate_schedule("11100", ops)
        self.assertEqual(deepnet_protocol_depth(ops, (0, 1, 1, 1, 1)), 3)
        # 显式元数据改变输出头类型后，也必须按真实 next op 拒绝，不能猜 fc 后无 conv。
        ops[-1]["op"] = "conv"
        with self.assertRaisesRegex(P2BeforeConvUnsupported, "fc1.*logits"):
            validate_schedule("11110", ops)

    def test_im2col_extra_level_not_hidden(self):
        ops = explicit_ops()
        self.assertEqual(schedule_depth("00000", ops, p2_before_conv_extra_depth=1), 14)
        self.assertEqual(schedule_depth("01111", ops, p2_before_conv_extra_depth=1), 4)
        with self.assertRaises(DepthBudgetExceeded):
            validate_schedule("01111", ops, 3, p2_before_conv=True,
                              p2_before_conv_extra_depth=1)
        validate_schedule("01111", ops, 4, p2_before_conv=True,
                          p2_before_conv_extra_depth=1)
        self.assertEqual(schedule_depth("11100", ops, p2_before_conv_extra_depth=1), 5)
        for bad in (-1, 2, 1.0, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                schedule_depth("11111", ops, p2_before_conv_extra_depth=bad)
        for bad in (-1, 6.0, True):
            with self.subTest(budget=bad), self.assertRaises(ValueError):
                validate_schedule("11111", ops, bad)
        with self.assertRaises(ValueError):
            validate_schedule("11111", ops, p2_before_conv="false")


class Im2colTests(unittest.TestCase):
    def check_case(self, shape, k, stride, pad, slots, padded_input=False):
        x = (np.arange(np.prod(shape), dtype=np.float64) + 1).reshape(shape)
        flat = x.ravel()
        blocks = []
        for start in range(0, flat.size, slots):
            values = flat[start:start + slots]
            if padded_input and values.size < slots:
                # 未使用输入尾槽即使非零，也不能泄漏进有效 im2col 或输出补零区。
                values = np.pad(values, (0, slots - values.size), constant_values=999)
            blocks.append(FakeCiphertext(values, depth=2))
        snapshots = [block.values.copy() for block in blocks]
        ctx = FakeCtx(slots)
        ct = blocks[0] if len(blocks) == 1 else blocks
        result = im2col_ct(ct, *shape, k, stride, pad, ctx)
        rows, _, _ = im2col_rows(x, k, stride, pad)
        reference = to_blocks(rows, slots)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(reference))
        for got, expected in zip(result, reference):
            np.testing.assert_array_equal(got.values, expected)
            self.assertEqual(got.size, slots)
            self.assertEqual(got.depth, 3)
        for block, snapshot in zip(blocks, snapshots):
            np.testing.assert_array_equal(block.values, snapshot)
            self.assertEqual(block.depth, 2)
        self.assertEqual(result.extra_depth, 1)
        self.assertEqual(result.linear_sparse_calls, len(ctx.calls))
        self.assertIsNone(result.rotations)
        self.assertEqual(ctx.calls[0][3], len(reference) * slots)
        self.assertEqual(ctx.calls[0][4], 1)
        return result, ctx

    def test_small_tensors_cross_blocks_stride_and_padding(self):
        cases = [((1, 3, 3), 2, 1, 0, 8), ((3, 4, 4), 3, 2, 1, 8),
                 ((2, 2, 2), 3, 1, 1, 8), ((1, 3, 5), 2, 2, 0, 8),
                 ((1, 1, 1), 1, 1, 0, 8)]
        for case in cases:
            for padded in (False, True):
                with self.subTest(case=case, padded=padded):
                    self.check_case(*case, padded_input=padded)

    def test_known_tap_major_values_and_zero_padding(self):
        ctx = FakeCtx(8)
        result = im2col_ct(FakeCiphertext([1, 2, 3, 4]), 1, 2, 2, 2, 1, 0, ctx)
        np.testing.assert_array_equal(result[0].values, [1, 2, 3, 4, 0, 0, 0, 0])
        result, ctx = self.check_case((3, 4, 4), 3, 2, 1, 8)
        ii, oi, values, _, _ = ctx.calls[0]
        self.assertGreater(ii.max(), 8)
        self.assertGreater(oi.max(), 8)
        self.assertTrue(np.any(ii // 8 != oi // 8))
        self.assertEqual(result[0].values[0], 0)
        np.testing.assert_array_equal(values, np.ones(values.size))
        np.testing.assert_array_equal(result[-1].values[-4:], np.zeros(4))

    def test_geometry_and_block_validation_before_sparse_call(self):
        bad_geometry = [(0, 2, 2, 1, 1, 0), (1, -2, 2, 1, 1, 0),
                        (1, 2, 2, 0, 1, 0), (1, 2, 2, 1, 0, 0),
                        (1, 2, 2, 1, 1, -1), (True, 2, 2, 1, 1, 0),
                        (1, 2, 2, 1.0, 1, 0), (1, 2, 2, 5, 1, 0),
                        (1, 3, 3, 1, 1, 0)]
        for geometry in bad_geometry:
            ctx = FakeCtx(8)
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                im2col_ct(FakeCiphertext(np.ones(4)), *geometry, ctx)
            self.assertFalse(ctx.calls)
        for blocks in ([], [FakeCiphertext(np.ones(4))] * 2,
                       [FakeCiphertext(np.ones(3))],
                       [FakeCiphertext(np.ones(5))]):
            ctx = FakeCtx(8)
            with self.subTest(sizes=[b.size for b in blocks]), self.assertRaises(ValueError):
                im2col_ct(blocks, 1, 2, 2, 1, 1, 0, ctx)
            self.assertFalse(ctx.calls)
        ctx = FakeCtx(8)
        with self.assertRaises(ValueError):
            im2col_ct([FakeCiphertext(np.ones(7)), FakeCiphertext(np.ones(1))],
                      1, 3, 3, 2, 1, 0, ctx)
        self.assertFalse(ctx.calls)

    def test_binding_zero_only_block_limit_is_explicit(self):
        ctx = FakeCtx(2)
        with self.assertRaisesRegex(ValueError, "纯零输出 block"):
            im2col_ct(FakeCiphertext([1]), 1, 1, 1, 3, 1, 1, ctx)
        self.assertFalse(ctx.calls)

    def test_no_native_seal_import(self):
        self.assertNotIn("litepi_seal", sys.modules)
        self.assertNotIn("tenseal", sys.modules)


if __name__ == "__main__":
    unittest.main()
