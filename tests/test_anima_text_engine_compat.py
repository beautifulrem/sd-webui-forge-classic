"""Verify extension interfaces and encoder policy without model dependencies."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_engine():
    parsing = {}
    exec(compile((ROOT / "backend/text_processing/parsing.py").read_text(), "parsing.py", "exec"), parsing)
    parse = parsing["parse_prompt_attention"]
    opts = SimpleNamespace(emphasis="Original")
    namespace = {
        "opts": opts,
        "parsing": SimpleNamespace(parse_prompt_attention=parse),
        "emphasis": SimpleNamespace(
            get_current_option=lambda name: lambda: SimpleNamespace(name=name),
            uses_emphasis=lambda line: any(weight != 1 for _, weight in parse(line)),
        ),
        "SDClipModel": lambda model, **kwargs: SimpleNamespace(transformer=model, **kwargs),
        "INF": 99999999,
        "torch": SimpleNamespace(Tensor=object, tensor=lambda value, **kwargs: value, int="int"),
        "dynamic_args": SimpleNamespace(last_extra_generation_params={}),
    }
    tree = ast.parse((ROOT / "backend/text_processing/anima_engine.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.Assign))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "anima_engine.py", "exec"), namespace)
    return namespace, opts


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        encode = lambda text: [ord(char) for char in text if not char.isspace()]
        return {"input_ids": encode(texts) if isinstance(texts, str) else [encode(text) for text in texts]}


class AnimaTextCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.namespace, self.opts = load_engine()
        self.qwen = Tokenizer()
        self.t5 = Tokenizer()
        self.model = object()
        self.engine = self.namespace["Qwen06Engine"](self.model, self.qwen, self.t5)

    def test_legacy_import_and_raw_tokenizers_remain_available(self):
        self.assertIs(self.namespace["AnimaTextProcessingEngine"], type(self.engine))
        self.assertIs(self.engine.text_encoder, self.model)
        self.assertIs(self.engine.t5_tokenizer, self.t5)
        self.assertIs(self.engine.qwen_tokenizer, self.qwen)

    def test_negpip_weights_align_with_t5_tokens_and_eos(self):
        chunk, = self.engine.tokenize_line("a (b:-2) [c]")
        self.assertEqual(chunk.t5_tokens, [ord("a"), ord("b"), ord("c"), 1])
        self.assertEqual(chunk.t5_multipliers, [1, -2, 1 / 1.1, 1])
        self.assertEqual(chunk.qwen_tokens, [ord("a"), ord("b"), ord("c")])
        self.assertEqual(chunk.qwen_multipliers, [1, 1, 1])

    def test_nested_weights_preserve_remi_parser_semantics(self):
        chunk, = self.engine.tokenize_line("((a:2):3)")
        self.assertEqual(chunk.t5_multipliers, [6, 1])

    def test_freefuse_can_tokenize_concepts_without_special_tokens(self):
        chunk, = self.engine.tokenize_line("cat (dog:2)")
        ids = self.engine.t5_tokenizer(["dog"], truncation=False, add_special_tokens=False)["input_ids"][0]
        self.assertEqual(chunk.t5_tokens[3:6], ids)
        self.assertEqual(chunk.t5_tokens[-1], 1)
        self.assertEqual(self.t5.calls[-1][1], {"truncation": False, "add_special_tokens": False})

    def test_empty_prompt_keeps_qwen_padding_and_t5_eos(self):
        chunk, = self.engine.tokenize_line("")
        self.assertEqual(chunk.qwen_tokens, [151643])
        self.assertEqual(chunk.t5_tokens, [1])
        self.assertEqual(chunk.t5_multipliers, [1])

    def test_ignore_consumes_syntax_and_neutralizes_negative_weights(self):
        self.opts.emphasis = "Ignore"
        chunk, = self.engine.tokenize_line("(a:-2)")
        self.assertEqual(chunk.t5_tokens, [ord("a"), 1])
        self.assertEqual(chunk.t5_multipliers, [1, 1])

    def test_none_keeps_weight_syntax_literal(self):
        self.opts.emphasis = "None"
        chunk, = self.engine.tokenize_line("(a:-2)")
        self.assertEqual(chunk.t5_tokens, [ord(char) for char in "(a:-2)"] + [1])
        self.assertTrue(all(weight == 1 for weight in chunk.t5_multipliers))

    def test_prompt_cache_includes_t5_preprocessing_and_expires_per_call(self):
        calls = []
        self.engine.process_tokens = lambda tokens, weights: [object()]
        self.engine.anima_preprocess = lambda z, ids, weights: calls.append(ids) or object()
        output = self.engine(["cat", "cat", "dog"])
        self.assertEqual(len(output), 3)
        self.assertEqual(len(calls), 2)
        self.assertIs(output[0], output[1])
        self.assertIsNot(output[0], output[2])
        self.engine(["cat"])
        self.assertEqual(len(calls), 3)

    def test_common_encoder_uses_configured_device_with_offloaded_weights(self):
        tree = ast.parse((ROOT / "backend/text_processing/_comfy.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SDClipModel")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        selected = []
        output = SimpleNamespace(float=lambda: "conditioning")

        class Model:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

            def __call__(self, *args, **kwargs):
                self.kwargs = kwargs
                return output, None

        model = Model()
        clip = SimpleNamespace(
            transformer=model,
            process_tokens=lambda tokens, device: selected.append(device) or ("embeds", "mask", [1], []),
            enable_attention_masks=True,
            layer="last", layer_idx=None,
            layer_norm_hidden_state=False,
            zero_out_masked=False,
            return_projected_pooled=True,
            return_attention_masks=True,
        )
        namespace = {
            "memory_management": SimpleNamespace(text_encoder_device=lambda: "mps"),
            "torch": SimpleNamespace(float32="float32"),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "_comfy.py", "exec"), namespace)
        self.assertEqual(namespace["forward"](clip, [[42]]), ("conditioning", None, {"attention_mask": "mask"}))
        self.assertEqual(selected, ["mps"])
        self.assertEqual(model.kwargs["dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
