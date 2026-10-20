"""Bundle boundary tests without downloading weights or initializing a GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('local_text_encoder', Path(__file__).resolve().parents[1] / 'kimodo/model/local_text_encoder.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BundleTests(unittest.TestCase):
    def bundle(self, root, dimensions=4096):
        (root / 'config.json').write_text(json.dumps({'architectures': ['LlamaBiModel'], 'model_type': 'llama', 'hidden_size': dimensions, 'quantization_config': {'load_in_4bit': True}}))
        (root / 'model.safetensors').touch()
        (root / 'tokenizer_config.json').write_text('{}')
        adapter = root / 'supervised_adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{"peft_type":"LORA"}')
        (adapter / 'adapter_model.safetensors').touch()

    def test_complete_quantized_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root)
            self.assertEqual(module.inspect_bundle(root), (root.resolve(), root.resolve() / 'supervised_adapter', True))

    def test_rejects_incompatible_dimensions(self):
        for dimensions in (2048, 3072):
            with self.subTest(dimensions=dimensions), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.bundle(root, dimensions)
                with self.assertRaisesRegex(ValueError, '4096'):
                    module.inspect_bundle(root)

    def test_rejects_missing_supervised_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root)
            (root / 'supervised_adapter/adapter_model.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'supervised'):
                module.inspect_bundle(root)


if __name__ == '__main__':
    unittest.main()
