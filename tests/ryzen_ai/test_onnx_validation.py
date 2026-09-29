import sys
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from validate_onnx import normalize_constant_shapes, validate_model


class OnnxValidationTests(unittest.TestCase):
    def model(self):
        tensor = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
        node = helper.make_node("Constant", [], ["axes"], value=tensor)
        output = helper.make_tensor_value_info("axes", TensorProto.INT64, [])
        graph = helper.make_graph([node], "constant-rank", [], [output])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])

    def validate(self, model):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            onnx.save(model, path)
            validate_model(path)

    def test_bad_rank_is_rejected(self):
        with self.assertRaises(onnx.shape_inference.InferenceError):
            self.validate(self.model())

    def test_normalization_preserves_tensor(self):
        model = self.model()
        original_tensor = model.graph.node[0].attribute[0].t.SerializeToString()
        self.assertEqual(normalize_constant_shapes(model), 1)
        self.assertEqual(normalize_constant_shapes(model), 0)
        self.assertEqual(model.graph.node[0].attribute[0].t.SerializeToString(), original_tensor)
        self.validate(model)

    def test_undefined_output_is_rejected(self):
        model = self.model()
        normalize_constant_shapes(model)
        model.graph.output[0].name = "missing"
        with self.assertRaises(onnx.checker.ValidationError):
            self.validate(model)


if __name__ == "__main__":
    unittest.main()
