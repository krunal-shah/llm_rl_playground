from transformer_implementation import Transformer
from unittest.mock import patch
import torch
from torch.nn.functional import log_softmax

PROB_020 = log_softmax(torch.tensor([0, 0, 2]), dtype=torch.float32)[0]
PROB_100 = log_softmax(torch.tensor([1, 0, 0]), dtype=torch.float32)[0]
PROB_200 = log_softmax(torch.tensor([2, 0, 0]), dtype=torch.float32)[0]

INPUT_DATA = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])

TEST_CASE_1 = {
    "forwards": [
        torch.tensor(
            [
                [[0, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 2], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
    ],
    "preds": torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 1, 1], [1, 1, 2, 0, 0]]),
    "pred_probs": torch.tensor(
        [[1, 1, PROB_200, PROB_100, PROB_200], [1, 1, 1, PROB_100, PROB_200], [1, PROB_100, PROB_200, 0, 0]]
    ),
}

TEST_CASE_2 = {
    "forwards": [
        torch.tensor(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
    ],
    "preds": torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 2, 0], [1, 1, 2, 0, 0]]),
    "pred_probs": torch.tensor(
        [[1, 1, PROB_100, PROB_200, PROB_100], [1, 1, 1, PROB_100, 0], [1, PROB_100, PROB_200, 0, 0]]
    ),
}


class TestTransformer:
    def setup_class(self):
        self.transformer = Transformer(vocab_size=15, max_length=5, eos_idx=2)

    def test_generate_test_case_1(self):
        with patch.object(self.transformer, "forward", side_effect=TEST_CASE_1["forwards"]):
            outputs = self.transformer.generate(INPUT_DATA, require_probs=True, sampling="greedy")
            assert torch.all(outputs["preds"] == TEST_CASE_1["preds"])
            assert torch.allclose(outputs["pred_probs"], TEST_CASE_1["pred_probs"])

    def test_generate_test_case_2(self):
        with patch.object(self.transformer, "forward", side_effect=TEST_CASE_2["forwards"]):
            outputs = self.transformer.generate(INPUT_DATA, require_probs=True, sampling="greedy")
            assert torch.all(outputs["preds"] == TEST_CASE_2["preds"])
            assert torch.allclose(outputs["pred_probs"], TEST_CASE_2["pred_probs"])
