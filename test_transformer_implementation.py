import pytest
from transformer_implementation import Transformer
from unittest.mock import patch
import torch


class TestTransformer:
    def setup_class(self):
        self.transformer = Transformer(vocab_size=15, max_length=5, eos_idx=2)

    def test_generate_simple(self):
        inputs = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        forward_outputs = [
            torch.tensor(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
                    [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            ),
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                    [[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            ),
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                    [[0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                ]
            ),
        ]

        with patch.object(self.transformer, "forward", side_effect=forward_outputs):
            outputs = self.transformer.generate(inputs)
            assert torch.all(outputs == torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 2, 0], [1, 1, 2, 0, 0]]))

    def test_generate_with_dynamic_batching_simple(self):
        inputs = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        forward_outputs = [
            torch.tensor(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
                    [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            ),
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            ),
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
                ]
            ),
        ]

        with patch.object(self.transformer, "forward", side_effect=forward_outputs):
            outputs = self.transformer.generate_with_dynamic_batching(inputs)
            assert torch.all(outputs == torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 2, 0], [1, 1, 2, 0, 0]]))
