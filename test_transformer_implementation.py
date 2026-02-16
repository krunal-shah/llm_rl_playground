from transformer_implementation import Transformer
from unittest.mock import patch
import torch
from torch.nn.functional import log_softmax

prob_020 = log_softmax(torch.tensor([0, 0, 2]), dtype=torch.float32)[0]
prob_100 = log_softmax(torch.tensor([1, 0, 0]), dtype=torch.float32)[0]
prob_200 = log_softmax(torch.tensor([2, 0, 0]), dtype=torch.float32)[0]


class TestTransformer:
    def setup_class(self):
        self.transformer = Transformer(vocab_size=15, max_length=5, eos_idx=2)

    def test_generate_simple(self):
        inputs = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        forward_outputs = [
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
                    [[0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                    [[0, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=torch.float32,
            ),
        ]

        with patch.object(self.transformer, "forward", side_effect=forward_outputs):
            outputs, probs = self.transformer.generate(inputs)
            assert torch.all(outputs == torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 1, 2], [1, 1, 2, 0, 0]]))
            assert torch.allclose(
                probs,
                torch.tensor(
                    [[1, 1, prob_200, prob_100, prob_200], [1, 1, 1, prob_100, prob_020], [1, prob_100, prob_200, 0, 0]]
                ),
            )

    def test_generate_simple_dynamic_data(self):
        inputs = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        forward_outputs = [
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
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=torch.float32,
            ),
        ]

        with patch.object(self.transformer, "forward", side_effect=forward_outputs):
            outputs, probs = self.transformer.generate(inputs)
            assert torch.all(outputs == torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 2, 0], [1, 1, 2, 0, 0]]))
            assert torch.allclose(
                probs,
                torch.tensor(
                    [[1, 1, prob_100, prob_200, prob_100], [1, 1, 1, prob_100, 0], [1, prob_100, prob_200, 0, 0]]
                ),
            )

    def test_generate_with_dynamic_batching_simple_generate_data(self):
        inputs = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        forward_outputs = [
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
        ]

        with patch.object(self.transformer, "forward", side_effect=forward_outputs):
            outputs, probs = self.transformer.generate_with_dynamic_batching(inputs)
            assert torch.all(outputs == torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 1, 2], [1, 1, 2, 0, 0]]))
            assert torch.allclose(
                probs,
                torch.tensor(
                    [[1, 1, prob_200, prob_100, prob_200], [1, 1, 1, prob_100, prob_020], [1, prob_100, prob_200, 0, 0]]
                ),
            )

    def test_generate_with_dynamic_batching_simple(self):
        inputs = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        forward_outputs = [
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
        ]

        with patch.object(self.transformer, "forward", side_effect=forward_outputs):
            outputs, probs = self.transformer.generate_with_dynamic_batching(inputs)
            assert torch.all(outputs == torch.tensor([[1, 1, 1, 1, 2], [1, 1, 1, 2, 0], [1, 1, 2, 0, 0]]))
            assert torch.allclose(
                probs,
                torch.tensor(
                    [[1, 1, prob_100, prob_200, prob_100], [1, 1, 1, prob_100, 0], [1, prob_100, prob_200, 0, 0]]
                ),
            )
