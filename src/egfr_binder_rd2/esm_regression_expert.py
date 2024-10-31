from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from evo_prot_grad.experts.base_experts import AttributeExpert
import evo_prot_grad.common.embeddings as embeddings
import evo_prot_grad.common.utils as utils

class EsmRegressionExpert(AttributeExpert):
    """Expert class for ESM-style tokenization with regression output."""

    def __init__(
        self,
        temperature: float,
        model: nn.Module,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "cpu",
    ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

        # Create a vocab dictionary that maps tokens to their indices
        vocab = {token: idx for idx, token in enumerate(tokenizer.get_vocab())}

        super().__init__(
            temperature=temperature,
            model=model,
            scoring_strategy="attribute_value",
            device=device,
            tokenizer=None,
        )

        self.tokenizer = tokenizer
        self.model.esm_model.esm.embeddings.word_embeddings = (
            embeddings.OneHotEmbedding(model.esm_model.esm.embeddings.word_embeddings)
        )

        # Create the expert_to_canonical_order tensor
        self.expert_to_canonical_order = utils.expert_alphabet_to_canonical(
            list(vocab.keys()), self.device
        )

    def _get_last_one_hots(self) -> torch.Tensor:
        return self.model.esm_model.esm.embeddings.word_embeddings.one_hots

    def tokenize(self, inputs: List[str]):
        return self.tokenizer(
            inputs, add_special_tokens=True, padding=True, return_tensors="pt"
        ).to(self.device)

    def get_model_output(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenize(inputs)
        model_output = self.model(batch)

        regression_output = model_output["predictions"]
        regression_output = regression_output.view(-1)

        oh = self._get_last_one_hots()
        return oh, regression_output

    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_oh, regression_output = self.get_model_output(inputs)
        score = self.variant_scoring(x_oh, regression_output, self._wt_oh)
        return x_oh, score


def build(**kwargs):
    """Builds an EsmRegressionExpert."""
    return EsmRegressionExpert(**kwargs)