import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class EnsembleQA(nn.Module):
    '''
    EnsembleQA QA model

    Parameters
    ----------
    n_models : int, optional
        Number of models to ensemble
    '''

    def __init__(self, n_models, device):
        super().__init__()
        self.n_models = n_models
        self.weights = torch.tensor([1.0] * n_models, dtype=torch.float, requires_grad=True).to(device)

    def forward(self, predict_start_logits, predict_end_logits, start_positions=None, end_positions=None):

        start_logits = self.weights[:, None] * predict_start_logits
        start_logits = torch.sum(start_logits, dim=1)

        end_logits = self.weights[:, None] * predict_end_logits
        end_logits = torch.sum(end_logits, dim=1)

        outputs = (start_logits, end_logits, self.weights)
        # Training mode
        if start_positions is not None and end_positions is not None:

            # Generate p_ij array, not used
            # n_exmaples, n_models, seq_len = predict_start_logits.shape
            # a = predict_start_logits.repeat(1, 1, seq_len).view(n_exmaples, n_models, seq_len, seq_len)
            # b = predict_end_logits.repeat(1, 1, seq_len).view(n_exmaples, n_models, seq_len, seq_len)
            # p_ij = a.transpose(3, 2) + b
            # print(p_ij.shape)

            # p_ij_inter = p_ij.view(n_exmaples, n_models, -1)  # (n_exmaples, n_models, seq_len * seq_len)
            # p_ij_inter = self.weights[:, None] * p_ij_inter  # (n_exmaples, n_models, seq_len * seq_len)
            # max_pij, max_pij_index = torch.max(p_ij_inter, dim=2)   # (n_exmaples, n_models)

            # predict_start_logits: (n_exmaples, n_models, seq_len)

            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs
