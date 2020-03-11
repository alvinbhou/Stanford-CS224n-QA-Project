import torch
import torch.nn as nn


class EnsembleQA(nn.Module):
    '''
    EnsembleQA QA model

    Parameters
    ----------
    model_type : str, optional
        The pretrained-model type
    output_hidden : boolean, optional
        Whether the model output the hidden states
    '''

    def __init__(self, n_models=3):
        super().__init__()
        self.n_models = n_models
        self.weights = torch.tensor([1] * n_models, requires_grad=True)

    def forward(self, predict_start_logits, predict_end_logits, start_positions=None, end_positions=None):
        # Training mode
        if start_positions is not None and end_positions is not None:
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
