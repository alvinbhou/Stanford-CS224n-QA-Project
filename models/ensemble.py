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

    def __init__(self, n_models):
        super(EnsembleQA, self).__init__()
        self.n_models = n_models
        self.weights = nn.Parameter(torch.ones(n_models))

    def forward(self, predict_start_logits, predict_end_logits, start_positions=None, end_positions=None):

        start_logits = self.weights[:, None] * predict_start_logits
        start_logits = torch.sum(start_logits, dim=1)

        end_logits = self.weights[:, None] * predict_end_logits
        end_logits = torch.sum(end_logits, dim=1)

        outputs = (start_logits, end_logits)
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

        return outputs


class EnsembleStackQA(nn.Module):
    '''
    EnsembleQA QA model

    Parameters
    ----------
    n_models : int, optional
        Number of models to ensemble
    '''

    def __init__(self, n_models, seq_len=256):
        super(EnsembleStackQA, self).__init__()
        self.n_models = n_models
        self.seq_len = seq_len
        self.fc1_start = nn.Linear(n_models, 512)
        self.fc2_start = nn.Linear(512, 256)
        self.fc3_start = nn.Linear(256, 1)
        self.fc1_end = nn.Linear(n_models, 512)
        self.fc2_end = nn.Linear(512, 256)
        self.fc3_end = nn.Linear(256, 1)

    def forward(self, predict_start_logits, predict_end_logits, start_positions=None, end_positions=None):
        start_logits = self.fc1_start(predict_start_logits.transpose(2, 1))
        start_logits = self.fc2_start(start_logits)
        start_logits = self.fc3_start(start_logits).squeeze()  # (batch_size, 256)

        end_logits = self.fc1_end(predict_end_logits.transpose(2, 1))
        end_logits = self.fc2_end(end_logits)
        end_logits = self.fc3_end(end_logits).squeeze()  # (batch_size, 256)

        outputs = (start_logits, end_logits)
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

        return outputs
