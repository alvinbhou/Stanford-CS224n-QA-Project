import torch
import torch.nn as nn
from transformers import BertForQuestionAnswering, BertConfig, BertTokenizer


class BertQA(nn.Module):
    '''
    BERT QA model

    Parameters
    ----------
    model_type : str, optional
        The pretrained-model type
    output_hidden : boolean, optional
        Whether the model output the hidden states
    '''

    def __init__(self, model_type='bert-base-uncased', do_cls=True):
        super().__init__()
        self.do_cls = do_cls
        self.config = BertConfig.from_pretrained(model_type, output_hidden_states=do_cls)
        self.model = BertForQuestionAnswering.from_pretrained(model_type, config=self.config)
        self.fc_cls = nn.Linear(1024, 2) if 'large' in model_type else nn.Linear(768, 2)
        self.criterion_cls = nn.CrossEntropyLoss()

    def forward(self, input_ids, y_cls=None, **kwargs):
        if self.training and self.do_cls:
            assert y_cls is not None, 'No label for y_cls!'
            outputs = self.model(input_ids, **kwargs)
            loss, start_scores, end_scores, hidden_states = outputs[0], outputs[1], outputs[2], outputs[3]  # loss: original QA loss from model
            output_embeddings = hidden_states[-1]                                           # (batch_size, sequence_length, hidden_size)
            logits_cls = self.fc_cls(output_embeddings.permute([1, 0, 2])[0]).squeeze(dim=0)     # (1, batch_size, hidden_size) -> (batch_size, 2)
            loss_cls = self.criterion_cls(logits_cls, y_cls)
            return (loss, loss_cls), start_scores, end_scores, logits_cls

        elif not self.training and self.do_cls:                                             # Eval mode
            start_logits, end_logits, hidden_states = self.model(input_ids, **kwargs)
            output_embeddings = hidden_states[-1]                                           # (batch_size, sequence_length, hidden_size)
            logits_cls = self.fc_cls(output_embeddings.permute([1, 0, 2])[0]).squeeze()     # (1, batch_size, hidden_size) -> (batch_size, 2)
            return start_logits, end_logits, logits_cls
        else:
            return self.model(input_ids, **kwargs)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertQA(do_cls=False, model_type='bert-large-uncased-whole-word-masking-finetuned-squad')
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    input_ids = tokenizer.encode(question, text)
    print('Input ids', input_ids)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    print('Token type ids', token_type_ids)  # 0 for question, 1 for answer
    inputs = torch.tensor([input_ids])
    print('Input shape', inputs.shape)  # [CLS] + 5 + [SEP] + 6 + [SEP] = 14
    start_scores, end_scores = model(inputs, token_type_ids=torch.tensor([token_type_ids]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
    print('Answer:', answer)
    # print('Hidden states shape', hidden_states[0].shape)
    assert answer == "a nice puppet"
