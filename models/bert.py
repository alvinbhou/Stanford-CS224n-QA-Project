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

    def __init__(self, model_type='bert-large-uncased-whole-word-masking-finetuned-squad', output_hidden=True):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_type, output_hidden_states=output_hidden)
        self.model = BertForQuestionAnswering.from_pretrained(model_type, config=self.config)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertQA()
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    input_ids = tokenizer.encode(question, text)
    print('Input ids', input_ids)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    print('Token type ids', token_type_ids)  # 0 for question, 1 for answer
    inputs = torch.tensor([input_ids])
    print('Input shape', inputs.shape)  # [CLS] + 5 + [SEP] + 6 + [SEP] = 14
    start_scores, end_scores, hidden_states = model(inputs, token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
    print('Answer:', answer)
    print('Hidden states shape', hidden_states[0].shape)
    assert answer == "a nice puppet"
