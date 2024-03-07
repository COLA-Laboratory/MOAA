# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log(
                (1 - prob)[:, 0] + 1e-10
            ) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def predict(self, code):
        self.eval()
        # encoding the source code
        code_tokens = self.tokenizer.tokenize(code)[: self.args.block_size - 2]
        source_tokens = (
            [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        )
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.args.block_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length

        # predict the label
        source_ids = torch.Tensor([source_ids]).long().to("cuda")
        logits = self.forward(source_ids)[0]
        logits = logits.cpu().detach().numpy()
        prob = [1 - logits[0], logits[0]]
        return prob
