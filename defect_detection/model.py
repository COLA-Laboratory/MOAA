# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F


class CodeBERTModel(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(CodeBERTModel, self).__init__()
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
        code_tokens = self.tokenizer.tokenize(code)[:510]
        source_tokens = (
            [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        )
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length

        # predict the label
        source_ids = torch.Tensor([source_ids]).long().to("cuda")
        logits = self.forward(source_ids)[0]
        logits = logits.cpu().detach().numpy()
        prob = [1 - logits[0], logits[0]]
        return prob


class GraphCodeBERTModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GraphCodeBERTModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.query = 0

    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None):
        # embedding

        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = (
            nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        )
        nodes_to_token_mask = (
            nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        )
        avg_embeddings = torch.einsum(
            "abc,acd->abd", nodes_to_token_mask, inputs_embeddings
        )
        inputs_embeddings = (
            inputs_embeddings * (~nodes_mask)[:, :, None]
            + avg_embeddings * nodes_mask[:, :, None]
        )
        outputs = self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
        )[0]

        logits = outputs
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log(
                (1 - prob)[:, 0] + 1e-10
            ) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class CodeT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs["decoder_hidden_states"][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs["decoder_hidden_states"][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][
            :, 0, :
        ]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == "codet5":
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == "bart":
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == "roberta":
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
