import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CodeBERTModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeBERTModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e6))


class GraphCodeBERTModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GraphCodeBERTModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None):

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

        outputs = self.encoder.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
        )[0]

        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class CodeT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 66)
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

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)
        vec = self.get_t5_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
