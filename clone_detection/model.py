# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1) * 2)
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

    def predict(self, orig_code1, orig_code2):
        self.eval()
        code_tokens1 = self.tokenizer.tokenize(orig_code1)[:510]
        code_tokens1 = (
            [self.tokenizer.cls_token] + code_tokens1 + [self.tokenizer.sep_token]
        )
        code_ids1 = self.tokenizer.convert_tokens_to_ids(code_tokens1)
        padding_length = 512 - len(code_ids1)
        code_ids1 += [self.tokenizer.pad_token_id] * padding_length

        code_tokens2 = self.tokenizer.tokenize(orig_code2)[:510]
        code_tokens2 = (
            [self.tokenizer.cls_token] + code_tokens2 + [self.tokenizer.sep_token]
        )
        code_ids2 = self.tokenizer.convert_tokens_to_ids(code_tokens2)
        padding_length = 512 - len(code_ids2)
        code_ids2 += [self.tokenizer.pad_token_id] * padding_length

        source_ids = code_ids1 + code_ids2
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
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(
        self,
        inputs_ids_1,
        position_idx_1,
        attn_mask_1,
        inputs_ids_2,
        position_idx_2,
        attn_mask_2,
        labels=None,
    ):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat(
            (inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1
        ).view(bs * 2, l)
        position_idx = torch.cat(
            (position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1
        ).view(bs * 2, l)
        attn_mask = torch.cat(
            (attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1
        ).view(bs * 2, l, l)

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

        outputs = self.encoder.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
            token_type_ids=position_idx.eq(-1).long(),
        )[0]
        logits = self.classifier(outputs)
        # shape: [batch_size, num_classes]
        prob = F.softmax(logits, dim=-1)
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
        self.classifier = RobertaClassificationHead(config)
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
