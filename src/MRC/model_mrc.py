from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm.modeling_xlm import XLMModel
from torch import nn
from torch.nn import CrossEntropyLoss
import logging
import torch

logger = logging.getLogger(__name__)


def get_seq_encoder(config):
    if config['model_type'] == 'bert':
        return BertModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlmr':
        return XLMRobertaModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlm':
        return XLMModel.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])


class TransformerQa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.seq_encoder = get_seq_encoder(config)
        self.seq_config = self.seq_encoder.config

        self.qa_outputs = nn.Linear(self.config.hidden_size + 200, 2)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None, dis_model=None):
        conf, batch_size, seq_len = self.config, input_ids.shape[0], input_ids.shape[1]
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                  'output_attentions': False, 'output_hidden_states': False, 'return_dict': False}
        if conf['model_type'] != 'mt5':
            inputs['token_type_ids'] = token_type_ids
        outputs = self.seq_encoder(**inputs)
        sequence_output = outputs[0]  # [batch size, seq len, seq hidden]
        sequence_output = torch.cat([sequence_output, dis_model.mean1(sequence_output)], -1)

        # Get QA logits on final output
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)  # [batch size, seq len]

        # Get loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
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

        output = start_logits, end_logits
        return (total_loss, output) if total_loss is not None else output
