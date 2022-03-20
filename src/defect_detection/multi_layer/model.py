import torch.nn as nn
import torch.nn.functional as F


class Roberta(nn.Module):
    def __init__(self, encoder):
        super(Roberta, self).__init__()
        self.encoder = encoder
        
    def forward(self, input_ids=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)
        logits = outputs[0]
        hidden_states = outputs[1][1:]
        hidden_states = [x[:, 0, :] for x in hidden_states]

        return logits, hidden_states


def distill_loss(std_logits, tea_logits, labels, alpha=0.6, temperature=2.0):
    labels = labels.long()

    loss = F.cross_entropy(std_logits, (1 - labels))

    ce_loss = F.kl_div(F.log_softmax(std_logits/temperature), F.softmax(tea_logits/temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return alpha * loss + (1.0 - alpha) * ce_loss

def patience_loss(std_logits, tea_logits, normalized_patience=False):
    if normalized_patience:
        std_logits = F.normalize(std_logits, p=2, dim=1)
        tea_logits = F.normalize(tea_logits, p=2, dim=1)
    mse_loss = F.mse_loss(std_logits, tea_logits, reduction="sum")
    mse_loss /= std_logits.size(0)

    return mse_loss

