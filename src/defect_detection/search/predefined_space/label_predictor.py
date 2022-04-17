import torch

import torch.nn.functional as F

def ce_loss_func(std_logits, tea_logits, labels, alpha=0.9, temperature=2.0):
    labels = labels.long()

    loss = F.cross_entropy(std_logits, (1 - labels))

    ce_loss = F.kl_div(F.log_softmax(std_logits/temperature), F.softmax(tea_logits/temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return alpha * loss + (1. - alpha) * ce_loss

def predict(hidden):
