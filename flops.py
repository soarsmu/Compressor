"""Computes the flops needed for training/running transformer networks."""
"""Partial code is from https://github.com/google-research/electra/blob/master/flops_computation.py"""

import collections

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class TransformerHparams(object):
    """Computes the train/inference FLOPs for transformers."""

    def __init__(self, h=768, l=12, s=514, v=50265, i=3072, heads=12):
        self.h = h  # hidden size
        self.l = l  # number of layers
        self.s = s  # sequence length
        self.v = v  # vocab size
        self.e = h  # embedding size
        self.i = h * 4 if i is None else i  # intermediate size
        self.kqv = h
        self.heads = heads

    def get_block_flops(self):
        block_flops = dict(
            kqv=3 * 2 * self.h * self.kqv,
            kqv_bias=3 * self.kqv,
            attention_scores=2 * self.kqv * self.s,
            attn_softmax=SOFTMAX_FLOPS * self.s * self.heads,
            attention_dropout=DROPOUT_FLOPS * self.s * self.heads,
            attention_scale=self.s * self.heads,
            attention_weighted_avg_values=2 * self.h * self.s,
            attn_output=2 * self.h * self.h,
            attn_output_bias=self.h,
            attn_output_dropout=DROPOUT_FLOPS * self.h,
            attn_output_residual=self.h,
            attn_output_layer_norm=LAYER_NORM_FLOPS,
            intermediate=2 * self.h * self.i,
            intermediate_act=ACTIVATION_FLOPS * self.i,
            intermediate_bias=self.i,
            output=2 * self.h * self.i,
            output_bias=self.h,
            output_dropout=DROPOUT_FLOPS * self.h,
            output_residual=self.h,
            output_layer_norm=LAYER_NORM_FLOPS * self.h
        )
        return sum(block_flops.values()) * self.s

    def get_embedding_flops(self):
        """Get the forward-pass FLOPs the transformer inputs or output softmax."""
        embedding_flops = {}
        embedding_flops["main_multiply"] = 2 * self.e * self.v

        embedding_flops.update(dict(
            tok_type_and_position=2 * self.e * (self.s + 2),
            add_tok_type_and_position=2 * self.e,
            emb_layer_norm=LAYER_NORM_FLOPS * self.e,
            emb_dropout=DROPOUT_FLOPS * self.e
        ))

        return sum(embedding_flops.values()) * self.s

    def get_binary_classification_flops(self):
        classification_flops = dict(
            hidden=2 * self.h * self.h,
            hidden_bias=self.h,
            hidden_act=DROPOUT_FLOPS * self.h + ACTIVATION_FLOPS * self.h,
            logits=2 * self.h
            # soft_logits=2 * SOFTMAX_FLOPS
        )
        return sum(classification_flops.values()) * self.s

    def get_infer_flops(self):
        """Get the FLOPs for running inference with the transformer on a
        classification task."""
        # return (self.get_embedding_flops())
        return ((self.l * self.get_block_flops()) +
                self.get_embedding_flops() +
                self.get_binary_classification_flops())

    def get_params(self):
        embedding_params = {}
        embedding_params.update(dict(
            token_params=self.v * self.h,
            position_params=self.s * self.h,
            type_and_layer_norm=self.h * 3
        ))

        block_params = {}
        block_params.update(dict(
            attention_params=3 * (self.h * self.h + self.h),
            linear_params=self.h * self.h + self.h,
            fnn_params=self.h * self.i * 2 + self.i + self.h,
            layer_norm=self.h * 4,
            # pooler_params=self.h*self.h + self.h
        ))

        classification_params = {}
        classification_params.update(dict(
            pooler_params=self.h*self.h + self.h,
            dense_params=self.h * self.h + self.h,
            linear_params=self.h * 2 + 2
        ))
        # print(sum(embedding_params.values()), sum(block_params.values()) * self.l, sum(classification_params.values()))
        return sum(embedding_params.values()) + sum(block_params.values()) * self.l + sum(classification_params.values())


MODEL_FLOPS = collections.OrderedDict([
    ("roberta", [TransformerHparams().get_infer_flops(),
     TransformerHparams().get_params()])
])


def main():
    for k, v in MODEL_FLOPS.items():
        print(k, v)


if __name__ == "__main__":
    main()
