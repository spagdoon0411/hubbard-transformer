# Write a function that initializes weights from the model dict

from model.model import HubbardWaveFunction
import torch.nn.init as init


wv = HubbardWaveFunction(
    embed_dim=32,
    n_heads=2,
    n_layers=2,
    dim_feedforward=64,
    particle_number=4,
    max_len=10,
)

# TODO: this should be generated as a function from model params


# Define the initialization mapping
initialization_mapping = {
    "kaiming_uniform": [
        "transformer_encoder.layers.0.self_attn.in_proj_weight",
        "transformer_encoder.layers.0.self_attn.out_proj.weight",
        "transformer_encoder.layers.0.linear1.weight",
        "transformer_encoder.layers.0.linear2.weight",
        "transformer_encoder.layers.1.self_attn.in_proj_weight",
        "transformer_encoder.layers.1.self_attn.out_proj.weight",
        "transformer_encoder.layers.1.linear1.weight",
        "transformer_encoder.layers.1.linear2.weight",
    ],
    "kaiming_normal": [
        "embedding.token_embedding.occs_to_logits",
    ],
    "xavier_uniform": [
        "embedding.param_embedding.interaction_weights",
        "embedding.param_embedding.n_param_to_target",
        "deembedding.prob_head",
        "deembedding.phase_head",
    ],
    "xavier_normal": [],
    "constant": [
        "logit_norm.weight",
        "logit_norm.bias",
        "transformer_encoder.layers.0.self_attn.in_proj_bias",
        "transformer_encoder.layers.0.self_attn.out_proj.bias",
        "transformer_encoder.layers.0.linear1.bias",
        "transformer_encoder.layers.0.linear2.bias",
        "transformer_encoder.layers.0.norm1.weight",
        "transformer_encoder.layers.0.norm1.bias",
        "transformer_encoder.layers.0.norm2.weight",
        "transformer_encoder.layers.0.norm2.bias",
        "transformer_encoder.layers.1.self_attn.in_proj_bias",
        "transformer_encoder.layers.1.self_attn.out_proj.bias",
        "transformer_encoder.layers.1.linear1.bias",
        "transformer_encoder.layers.1.linear2.bias",
        "transformer_encoder.layers.1.norm1.weight",
        "transformer_encoder.layers.1.norm1.bias",
        "transformer_encoder.layers.1.norm2.weight",
        "transformer_encoder.layers.1.norm2.bias",
        "post_transform_norm.weight",
        "post_transform_norm.bias",
    ],
    "uniform": [],
}


# Function to initialize weights
def initialize_weights(model):
    for name, param in model.named_parameters():
        if name in initialization_mapping["xavier_uniform"]:
            init.xavier_uniform_(param)
        elif name in initialization_mapping["kaiming_uniform"]:
            init.kaiming_uniform_(
                param,
                nonlinearity="relu",
                mode="fan_in",
            )
        elif name in initialization_mapping["kaiming_normal"]:
            init.kaiming_normal_(
                param,
                nonlinearity="relu",
                mode="fan_in",
            )
        elif name in initialization_mapping["xavier_normal"]:
            init.xavier_normal_(param)
        elif name in initialization_mapping["constant"]:
            if "bias" in name:
                init.constant_(param, 0)  # Initialize biases to 0
            else:
                init.constant_(param, 1)  # Initialize weights to 1
        elif name in initialization_mapping["uniform"]:
            init.uniform_(param, -0.1, 0.1)  # Example range for uniform initialization


initialize_weights(wv)
