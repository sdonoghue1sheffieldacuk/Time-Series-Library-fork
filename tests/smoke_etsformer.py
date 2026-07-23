import torch
from types import SimpleNamespace

# Ensure repo path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.ETSformer import Model

# Minimal config matching expected attributes
configs = SimpleNamespace()
configs.task_name = 'short_term_forecast'
configs.seq_len = 24
configs.label_len = 12
configs.pred_len = 12
configs.e_layers = 1
configs.d_layers = 1
configs.enc_in = 1
configs.d_model = 16
configs.embed = 'fixed'
configs.freq = 'h'
configs.dropout = 0.1
configs.n_heads = 4
configs.top_k = 5
configs.d_ff = 64
configs.activation = 'gelu'
configs.attn_shape_start = 0.2
configs.attn_shape_end = 1.0
configs.attn_shape_power = 1.0
configs.factor = 5
configs.c_out = 1
configs.num_class = 10

# Instantiate model
model = Model(configs)
model.eval()

B = 2
seq_len = configs.seq_len
enc_in = configs.enc_in

# Dummy inputs: x_enc, x_mark_enc, x_dec, x_mark_dec
x_enc = torch.randn(B, seq_len, enc_in)
x_mark_enc = torch.randn(B, seq_len, 4)  # time features, size may vary
x_dec = torch.randn(B, configs.pred_len, enc_in)
x_mark_dec = torch.randn(B, configs.pred_len, 4)

with torch.no_grad():
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

print('Output shape:', out.shape)
print('Output sample:', out[0, :2, 0].tolist())
