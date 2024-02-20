import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
#from src.constants import *
torch.manual_seed(1)


class TransNAS_TSAD(nn.Module):
    def __init__(self, feats,nhead,num_ffn_layers=2, lr=0.001, dropout_rate=0.1, dim_feedforward=16, batch=128,
                 encoder_layers=1, decoder_layers=1, attention_type="scaled_dot_product", 
                 positional_encoding_type="sinusoidal", phase_type="1phase", 
                 gaussian_noise_std=0, time_warping=False, time_masking=False, n_window =10, 
                 self_conditioning=True, layer_norm=True, use_linear_embedding=False,
                 activation_function="leaky_relu"):
        super(TransNAS_TSAD, self).__init__()

        d_model = 2 * feats if phase_type != "1phase" else feats
        head = feats

        # Model Metadata
        self.name = 'TransNAS_TSAD'
        self.lr = lr
        self.batch = batch
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.phase_type = phase_type

        # Activation Function
        if activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_function == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_function == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

        # Linear Embedding Layer
        if use_linear_embedding:
            self.linear_embedding = nn.Linear(2 * feats, d_model)
        else:
            self.linear_embedding = None

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, self.n_window, encoding_type=positional_encoding_type)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_ffn_layers=num_ffn_layers, dropout=dropout_rate, attention_type                =attention_type, activation_function=activation_function)
        self.transformer_encoder = TransformerEncoder(encoder_layer, encoder_layers)

        # Transformer Decoder
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_ffn_layers=num_ffn_layers, dropout=dropout_rate, attention_type                =attention_type, activation_function=activation_function)
        self.transformer_decoder1 = TransformerDecoder(decoder_layer, decoder_layers)
        
        
        
        if phase_type == "2phase":
            self.transformer_decoder2 = TransformerDecoder(decoder_layer, decoder_layers)

        # Final Layer
        self.fcn = nn.Sequential(nn.Linear(d_model, feats), self.activation)

        # Additional Configurations
        self.gaussian_noise_std = gaussian_noise_std
        self.time_warping = time_warping
        self.time_masking = time_masking
        self.self_conditioning = self_conditioning

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None


    def encode(self, src, c, tgt):

        src = torch.cat((src, c), dim=2)
        
        if self.phase_type != "1phase":
            tgt = tgt.repeat(1, 1, 2)
        
        # Gaussian Noise
        if self.training and self.gaussian_noise_std > 0:
            src += torch.randn_like(src) * self.gaussian_noise_std

        # Time Warping & Masking
        if self.training:
            if self.time_warping:
                src = time_warping(src)
            if self.time_masking:
                src = time_masking(src)

        
        if self.linear_embedding is not None:
          src = self.linear_embedding(src)

        if self.layer_norm:
            src = self.layer_norm(src)

        src *= math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        return tgt, memory

    def forward(self, src, tgt):

        
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))

        if self.phase_type == "2phase":
            if self.self_conditioning:
                c = (x1 - src) ** 2
            x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
            return x1, x2

        elif self.phase_type == "iterative":
            best_anomaly_score = torch.mean((x1 - src) ** 2)
            best_x = x1

            for _ in range(10):  # Maximum iterations set to 10
                c = (best_x - src) ** 2
                curr_x = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
                curr_anomaly_score = torch.mean((curr_x - src) ** 2)
                
                if curr_anomaly_score < best_anomaly_score:
                    best_anomaly_score = curr_anomaly_score
                    best_x = curr_x

            return best_x

        return x1



