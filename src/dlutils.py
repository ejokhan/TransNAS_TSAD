import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, encoding_type="sinusoidal"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.encoding_type = encoding_type

        if self.encoding_type not in ['sinusoidal', 'fourier']:
            raise ValueError(f"Positional encoding type '{encoding_type}' not recognized. Choose 'sinusoidal' or 'fourier'.")

    def forward(self, x):
        seq_length = x.size(1)
        
        if self.encoding_type == 'sinusoidal':
            pe = torch.zeros(seq_length, self.d_model, device=x.device)
            position = torch.arange(0, seq_length, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float().to(x.device) * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe
            
        elif self.encoding_type == 'fourier':
            position = torch.arange(0, seq_length, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float().to(x.device) * (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(seq_length, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position / div_term)
            pe[:, 1::2] = torch.cos(position / div_term)
            pe = pe.unsqueeze(0)
            x = x + pe

        return self.dropout(x)



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, num_ffn_layers=2, dropout=0, attention_type="scaled_dot_product", activation_function="leaky_relu"):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention_type = attention_type
        self.num_ffn_layers = num_ffn_layers
        self.activation = self.get_activation(activation_function)
        
        if attention_type == "scaled_dot_product":
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        # Initialize dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Define multiple FFN layers
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        ) for _ in range(self.num_ffn_layers)])
    
    def get_activation(self, name):
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh
        }
        return activations.get(name, nn.LeakyReLU)  # default to LeakyReLU if not found

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        if self.attention_type == "scaled_dot_product":
            src2, _ = self.self_attn(src, src, src)
        # Placeholder for future attention mechanisms
        # elif self.attention_type == "some_new_attention":
        #     # Code for new attention mechanism

        src = src + self.dropout1(src2)
        for ffn_layer in self.ffn:
            src = src + ffn_layer(src)
        
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, num_ffn_layers=2, dropout=0, attention_type="scaled_dot_product", activation_function="leaky_relu"):
        super(TransformerDecoderLayer, self).__init__()

        self.attention_type = attention_type
        self.num_ffn_layers = num_ffn_layers
        self.activation = self.get_activation(activation_function)
        
        if attention_type == "scaled_dot_product":
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        # Initialize dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Define multiple FFN layers
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        ) for _ in range(self.num_ffn_layers)])
    
    def get_activation(self, name):
        # same activation function getter as in the encoder
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh
        }
        return activations.get(name, nn.LeakyReLU)  # default to LeakyReLU if not found

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        if self.attention_type == "scaled_dot_product":
            tgt2, _ = self.self_attn(tgt, tgt, tgt)
            tgt = tgt + self.dropout1(tgt2)
            tgt2, _ = self.multihead_attn(tgt, memory, memory)
        # Placeholder for future attention mechanisms
        # elif self.attention_type == "some_new_attention":
        #     # Code for new attention mechanism

        tgt = tgt + self.dropout2(tgt2)
        for ffn_layer in self.ffn:
            tgt = tgt + ffn_layer(tgt)

        return tgt


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        
class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


import numpy as np

def time_warping(ts, warp_factor=0.2):
    """
    Apply time warping to the given time series.

    Parameters:
    - ts: Input time series of shape (sequence_length, features)
    - warp_factor: Degree of time stretching/compression

    Returns:
    - Warped time series
    """

    # Ensure the input is a numpy array for processing
    ts_numpy = ts.detach().cpu().numpy()

    # Generate random interpolation points
    seq_len = ts_numpy.shape[0]
    p1, p2 = np.random.randint(1, seq_len - 1, 2)
    p1, p2 = min(p1, p2), max(p1, p2)

    # Generate interpolated values
    interpolated_points = np.linspace(p1, p2, int((p2 - p1) * (1 + warp_factor)))

    # Create a new time series with the warped section
    new_ts_list = []
    for i in range(seq_len):
        if i < p1:
            new_ts_list.append(ts_numpy[i])
        elif i < p2:
            idx = int((i - p1) * len(interpolated_points) / (p2 - p1))
            new_ts_list.append(ts_numpy[int(interpolated_points[idx])])
        else:
            new_ts_list.append(ts_numpy[i])

    # Convert back to tensor and move to the original device
    #new_ts_tensor = torch.tensor(new_ts_list, dtype=torch.float32).to(ts.device)
    new_ts_array = np.array(new_ts_list)
    new_ts_tensor = torch.tensor(new_ts_array, dtype=torch.float32).to(ts.device)

    return new_ts_tensor.double()



def time_masking(ts, mask_factor=0.2):
    """
    Apply time masking to the given time series.

    Parameters:
    - ts: Input time series tensor of shape (sequence_length, features)
    - mask_factor: Fraction of the sequence to mask

    Returns:
    - Time series tensor with masked values
    """

    # Ensure the input is a numpy array for processing
    ts_numpy = ts.detach().cpu().numpy()

    # Decide the number of values to mask
    mask_len = int(ts_numpy.shape[0] * mask_factor)

    # Choose a random start point
    start = np.random.randint(0, ts_numpy.shape[0] - mask_len)

    # Mask the values
    ts_numpy[start:start+mask_len] = 0

    # Convert back to tensor and move to the original device
    masked_ts_tensor = torch.tensor(ts_numpy, dtype=ts.dtype).to(ts.device)

    return masked_ts_tensor


