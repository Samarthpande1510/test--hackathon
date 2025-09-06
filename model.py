# models.py
import torch
import torch.nn as nn
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class DNATransformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=256, nhead=8, num_layers=6, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, seq_len) long tensor
        seq_len = x.size(1)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0).to(x.device)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)
        return x  # (batch, d_model)


class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim=256, kingdom_out=6, phylum_out=50, class_out=200):
        super().__init__()
        self.kingdom_head = nn.Linear(input_dim, kingdom_out)
        self.phylum_head = nn.Linear(input_dim, phylum_out)
        self.class_head = nn.Linear(input_dim, class_out)
        self.confidence = nn.Linear(input_dim, 1)

    def forward(self, x):
        kingdom = torch.softmax(self.kingdom_head(x), dim=1)
        phylum = torch.softmax(self.phylum_head(x), dim=1)
        class_pred = torch.softmax(self.class_head(x), dim=1)
        confidence = torch.sigmoid(self.confidence(x))
        return {
            'kingdom': kingdom,
            'phylum': phylum,
            'class': class_pred,
            'confidence': confidence
        }


class NoveltyVAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def load_model_state(model, path):
    if path and os.path.exists(path):
        try:
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state)
            logger.info(f"Loaded weights from {path}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    else:
        logger.info(f"No checkpoint at {path}, using random init.")
    return model


def initialize_models(cfg, device):
    """
    Initialize model instances and optionally load weights from disk.
    cfg: dict-like config
    device: 'cpu' or 'cuda'
    """
    dna_encoder = DNATransformer(
        vocab_size=cfg.get('vocab_size', 5),
        d_model=cfg.get('d_model', 256),
        nhead=cfg.get('nhead', 8),
        num_layers=cfg.get('num_layers', 6),
        max_len=cfg.get('transformer_max_len', 1000)
    )
    tax_classifier = HierarchicalClassifier(
        input_dim=cfg.get('d_model', 256)
    )
    novelty_vae = NoveltyVAE(input_dim=cfg.get('d_model', 256), latent_dim=64)

    # Load states if available
    dna_encoder = load_model_state(dna_encoder, cfg.get('dna_encoder_path'))
    tax_classifier = load_model_state(tax_classifier, cfg.get('tax_classifier_path'))
    novelty_vae = load_model_state(novelty_vae, cfg.get('novelty_vae_path'))

    dna_encoder.to(device).eval()
    tax_classifier.to(device).eval()
    novelty_vae.to(device).eval()

    return dna_encoder, tax_classifier, novelty_vae
