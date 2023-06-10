import torch
import torch.nn as nn

import open_clip

class FrozenOpenCLIPEmbedder(nn.Module):
    """OpenCLIP Transformer to encode text"""
    def __init__(self, arch='ViT-H-14', version='laion2b_s32b_b79k', device='cpu', max_length=77, freeze=True, layer='penultimate', cache_dir='models'):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name=arch, pretrained=version, cache_dir=cache_dir)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.layer = layer
        if layer == 'penultimate':
            self.layer_idx = 1 
        elif layer == 'last':
            self.layer_idx = 0
        else:
            raise ValueError('only last and penultimate layers are supported')

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z
            
    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text) 
