import torch

def videomaev2_tokens(hf_model, pixel_values):  # pixel_values: (B,C,T,H,W)
    vit = hf_model.model  # VisionTransformer внутри VideoMAEv2
    B = pixel_values.size(0)

    x = vit.patch_embed(pixel_values)  # (B, N, D)
    if vit.pos_embed is not None:
        x = x + vit.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
    x = vit.pos_drop(x)

    for blk in vit.blocks:
        x = blk(x)

    # нормализация токенов (в модели fc_norm есть, если use_mean_pooling=True)
    if vit.fc_norm is not None:
        x = vit.fc_norm(x)  # LayerNorm по последней размерности
    return x  # (B, N, D)


def videomaev2(hf_model, pixel_values, pool_spatial="mean", upsample_to_frames=False):
    vit = hf_model.model
    tokens = videomaev2_tokens(hf_model, pixel_values)  # (B, N, D)

    B, N, D = tokens.shape
    T = pixel_values.shape[2]
    tube = vit.tubelet_size
    Tt = T // tube  # T'
    # patch_size в коде хранится как tuple (ph, pw)
    ph, pw = vit.patch_embed.patch_size
    H, W = pixel_values.shape[3], pixel_values.shape[4]
    S = (H // ph) * (W // pw)

    x = tokens.view(B, Tt, S, D)  # (B, T', S, D)

    if pool_spatial == "mean":
        x = x.mean(dim=2)         # (B, T', D)
    elif pool_spatial == "max":
        x = x.max(dim=2).values   # (B, T', D)
    else:
        raise ValueError("pool_spatial must be 'mean' or 'max'")

    if upsample_to_frames and tube > 1:
        x = x.repeat_interleave(tube, dim=1)  # (B, T, D) приблизительно
    return x
