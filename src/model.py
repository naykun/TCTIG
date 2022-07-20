from inspect import isfunction
from math import ceil
from functools import partial
from itertools import islice, cycle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence

from rotary_embedding_torch import RotaryEmbedding, broadcat, apply_rotary_emb
from g_mlp_pytorch import gMLPBlock

# helpers

def exists(val):
    return val is not None

# def default(val, d):
#     return val if exists(val) else d

def cast_tuple(val, depth = 1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True)
    return (t * alpha).softmax(dim = dim)

def apply_pos_emb(pos_emb, qkv):
    n = qkv[0].shape[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))

# classes
class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., stable = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rotary_pos_emb = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax if not self.stable else stable_softmax

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))

        q = q * self.scale

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., stable = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.causal = causal

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond, mask = None, rotary_pos_emb = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax if not self.stable else stable_softmax

        q = self.to_q(x)
        kv = self.to_kv(cond).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        if exists(rotary_pos_emb):
            q = apply_pos_emb(rotary_pos_emb, (q,))

        q = q * self.scale

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


# sparse attention with convolutional pattern, as mentioned in the blog post. customizable kernel size and dilation

class SparseConvCausalAttention(nn.Module):
    def __init__(self, dim, seq_len, image_size = 32, kernel_size = 5, dilation = 1, heads = 8, dim_head = 64, dropout = 0., stable = False, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size must be odd'

        inner_dim = dim_head *  heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.stable = stable

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rotary_pos_emb = None):
        b, n, _, h, img_size, kernel_size, dilation, seq_len, device = *x.shape, self.heads, self.image_size, self.kernel_size, self.dilation, self.seq_len, x.device
        softmax = torch.softmax if not self.stable else stable_softmax

        img_seq_len = img_size ** 2
        text_len = seq_len - img_seq_len

        # padding

        padding = seq_len - n
        mask = default(mask, lambda: torch.ones(b, text_len, device = device).bool())

        x = F.pad(x, (0, 0, 0, padding), value = 0)
        mask = mask[:, :text_len]

        # derive query / keys / values

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(rotary_pos_emb):
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d', h = h), (q, k, v))
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        q *= self.scale

        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))

        # text attention

        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(dots_text)

        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        attn_text = softmax(dots_text, dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # image attention

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding = effective_kernel_size // 2

        k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h = img_size), (k_img, v_img))
        k_img, v_img = map(lambda t: F.unfold(t, kernel_size, padding = padding, dilation = dilation), (k_img, v_img))
        k_img, v_img = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j = kernel_size ** 2), (k_img, v_img))

        # let image attend to all of text

        dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)
        dots_image_to_text = einsum('b i d, b j d -> b i j', q_img, k_text)

        # calculate causal attention for local convolution

        i, j = dots_image.shape[-2:]
        img_seq = torch.arange(img_seq_len, device = device)
        k_img_indices = rearrange(img_seq.float(), '(h w) -> () () h w', h = img_size)
        k_img_indices = F.pad(k_img_indices, (padding,) * 4, value = img_seq_len) # padding set to be max, so it is never attended to
        k_img_indices = F.unfold(k_img_indices, kernel_size, dilation = dilation)
        k_img_indices = rearrange(k_img_indices, 'b j i -> b i j')

        # mask image attention

        q_img_indices = rearrange(img_seq, 'i -> () i ()')
        causal_mask =  q_img_indices < k_img_indices

        # concat text mask with image causal mask

        causal_mask = repeat(causal_mask, '() i j -> b i j', b = b * h)
        mask = repeat(mask, 'b j -> (b h) i j', i = i, h = h)
        mask = torch.cat((~mask, causal_mask), dim = -1)

        # image can attend to all of text

        dots = torch.cat((dots_image_to_text, dots_image), dim = -1)
        dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim = -1)

        # aggregate

        attn_image_to_text, attn_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum('b i j, b i j d -> b i d', attn_image, v_img)
        out_image_to_text = einsum('b i j, b j d -> b i d', attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # combine attended values for both text and image

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out =  self.to_out(out)
        return out[:, :n]

# sparse axial causal attention

class SparseAxialCausalAttention(nn.Module):
    def __init__(self, dim, seq_len, image_size = 32, axis = 0, heads = 8, dim_head = 64, dropout = 0., stable = False, **kwargs):
        super().__init__()
        assert axis in {0, 1}, 'axis must be either 0 (along height) or 1 (along width)'
        self.axis = axis

        inner_dim = dim_head *  heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size

        self.stable = stable

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rotary_pos_emb = None):
        b, n, _, h, img_size, axis, seq_len, device = *x.shape, self.heads, self.image_size, self.axis, self.seq_len, x.device
        softmax = torch.softmax if not self.stable else stable_softmax

        img_seq_len = img_size ** 2
        text_len = seq_len - img_seq_len

        # padding

        padding = seq_len - n
        mask = default(mask, lambda: torch.ones(b, text_len, device = device).bool())

        x = F.pad(x, (0, 0, 0, padding), value = 0)
        mask = mask[:, :text_len]

        # derive queries / keys / values

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # if exists(rotary_pos_emb):
        #     q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
        if exists(rotary_pos_emb):
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d', h = h), (q, k, v))
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        q *= self.scale

        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))

        # text attention

        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(dots_text)

        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        attn_text = softmax(dots_text, dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # image attention

        split_axis_einops = 'b (h w) c -> b h w c' if axis == 0 else 'b (h w) c -> b w h c'
        merge_axis_einops = 'b x n d -> b (x n) d' if axis == 0 else 'b x n d -> b (n x) d'

        # split out axis

        q_img, k_img, v_img = map(lambda t: rearrange(t, split_axis_einops, h = img_size), (q_img, k_img, v_img))

        # similarity

        dots_image_to_image = einsum('b x i d, b x j d -> b x i j', q_img, k_img)
        dots_image_to_text = einsum('b x i d, b j d -> b x i j', q_img, k_text)

        dots = torch.cat((dots_image_to_text, dots_image_to_image), dim = -1)

        # mask so image has full attention to text, but causal along axis

        bh, x, i, j = dots.shape
        causal_mask = torch.ones(i, img_size, device = device).triu_(img_size - i + 1).bool()
        causal_mask = repeat(causal_mask, 'i j -> b x i j', b = bh, x = x)

        mask = repeat(mask, 'b j -> (b h) x i j', h = h, x = x, i = i)
        mask = torch.cat((~mask, causal_mask), dim = -1)

        dots.masked_fill_(mask, mask_value)

        # attention.

        attn = softmax(dots, dim = -1)

        # aggregate

        attn_image_to_text, attn_image_to_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum('b x i j, b x j d -> b x i d', attn_image_to_image, v_img)
        out_image_to_text = einsum('b x i j, b j d -> b x i d', attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # merge back axis

        out_image = rearrange(out_image, merge_axis_einops, x = img_size)

        # combine attended values for both text and image

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out =  self.to_out(out)
        return out[:, :n]

# microsoft sparse attention CUDA kernel

class SparseAttention(Attention):
    def __init__(
        self,
        *args,
        block_size = 16,
        text_seq_len = 256,
        num_random_blocks = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        from deepspeed.ops.sparse_attention import SparseSelfAttention, VariableSparsityConfig
        self.block_size = block_size

        num_random_blocks = default(num_random_blocks, self.seq_len // block_size // 4)
        global_block_indices = list(range(ceil(text_seq_len / block_size)))

        self.attn_fn = SparseSelfAttention(
            sparsity_config = VariableSparsityConfig(
                num_heads = self.heads,
                block = self.block_size,
                num_random_blocks = num_random_blocks,
                global_block_indices = global_block_indices,
                attention = 'unidirectional' if self.causal else 'bidirectional'
            ),
            max_seq_length = self.seq_len,
            attn_mask_mode = 'add'
        )

    def forward(self, x, mask = None, rotary_pos_emb = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        remainder = n % self.block_size
        mask = default(mask, lambda: torch.ones(b, n, device = device).bool())

        if remainder > 0:
            padding = self.block_size - remainder
            x = F.pad(x, (0, 0, 0, padding), value = 0)
            mask = F.pad(mask, (0, padding), value = False)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if exists(rotary_pos_emb):
            q, k, v = apply_pos_emb(rotary_pos_emb, (q, k, v))

        key_pad_mask = None
        if exists(mask):
            key_pad_mask = ~mask

        attn_mask = None
        if self.causal:
            i, j = q.shape[-2], k.shape[-2]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            attn_mask = torch.zeros(i, j, device = device).to(q)
            mask_value = max_neg_value(q) / 2
            attn_mask.masked_fill_(mask, mask_value)

        out = self.attn_fn(q, k, v, attn_mask = attn_mask, key_padding_mask = key_pad_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out[:, :n]

class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim = self.dim, keepdim = True)
        return x / maxes

# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        # self.fn = fn
    def forward(self, x, **kwargs):
        return x * self.scale

# layer norm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0., mult = 4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# token shift classes

class PreShiftToken(nn.Module):
    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len

    def forward(self, x, **kwargs):
        n = x.shape[1]
        seq_len, image_size = self.seq_len, self.image_size
        img_seq_len = image_size ** 2
        text_len = seq_len - img_seq_len + 1
        padding = seq_len - n + 1

        # get text and image tokens

        x_text, x_img = x[:, :text_len], x[:, text_len:]
        x_img = F.pad(x_img, (0, 0, 0, padding))
        x_img = rearrange(x_img, 'b (h w) d -> b h w d', h = image_size)

        # shift 1 from the left for text tokens

        x_text_shift, x_text_pass = x_text.chunk(2, dim = -1)
        x_text_shift = F.pad(x_text_shift, (0, 0, 1, -1))
        x_text = torch.cat((x_text_shift, x_text_pass), dim = -1)

        # shift from top, left for image tokens

        x_img_shift_top, x_img_shift_left, *x_img_pass = x_img.chunk(4, dim = -1)
        x_img_shift_left = F.pad(x_img_shift_left, (0, 0, 1, -1))
        x_img_shift_top = F.pad(x_img_shift_top, (0, 0, 0, 0, 1, -1))
        x_img = torch.cat((x_img_shift_top, x_img_shift_left, *x_img_pass), dim = -1)

        # merge text and image sequence back together

        x_img = rearrange(x_img, 'b h w d -> b (h w) d')
        x = torch.cat((x_text, x_img[:, :-padding]), dim = 1)
        return self.fn(x, **kwargs)


class SingleTransformer(nn.Module):
    def __init__(self, attention, attention_cond, ff, dim, depth):
        super().__init__()
        self.atten_norm = nn.LayerNorm(dim)
        self.attention = attention
        self.attention_scale = LayerScale(dim, depth)

        if attention_cond is not None:
            self.atten_norm_cond = nn.LayerNorm(dim)
            self.attention_cond = attention_cond
            self.attention_scale_cond = LayerScale(dim, depth)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = ff
        self.ff_scale = LayerScale(dim, depth)

    def forward(self, x, mask=None, cond=None, mask_cond=None, rotary_pos_emb = None):
        # attention
        att = self.atten_norm(x)
        att = self.attention(att, mask, rotary_pos_emb)
        att = self.attention_scale(att)
        x = x + att

        # attention_condition
        if cond is not None:
            att = self.atten_norm_cond(x)
            att = self.attention_cond(att, cond, mask_cond)
            att = self.attention_scale_cond(att)
            x = x + att

        # feedforward
        ff = self.ff_norm(x)
        ff = self.ff(ff)
        ff = self.ff_scale(ff)
        ff = x + ff
        return ff


# main transformer class
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        seq_len,
        reversible = False,
        causal = True,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_types = None,
        image_fmap_size = None,
        sparse_attn = False,
        stable = False,
        shift_tokens = False,
        rotary_emb = True,
        num_image_tokens=1024
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for ind, sparse_attn, attn_type in zip(range(depth), sparse_layer, attn_type_layer):
            if attn_type == 'full':
                attn_class = partial(Attention, stable = stable)
            elif attn_type == 'sparse':
                attn_class = SparseAttention
            elif attn_type == 'axial_row':
                attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 0, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'axial_col':
                attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 1, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'conv_like':
                attn_class = partial(SparseConvCausalAttention, seq_len = seq_len, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'mlp':
                attn_class = partial(gMLPBlock, seq_len = seq_len)
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')

            if attn_type != 'mlp':
                attn = attn_class(dim, causal = causal, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                attn = attn_class(dim = dim, causal = causal, dim_ff = dim * 4)

            ff = FeedForward(dim, mult = ff_mult, dropout = ff_dropout)

            if shift_tokens:
                attn, ff = map(lambda t: PreShiftToken(t, image_size = image_fmap_size, seq_len = seq_len), (attn, ff))
                
            attn_cond = CrossAttention(dim=dim, seq_len=seq_len, stable = stable)
            
            layers.append(SingleTransformer(attn, attn_cond, ff, dim, ind + 1))

        self.layers = layers

        # generate positional embeddings for rotary
	# text length equals to one when using encoder-decoder structure.
        pos_emb = None
        if rotary_emb:
            assert 'mlp' not in attn_types, 'you cannot use gMLPs if rotary embedding is turned on'
            rot_dim = dim_head // 3
            img_seq_len = (image_fmap_size ** 2)
        #     text_len = seq_len - img_seq_len + 1
            text_len = seq_len - img_seq_len

            text_pos_emb = RotaryEmbedding(dim = rot_dim)
            img_axial_pos_emb = RotaryEmbedding(dim = rot_dim, freqs_for = 'pixel')

            text_freqs = text_pos_emb(torch.arange(text_len))
            img_to_text_freqs = text_pos_emb(torch.full((img_seq_len,), 8192)) # image is given a position far away from text
            text_freqs = torch.cat((text_freqs, img_to_text_freqs), dim = 0)

            img_freqs_axial = img_axial_pos_emb(torch.linspace(-1, 1, steps = image_fmap_size))
            img_freqs = broadcat((rearrange(img_freqs_axial, 'i d -> i () d'), rearrange(img_freqs_axial, 'j d -> () j d')), dim = -1)
            img_freqs = rearrange(img_freqs, 'h w d -> (h w) d')

            text_axial_freqs = img_axial_pos_emb(torch.full((text_len,), -10.))  # text is given a position of -10 apart from the image axial positions, which is from range [-1, 1]
            text_axial_freqs = torch.cat((text_axial_freqs, text_axial_freqs), dim = -1)
            img_freqs = torch.cat((text_axial_freqs, img_freqs), dim = 0)

            pos_emb = torch.cat((text_freqs, img_freqs), dim = -1)
            pos_emb = rearrange(pos_emb, 'n d -> () () n d')

        self.register_buffer('pos_emb', pos_emb)

        self.to_logits_img = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_image_tokens),
        )

    def forward(self, x, **kwargs):
        for layer in self.layers:
            # import ipdb; ipdb.set_trace()
            x = layer(x, rotary_pos_emb = self.pos_emb, **kwargs)
        logits = self.to_logits_img(x)
        
        return logits
