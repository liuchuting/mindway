# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, ops

from .mindspore_adapter import dtype_to_min


@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import mindspore as ms
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(ms.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=ms.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: ms.Type,
    ) -> Optional[ms.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")
        return to_causal_4d(batch_size, query_length, key_value_length, dtype, self.sliding_window)

    def to_4d(
        self,
        attention_mask_2d: ms.Tensor,
        query_length: int,
        dtype: ms.Type,
        key_value_length: Optional[int] = None,
    ) -> ms.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        return to_4d(attention_mask_2d, query_length, dtype, key_value_length, self.is_causal, self.sliding_window)

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: Union[Tuple, List],
        dtype: ms.Type,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        return _make_causal_mask(input_ids_shape, dtype, past_key_values_length, sliding_window)

    @staticmethod
    def _expand_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        return _expand_mask(mask, dtype, tgt_len)


def to_causal_4d(
    batch_size: int,
    query_length: int,
    key_value_length: int,
    dtype: ms.Type,
    sliding_window: Optional[int] = None,
) -> Optional[ms.Tensor]:
    """
    Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
    bias to upper right hand triangular matrix (causal mask).
    """
    # If shape is not cached, create a new causal mask and cache it
    input_shape = (batch_size, query_length)
    past_key_values_length = key_value_length - query_length

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    causal_4d_mask = None
    if input_shape[-1] > 1 or sliding_window is not None:
        causal_4d_mask = _make_causal_mask(
            input_shape,
            dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )

    return causal_4d_mask


def to_4d(
    attention_mask_2d: ms.Tensor,
    query_length: int,
    dtype: ms.Type,
    key_value_length: Optional[int] = None,
    is_causal: bool = True,
    sliding_window: Optional[int] = None,
) -> ms.Tensor:
    """
    Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
    key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
    causal, a causal mask will be added.
    """
    input_shape = (attention_mask_2d.shape[0], query_length)

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    causal_4d_mask = None
    if (input_shape[-1] > 1 or sliding_window is not None) and is_causal:
        if key_value_length is None:
            raise ValueError(
                "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
            )

        past_key_values_length = key_value_length - query_length
        causal_4d_mask = _make_causal_mask(
            input_shape,
            dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )
    elif sliding_window is not None:
        raise NotImplementedError("Sliding window is currently only implemented for causal masking")

    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_attn_mask = _expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1])

    if causal_4d_mask is not None:
        expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), dtype_to_min(dtype))

    # expanded_attn_mask + causal_4d_mask can cause some overflow
    expanded_4d_mask = expanded_attn_mask

    return expanded_4d_mask


def _make_causal_mask(
    input_ids_shape: Union[Tuple, List],
    dtype: ms.Type,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full((tgt_len, tgt_len), dtype_to_min(dtype), dtype=dtype)
    mask_cond = ops.arange(mask.shape[-1])
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), ms.tensor(0).to(dtype))

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.cat([ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1

        context_mask = 1 - ops.triu(ops.ones_like(mask, dtype=ms.int32), diagonal=diagonal)
        mask = mask.masked_fill(context_mask.bool(), dtype_to_min(dtype))

    return mask[None, None, :, :].tile((bsz, 1, 1, 1))


def _expand_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].tile((1, 1, tgt_len, 1)).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), dtype_to_min(dtype))


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[ms.Tensor],
    input_shape: Union[Tuple, List],
    inputs_embeds: ms.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`ms.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`ms.Tensor`):
            The embedded inputs as a mindspore Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            is_causal=True,
            sliding_window=sliding_window,
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(inverted_mask.bool(), dtype_to_min(inputs_embeds.dtype))
    else:
        attention_mask = to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, sliding_window=sliding_window
        )

    return attention_mask


def _prepare_4d_attention_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`ms.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`ms.dtype`):
            The mindspore dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return _expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _create_4d_causal_attention_mask(
    input_shape: Union[Tuple, List],
    dtype: ms.Type,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[ms.Tensor]:
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`

    Args:
        input_shape (`tuple(int)` or `list(int)`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        dtype (`ms.dtype`):
            The mindspore dtype the created mask shall have.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, sliding_window=sliding_window
    )

    return attention_mask


def _prepare_4d_attention_mask_for_sdpa(mask: ms.Tensor, dtype: ms.dtype, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`ms.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    _, key_value_length = mask.shape
    tgt_len = tgt_len if tgt_len is not None else key_value_length

    # torch.jit.trace, symbolic_trace and torchdynamo with fullgraph=True are unable to capture data-dependent controlflows.
    if mint.all(mask == 1):
        return None
    else:
        return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


# Adapted from _prepare_4d_causal_attention_mask
def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[ms.Tensor],
    input_shape: Union[tuple, list],
    inputs_embeds: ms.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.

    In case no token is masked in the `attention_mask` argument, we simply set it to `None` for the cases `query_length == 1` and
    `key_value_length == query_length`, and rely instead on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # torch.jit.trace, symbolic_trace and torchdynamo with fullgraph=True are unable to capture the controlflow `is_causal=attention_mask is None and q_len > 1`
    # used as an SDPA argument. We keep compatibility with these tracing tools by always using SDPA's `attn_mask` argument in case we are tracing.
    # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).

    ignore_causal_mask = AttentionMaskConverter._ignore_causal_mask_sdpa(
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
        sliding_window=sliding_window,
    )

    if ignore_causal_mask:
        expanded_4d_mask = None
    elif attention_mask is None:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        if attention_mask.dim() == 4:
            expanded_4d_mask = attention_mask
        else:
            expanded_4d_mask = attn_mask_converter.to_4d(
                attention_mask,
                input_shape[-1],
                dtype=inputs_embeds.dtype,
                key_value_length=key_value_length,
            )

        # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
            expanded_4d_mask, min_dtype=dtype_to_min(inputs_embeds.dtype)
        )

    return expanded_4d_mask

