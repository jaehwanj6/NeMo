# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""
import torch

from nemo.collections.nlp.modules.common.megatron.megatron_perceiver_encoders import MegatronPerceiverEncoderModule
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()


__all__ = ["MegatronTransformerEncoderDecoderModule"]


class MegatronTransformerEncoderDecoderModule(MegatronModule):
    """Transformer encoder-decoder model.
    """

    def __init__(
        self,
        encoder,
        decoder,
        # AttnMaskType enum mask type (e.g., padding, casual)
        encoder_attn_mask_type: AttnMaskType = None,
        decoder_attn_mask_type: AttnMaskType = None,
        hidden_steps: int = None,
    ):
        super(MegatronTransformerEncoderDecoderModule, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.hidden_steps = hidden_steps
        if isinstance(encoder, MegatronPerceiverEncoderModule) and hidden_steps is None:
            raise ValueError(
                f"hidden_steps cannot be None for perceiver encoders. It is needed to compute the encoder-decoder cross attention mask."
            )

        # try to infer mask_type if not given
        if encoder_attn_mask_type is None:
            if encoder is None:
                encoder_attn_mask_type = None
            # Perceiver does not have a `.model` attribute, assume it always uses padding mask.
            elif isinstance(encoder, MegatronPerceiverEncoderModule):
                encoder_attn_mask_type = AttnMaskType.padding
            elif hasattr(encoder.model, 'self_attn_mask_type'):
                encoder_attn_mask_type = encoder.model.self_attn_mask_type
            else:
                raise AttributeError(
                    "Could not find an attribute for encoder self_attn_mask_type, make sure it is set when instatiating the encoder or pass it to the constructor of this class."
                )
        if decoder_attn_mask_type is None:
            if decoder is None:
                decoder_attn_mask_type = None
            elif hasattr(decoder.model, 'self_attn_mask_type'):
                decoder_attn_mask_type = decoder.model.self_attn_mask_type
            else:
                raise AttributeError(
                    "Could not find an attribute for decoder self_attn_mask_type, make sure it is set when instatiating the decoder or pass it to the constructor of this class."
                )

        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.decoder_attn_mask_type = decoder_attn_mask_type

        self._encoder_key = "encoder"
        self._decoder_key = "decoder"

    def encode(
        self, enc_input, enc_attn_mask, enc_layer_past=None, enc_get_key_value=False,
    ):
        if self.encoder is None:
            raise ValueError(f"Cannot call .encode(...) when self.encoder is None.")
        """Encodes embedder input using encoder"""
        enc_output = self.encoder(
            enc_input=enc_input,
            enc_attn_mask=enc_attn_mask,
            layer_past=enc_layer_past,
            get_key_value=enc_get_key_value,
        )

        return enc_output

    def decode(
        self, dec_input, dec_attn_mask, enc_output, enc_attn_mask, dec_layer_past=None, dec_get_key_value=False,
    ):
        if self.decoder is None:
            raise ValueError(f"Cannot call .decode(...) when self.decoder is None.")
        """Decodes embedder input using decoder and encoder input"""
        dec_output = self.decoder(
            dec_input=dec_input,
            dec_attn_mask=dec_attn_mask,
            layer_past=dec_layer_past,
            get_key_value=dec_get_key_value,
            enc_output=enc_output,
            enc_attn_mask=enc_attn_mask,
        )

        return dec_output

    def forward(
        self,
        enc_input,
        enc_attn_mask,
        dec_input,
        dec_attn_mask,
        enc_layer_past=None,
        enc_get_key_value=False,
        enc_output=None,
        dec_layer_past=None,
        dec_get_key_value=False,
        output_enc_hidden_only=False,
    ):
        # encoder
        if enc_output is None:
            if self.encoder is not None:
                enc_output = self.encode(
                    enc_input=enc_input,
                    enc_attn_mask=enc_attn_mask,
                    enc_layer_past=enc_layer_past,
                    enc_get_key_value=enc_get_key_value,
                )
            else:
                assert self.encoder_hidden_state is not None
                enc_output = self.encoder_hidden_state
        else:
            enc_output = enc_output.to(enc_input.dtype)

        if self.decoder is None or output_enc_hidden_only:
            return enc_output

        # decoder
        # Adjust encoder attention mask if encoder is a perceiver.
        if self.encoder is not None and isinstance(self.encoder, MegatronPerceiverEncoderModule):
            enc_attn_mask = torch.ones(enc_output.size(0), self.hidden_steps).to(enc_output.device)

        dec_output = self.decode(
            dec_input=dec_input,
            dec_attn_mask=dec_attn_mask,
            enc_output=enc_output,
            enc_attn_mask=enc_attn_mask,
            dec_layer_past=dec_layer_past,
            dec_get_key_value=dec_get_key_value,
        )

        return dec_output, enc_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}

        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.encoder.load_state_dict(state_dict[self._encoder_key], strict=strict)
        self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)
