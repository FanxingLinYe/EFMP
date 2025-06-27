
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention, BertIntermediate, BertOutput, BertSelfOutput
from transformers.modeling_utils import apply_chunking_to_forward

class efmpFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.text_attention = BertAttention(config)  # Self-attention for text
        self.image_attention = BertAttention(config)  # Self-attention for image
        self.text_cross_attention = BertSelfAttention(config)  # For Text-to-Image
        self.image_cross_attention = BertSelfAttention(config)  # For Image-to-Text
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.out_layer = BertSelfOutput(config)
        self.image_to_text_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
            self,
            hidden_states,  # Text features [batch_size, text_seq_len, hidden_size]
            encoder_hidden_states,  # Image features [batch_size, img_seq_len, hidden_size]
            gap_token,
            attention_mask=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        # Self-attention on text features
        text_attention_outputs = self.text_attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        text_output = text_attention_outputs[0]

        # Self-attention on image features
        image_attention_outputs = self.image_attention(
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        image_output = image_attention_outputs[0]

        # Parallel Cross-Attention
        # Text-to-Image: Text as Q, Image as K, V
        text_cross_outputs = self.text_cross_attention(
            text_output,  # Q: Text features
            attention_mask,
            None,
            image_output,  # K, V: Image features after self-attention
            encoder_attention_mask,
            None,
            output_attentions,
        )
        visual_perceived_text = text_cross_outputs[0] + gap_token

        # Image-to-Text: Image as Q, Text as K, V
        image_cross_outputs = self.image_cross_attention(
            image_output,  # Q: Image features after self-attention
            encoder_attention_mask,
            None,
            hidden_states,  # K, V: Text features
            attention_mask,
            None,
            output_attentions,
        )
        text_perceived_image = image_cross_outputs[0]
        # Project to match text sequence length
        text_seq_len = text_output.size(1)  # Get text sequence length
        img_seq_len = image_output.size(1)  # Get image sequence length
        if img_seq_len != text_seq_len:
            text_perceived_image = self.image_to_text_projection(text_perceived_image)
            text_perceived_image = text_perceived_image.repeat(1, text_seq_len // img_seq_len + 1, 1)
            text_perceived_image = text_perceived_image[:, :text_seq_len, :]

        # Fuse parallel outputs
        fused_output = visual_perceived_text + text_perceived_image
        fused_output = self.out_layer(fused_output, text_output)

        # Feed-forward network with chunking
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, fused_output
        )

        outputs = (layer_output,) + text_attention_outputs[1:] + image_attention_outputs[1:] + text_cross_outputs[1:] + image_cross_outputs[1:]
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

