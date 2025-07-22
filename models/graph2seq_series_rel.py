# models/graph2seq_series_rel.py
# Modified for Mixture‑of‑Experts decoder feed‑forward layers
# Author: 2025‑07‑17 – Option‑1 MoE integration

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.decoders.transformer import TransformerDecoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.gnn_encoder import GraphFeatEncoder
from onmt.modules import PositionalEncoding, Embeddings
from onmt.utils.misc import sequence_mask


# ----------------------------------------------------------------------
#               Mixture‑of‑Experts Feed‑Forward sub‑layer
# ----------------------------------------------------------------------
class MoEFeedForward(nn.Module):
    r"""
    Position‑wise Feed‑Forward layer with **top‑k sparse routing**.

    Parameters
    ----------
    d_model : int
        Transformer hidden size.
    d_ff : int
        Hidden size of each expert feed‑forward MLP (the same as the original FFN).
    num_experts : int
        Total number of experts in this layer.
    topk : int, default 1
        How many experts each token is sent to (top‑1 is Switch Transformer style).
    gate_temperature : float, default 1.0
        Temperature for the softmax gating distribution.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        topk: int = 1,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        assert 1 <= topk <= num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.topk = topk
        self.temperature = gate_temperature

        # ------------- experts -------------
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_ff, d_model, bias=True),
                    nn.Dropout(0.1),
                )
                for _ in range(num_experts)
            ]
        )

        # ------------- gating network -------------
        # A single linear layer is standard; no bias simplifies load‑balancing calc.
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # auxiliary load‑balancing loss variables (updated each forward)
        self._gate_probs: Optional[torch.Tensor] = None  # (tokens, num_experts)

    # -------- public helper to retrieve last routing probs (for logging) -------
    @property
    def last_routing_probs(self) -> Optional[torch.Tensor]:
        """Return tensor (n_tokens, num_experts) from the last forward call."""
        return self._gate_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (seq_len, batch, d_model)

        Returns
        -------
        Tensor, same shape as *x*.
        """
        s, b, h = x.shape
        x_flat = x.reshape(s * b, h)  # (tokens, d_model)

        # ------------ gating ------------
        logits = self.gate(x_flat) / self.temperature  # (tokens, num_experts)
        if self.topk == 1:
            # Hard routing (Switch). We keep the probabilities for load‑balance.
            indices = torch.argmax(logits, dim=-1, keepdim=True)  # (tokens, 1)
            self._gate_probs = F.one_hot(indices.squeeze(-1), self.num_experts).type_as(
                logits
            )  # hard one‑hot
            # Gather expert outputs efficiently:
            #   Compute every expert in parallel, then select.
            all_expert_out = torch.stack(
                [expert(x_flat) for expert in self.experts], dim=1
            )  # (tokens, num_experts, d_model)
            indices_expanded = (
                indices.unsqueeze(-1)
                .expand(-1, -1, h)
                .to(all_expert_out.device)
            )  # (tokens, 1, d_model)
            y_flat = all_expert_out.gather(1, indices_expanded).squeeze(1)
        else:
            # top‑k soft routing
            probs = F.softmax(logits, dim=-1)  # (tokens, num_experts)
            self._gate_probs = probs.detach()  # save for statistics
            topk_probs, topk_idx = torch.topk(probs, self.topk, dim=-1)  # (tokens, k)
            all_expert_out = torch.stack(
                [expert(x_flat) for expert in self.experts], dim=1
            )  # (tokens, num_experts, d_model)
            # Gather the k expert outputs, weight by probs, then sum
            tokens = x_flat.size(0)
            gathered = all_expert_out[
                torch.arange(tokens, device=x_flat.device).unsqueeze(1),
                topk_idx,
            ]  # (tokens, k, d_model)
            weighted = gathered * topk_probs.unsqueeze(-1)
            y_flat = weighted.sum(dim=1)  # (tokens, d_model)

        y = y_flat.reshape(s, b, h)
        return y


# ----------------------------------------------------------------------
#              Graph2SeqSeriesRel with MoE decoder layers
# ----------------------------------------------------------------------
class Graph2SeqSeriesRel(nn.Module):
    """
    Original Graph2SMILES model **plus decoder MoE**.
    """

    def __init__(self, args, vocabs):
        super().__init__()
        self.args = args
        self.vocabs = vocabs

        # -------------- Encoder ----------------
        self.graph_enc = GraphFeatEncoder(args)
        self.global_enc: Optional[nn.Module] = None
        if args.attn_enc_num_layers > 0:
            self.global_enc = TransformerEncoder(
                num_layers=args.attn_enc_num_layers,
                d_model=args.hidden_size,
                heads=args.enc_heads,
                d_ff=args.enc_ff_size,
                dropout=args.dropout,
                embeddings=None,
                position_encoding=False,
            )

        # -------------- Decoder Embedding ----------------
        self.tgt_embed = Embeddings(
            word_vec_size=args.hidden_size,
            position_encoding=True,
            dropout=args.dropout,
            pad_idx=vocabs["tgt"].pad(),
        )

        # -------------- Transformer Decoder ----------------
        self.decoder = TransformerDecoder(
            num_layers=args.dec_num_layers,
            d_model=args.hidden_size,
            heads=args.dec_heads,
            d_ff=args.dec_ff_size,
            dropout=args.dropout,
            embeddings=self.tgt_embed,
            alignment_layer=None,  # not needed for autoregressive generation
            alignment_heads=None,
        )

        # -------------- Swap in MoE feed‑forwards ----------------
        self._inject_moe_into_decoder()

        # -------------- Output projection ----------------
        self.generator = nn.Linear(args.hidden_size, len(vocabs["tgt"]))

        # -------------- Criterion convenience ----------------
        self._criterion = nn.NLLLoss(
            ignore_index=vocabs["tgt"].pad(), reduction="sum"
        )

    # ------------------------------------------------------------------ #
    #                        Private helper
    # ------------------------------------------------------------------ #
    def _inject_moe_into_decoder(self):
        """Replace each decoder layer's feed‑forward sub‑layer by MoE."""
        num_experts = getattr(self.args, "moe_num_experts", 4)
        topk = getattr(self.args, "moe_topk", 1)
        temp = getattr(self.args, "moe_gating_temperature", 1.0)

        for layer in self.decoder.transformer_layers:
            d_model = layer.self_attn.model_size
            # Original FFN hidden size can be read from existing layer
            # (weight shape is (d_ff, d_model))
            d_ff = layer.feed_forward.w_1.weight.size(0)
            layer.feed_forward = MoEFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                topk=topk,
                gate_temperature=temp,
            )

    # ------------------------------------------------------------------ #
    #                           Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        graph_batch,                # batch of graphs (dict of tensors)
        tgt: torch.Tensor,          # target token indices (tgt_len, batch)
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """
        Standard training forward pass (teacher forcing).

        Returns
        -------
        loss : torch.Tensor
            Scalar loss (summed over tokens).
        """
        # --------- Encode graph ---------
        memory_bank, enc_padding_mask = self.graph_enc(graph_batch)
        if self.global_enc is not None:
            memory_bank, _ = self.global_enc(
                memory_bank, enc_padding_mask
            )  # same mask

        # --------- Decode ---------
        dec_out, _ = self.decoder(
            tgt,                    # (t, b)
            memory_bank,            # (src_len, b, h)
            memory_lengths=(~enc_padding_mask).sum(-1),  # (b,)
            step=None,
            future=False,
        )

        # --------- Loss ---------
        logits = self.generator(dec_out)  # (t, b, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        tgt_y = tgt[1:]  # shift (predict token i from previous token i‑1)
        pred = log_probs[:-1].contiguous().view(-1, log_probs.size(-1))
        gold = tgt_y.contiguous().view(-1)

        loss = self._criterion(pred, gold)
        return loss

    # ------------------------------------------------------------------ #
    #                   Convenience: generate / inference
    # ------------------------------------------------------------------ #
    def generate(
        self,
        graph_batch,
        max_length: int = 256,
        beam_size: int = 5,
        device: Optional[torch.device] = None,
    ) -> List[str]:
        """
        Greedy/beam search generation wrapper (simplified).
        """
        device = device or next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            memory_bank, enc_padding_mask = self.graph_enc(graph_batch)
            if self.global_enc is not None:
                memory_bank, _ = self.global_enc(
                    memory_bank, enc_padding_mask
                )
            # Delegating actual beam search to ONMT's BeamSearch code is recommended.
            # For brevity we show a greedy decode loop here.
            batch_size = memory_bank.size(1)
            bos = self.vocabs["tgt"].bos()
            eos = self.vocabs["tgt"].eos()
            gen_tokens = (
                torch.full((1, batch_size), bos, dtype=torch.long, device=device)
            )

            finished = [False] * batch_size

            for _ in range(max_length):
                dec_out, _ = self.decoder(
                    gen_tokens,
                    memory_bank,
                    memory_lengths=(~enc_padding_mask).sum(-1),
                    step=None,
                    future=False,
                )
                next_logits = self.generator(dec_out[-1])  # (b, vocab)
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (b,1)
                gen_tokens = torch.cat([gen_tokens, next_token.T], dim=0)

                # Stop if all sequences generated <eos>
                finished = [
                    fin or nt.item() == eos for fin, nt in zip(finished, next_token)
                ]
                if all(finished):
                    break

            # Convert to strings
            outputs: List[str] = []
            for b in range(batch_size):
                tok_ids = gen_tokens[1:, b].tolist()
                if eos in tok_ids:
                    tok_ids = tok_ids[: tok_ids.index(eos)]
                tokens = [self.vocabs["tgt"].lookup_id(i) for i in tok_ids]
                outputs.append("".join(tokens))
            return outputs
