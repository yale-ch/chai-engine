# Vendored from https://github.com/wfzyx/von (src/von/models/option_marker.py), Apache-2.0, unchanged
# except this header. Loaded by chai.system_one.encoders.VonProvider.
"""Option-Marker Joint Decision Model for ModernBERT.

Enables single-pass non-autoregressive decision evaluation:
Pack premise and K options into a single sequence marked by [MASK] tokens and
score every option in one encoder pass, eliminating K separate cross-encoder passes.

Two attention modes:
- default (von-1.1 and earlier): full bidirectional self-attention, so every option
  also attends to every other option and its position depends on the packing order.
- independent_options (von-1.2+): each option attends only to the premise and to
  itself, with position ids reset to the prefix length, so its logit is a function
  of (premise, that option) alone and the result is provably option-order invariant.
  See build_independent_option_masks / build_option_invariant_position_ids.
"""

import re
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

_DIGIT_RUN_RE = re.compile(r"\d+")


def split_digits(text: str) -> str:
    """Space out every digit in every digit run: "2026" -> "2 0 2 6".

    ModernBERT's BPE merges multi-digit runs inconsistently by leading digit
    ("2026" -> "20"+"26", "692" -> "6"+"92", "500" -> one token) -- the same
    magnitude tokenizes differently depending on what it starts with, which
    makes digit-level arithmetic unlearnable from a training set this size
    (see benchmarks/probe_numeral_sensitivity.py: 70% of picks survive full
    digit randomisation). Splitting every digit into its own token removes
    that inconsistency. Must be applied identically at training time
    (`training/train_option_marker.py`'s `collate_marker_fn`) and here, or the
    two paths silently diverge -- gated by the same `digit_split` flag,
    persisted in `marker_calibration.json` so old checkpoints are unaffected.
    """
    return _DIGIT_RUN_RE.sub(lambda m: " ".join(m.group(0)), text)

class OptionMarkerScorer(nn.Module):
    """Calibrated MLP scoring head for option-marker representations."""

    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects (N_options, hidden_size) representations to scalar logits."""
        x = self.input_norm(x)
        h = self.dense(x)
        h = self.act(h)
        h = self.norm(h)
        h = self.dropout(h)
        return self.out_proj(h).squeeze(-1)


def build_independent_option_masks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_positions: List[List[int]],
    position_ids: torch.Tensor,
    sliding_window: Optional[int] = None,
) -> dict:
    """Builds attention masks that make each option's representation a function
    of (prefix, that option) alone -- never of other options or their order.

    Every packed sequence is [prefix tokens][MASK opt0][opt0 text][MASK opt1]...
    A prefix token may attend to any prefix token. An option token may attend to
    any prefix token or any token within its own option span, and nothing else --
    in particular, never another option's tokens. This is provably order-invariant:
    permuting which option occupies which slot cannot change any option's computed
    logit, since its computation never depends on what else is in the sequence.

    Applied to `full_attention` layers directly; ANDed with a sliding-window mask
    for `sliding_attention` layers so long premises keep efficient local-context
    behaviour. Crucially this local window is computed from the order-invariant
    `position_ids` (see build_option_invariant_position_ids), NOT raw sequence
    index -- ModernBERT's own sliding-window helper uses raw index distance,
    which is NOT order-invariant here since an option's raw index shifts with
    how many (attention-blocked) tokens of other options precede it.
    """
    B, seq_len = input_ids.shape
    device = input_ids.device
    # Index of the trailing [SEP]/EOS token (last real, non-pad position). It must
    # not be absorbed into whichever option happens to land in the final slot --
    # that would make the last slot special regardless of order.
    last_content_idx = attention_mask.long().sum(dim=1) - 1  # (B,)
    option_id = torch.full((B, seq_len), -1, dtype=torch.long, device=device)
    for b, positions in enumerate(mask_positions):
        for k, start in enumerate(positions):
            end = positions[k + 1] if k + 1 < len(positions) else last_content_idx[b].item()
            option_id[b, start:end] = k

    oi = option_id.unsqueeze(2)  # query position's option id, (B, seq, 1)
    oj = option_id.unsqueeze(1)  # key position's option id, (B, 1, seq)
    query_is_prefix = oi == -1
    key_is_prefix = oj == -1
    same_option = oi == oj
    allowed = (query_is_prefix & key_is_prefix) | (~query_is_prefix & (key_is_prefix | same_option))
    pad_ok = attention_mask.bool().unsqueeze(1)
    allowed = allowed & pad_ok
    eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
    allowed = allowed | eye  # a fully-masked row would produce NaN in softmax
    full_mask = allowed.unsqueeze(1)  # (B, 1, seq, seq)

    if sliding_window is None:
        sliding_mask = full_mask
    else:
        pi = position_ids.unsqueeze(2)  # (B, seq, 1)
        pj = position_ids.unsqueeze(1)  # (B, 1, seq)
        local = (pi - pj).abs() <= sliding_window
        sliding_mask = (allowed & local).unsqueeze(1)
        sliding_mask = sliding_mask | eye.unsqueeze(1)

    return {"full_attention": full_mask, "sliding_attention": sliding_mask}


def build_option_invariant_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_positions: List[List[int]],
) -> torch.Tensor:
    """Resets every option's position_ids to start right after the prefix.

    Blocking cross-option attention alone is NOT order-invariant under RoPE:
    RoPE encodes *relative* distance, so an option's relative offset from the
    prefix still shifts depending on how many (attention-blocked) tokens of
    other options sit between it and the prefix in the packed sequence. This
    makes every option start at the same position_ids offset (prefix length),
    as if it were the only option present, so combined with
    build_independent_option_masks its computation is a true function of
    (prefix, that option) alone -- independent of packing order. The trailing
    [SEP]/EOS token is excluded from the last option's span (same boundary fix
    as build_independent_option_masks) so it isn't option-order-dependent either.
    """
    B, seq_len = input_ids.shape
    device = input_ids.device
    last_content_idx = attention_mask.long().sum(dim=1) - 1  # (B,)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(B, 1)
    for b, positions in enumerate(mask_positions):
        if not positions:
            continue
        prefix_len = positions[0]
        for k, start in enumerate(positions):
            end = positions[k + 1] if k + 1 < len(positions) else last_content_idx[b].item()
            span_len = end - start
            position_ids[b, start:end] = torch.arange(
                prefix_len, prefix_len + span_len, device=device
            )
    return position_ids



def _neutralise(text: str, *specials: str) -> str:
    """Break any special-token literal in user text with a zero-width joiner."""
    for sp in specials:
        if sp and sp in text:
            text = text.replace(sp, sp[0] + "\u200d" + sp[1:])
    return text

class OptionMarkerModel(nn.Module):
    """ModernBERT decision model with single-pass option-marker scoring."""

    def __init__(
        self,
        base_model_id: str = "checkpoints/von-modernbert-rlcd",
        max_position_embeddings: int = 8192,
        dropout: float = 0.1,
        digit_split: bool = False,
    ):
        super().__init__()
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model_id)
        config.max_position_embeddings = max_position_embeddings
        self.encoder = AutoModel.from_pretrained(base_model_id, config=config)
        self.hidden_size = self.encoder.config.hidden_size
        self.scorer = OptionMarkerScorer(hidden_size=self.hidden_size, dropout=dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, model_max_length=max_position_embeddings)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.digit_split = digit_split

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: List[List[int]],
        independent_options: bool = False,
    ) -> List[torch.Tensor]:
        """Runs single forward pass and returns list of option logits per sample.

        independent_options=True swaps the default full-cross-attention mask for
        one that makes each option's logit a function of (prefix, that option)
        alone -- see build_independent_option_masks. Provably order-invariant by
        construction, at the cost of removing option-to-option attention.
        """
        if independent_options:
            position_ids = build_option_invariant_position_ids(input_ids, attention_mask, mask_positions)
            sliding_window = getattr(self.encoder.config, "sliding_window", None)
            enc_attention_mask = build_independent_option_masks(
                input_ids, attention_mask, mask_positions,
                position_ids, sliding_window,
            )
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=enc_attention_mask, position_ids=position_ids
            )
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, seq_len, H)

        batch_logits = []
        for b, pos_list in enumerate(mask_positions):
            opt_reps = last_hidden[b, pos_list]  # (K, H)
            logits = self.scorer(opt_reps)  # (K,)
            batch_logits.append(logits)

        return batch_logits

    def pack_sequence(
        self,
        state: str,
        question: str,
        options: List[str],
    ) -> str:  # noqa: D401  (see _neutralise)
        """Packs state, question, and candidate options into an option-marker string."""
        mask = self.tokenizer.mask_token
        sep = self.tokenizer.sep_token
        # User text must not be able to forge structure: a tool doc that says
        # "[MASK]" would add a phantom option marker (seen on ToolRet), and a
        # "[SEP]" would split the prefix. Neutralise the literals with a
        # zero-width join so the visible text is unchanged but the tokenizer no
        # longer maps them to the special ids.
        state, question = _neutralise(state, mask, sep), _neutralise(question, mask, sep)
        options = [_neutralise(o, mask, sep) for o in options]
        prefix = f"{question} {state}".strip() if question else state.strip()
        opts_packed = " ".join(f"{mask} {opt.strip()}" for opt in options)
        packed = f"{prefix} {sep} {opts_packed}"
        return split_digits(packed) if self.digit_split else packed
