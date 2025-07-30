import argparse
import os
import sys
import json
import pickle
import heapq
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.amp import autocast

# ensure llm_transparency_tool is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import llm_transparency_tool.routes.contributions as contributions
import llm_transparency_tool.routes.graph
from llm_transparency_tool.models.transparent_llm import TransparentLlm
from llm_transparency_tool.routes.graph_node import NodeType
from llm_transparency_tool.server.graph_selection import (
    GraphSelection,
    UiGraphEdge,
    UiGraphNode,
)
from llm_transparency_tool.server.utils import (
    B0,
    get_contribution_graph,
    load_dataset,
    load_model,
    get_val,
    possible_devices
)

def string_to_display(s: str) -> str:
    return s.replace(" ", "·")

@dataclass
class App:
    model_name: str = ""
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    amp_enabled: bool = False
    prepend_bos: bool = False
    contribution_threshold: float = 0.0
    top_k: int = 100
    max_samples: Optional[int] = None
    output_path: str = "./output/"
    dataset_path: str = "./data.pkl"
    normalize_before_unembedding: bool = False #This is set at false.

    def load_config(self, config_file: str):
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        self.model_name             = cfg.get("model_name", self.model_name)
        self.device                 = cfg.get("device", self.device)
        self.dtype                  = getattr(torch, cfg.get("dtype", "torch.float16"), self.dtype)
        self.amp_enabled            = cfg.get("amp_enabled", self.amp_enabled)
        self.prepend_bos            = cfg.get("prepend_bos", self.prepend_bos)
        self.contribution_threshold = cfg.get("contribution_threshold", self.contribution_threshold)
        self.top_k                  = cfg.get("top_k", self.top_k)

    @torch.no_grad()
    def compute_neuron_contributions(self, n_layers: int) -> torch.Tensor:
        # This grabs the tokens
        tokens = self.stateful_model.tokens()[B0]
        # build (T, L, N) tensor using original logic
        ffn_contributions = []
        resids = []
        resid_contributions = []
        for layer in range(n_layers):
            resid_mid = self.stateful_model.residual_after_attn(layer)[B0]
            resid_post = self.stateful_model.residual_out(layer)[B0]
            decomposed = self.stateful_model.decomposed_ffn_out(B0, layer, -1)
            layer_res = []
            c_resids = []
            for t in range(len(tokens)):
                c_ffn, c_resid = contributions.get_decomposed_mlp_contributions(
                    resid_mid[t], resid_post[t], decomposed[t],
                    renormalizing_threshold=self.contribution_threshold)
                layer_res.append(c_ffn)
                c_resids.append(c_resid)
            ffn_contributions.append(torch.stack(layer_res))  # (T,)
            resid_contributions.append(c_resids)
            stacked_resids = torch.stack([resid_mid, resid_post])
            resids.append(stacked_resids)
        # stack to (L, T) then transpose
        return torch.stack(ffn_contributions).transpose(1, 0), resid_contributions, resids  # (T, L, N) for contr, and resids

    @torch.no_grad()
    def draw_token_table(
        self,
        n_top: int,
        n_bottom: int,
        representation: torch.Tensor,
        predecessor: Optional[torch.Tensor] = None,
    ):
        device = representation.device
        model = self.stateful_model._model
        W_U = model.W_U.to(device)
        b_U = model.b_U.to(device)

        logits = representation @ W_U + b_U
        n_vocab = logits.shape[0]
        scores, indices = torch.topk(logits, n_top, largest=True)
        positions = list(range(n_top))

        if n_bottom > 0:
            low_scores, low_indices = torch.topk(logits, n_bottom, largest=False)
            indices = torch.cat((indices, low_indices.flip(0)))
            scores = torch.cat((scores, low_scores.flip(0)))
            positions += range(n_vocab - n_bottom, n_vocab)

        tokens = [string_to_display(w) for w in self.stateful_model.tokens_to_strings(indices)] #just use token ids, no need to convert

        if predecessor is not None:
            pre_logits = predecessor @ W_U + b_U
            _, sorted_pre_indices = pre_logits.sort(descending=True)
            sorted_pre_indices = sorted_pre_indices.tolist()
            pre_indices_dict = {index: pos for pos, index in enumerate(sorted_pre_indices)}
            old_positions = [pre_indices_dict[i] for i in indices.tolist()]

            def pos_gain_string(pos, old_pos):
                if pos == old_pos:
                    return ""
                sign = "↓" if pos > old_pos else "↑"
                return f"({sign}{abs(pos - old_pos)})"

            position_strings = [f"{i} {pos_gain_string(i, old_i)}" for (i, old_i) in zip(positions, old_positions)]
        else:
            position_strings = [str(pos) for pos in positions]
        
        return {t:[ps, s.item()] for (t, ps, s) in zip(tokens, position_strings, scores)}

    @torch.no_grad()
    def run(self, args):
        self.load_config(args.config)
        self.dataset_path = args.dataset_path
        self.output_path  = args.output_path
        self.max_samples  = args.max_samples
        os.makedirs(self.output_path, exist_ok=True)

        # load samples
        with open(self.dataset_path, 'rb') as f:
            all_samples = pickle.load(f)
        items = list(all_samples.items())
        if self.max_samples:
            items = items[:self.max_samples]

        # init model
        self.stateful_model = load_model(
            model_name  = self.model_name,
            _device     = self.device,
            _dtype      = self.dtype,
            prepend_bos = self.prepend_bos,
            revision = "main"
        )
        L = 32
        N = 11008
        n_top = 10
        n_bottom = 10

        # build heaps of (f_val, sent_id, token_idx, (resid_mid, resid_post), split_context_str, logit_lens_analysis)
        heaps: List[List[List[Tuple[float, int, int, Tuple, str, dict]]]] = [ [ [] for _ in range(N) ] for _ in range(L) ]
        for sent_id, sent in tqdm(items, desc="build heap"):  
            with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
                self.stateful_model.run([sent])
            context = self.stateful_model._model.to_str_tokens(self.stateful_model._last_run.tokens)
            contribs, cont_resid, resids = self.compute_neuron_contributions(L)
            contribs  = contribs.cpu()  # (T, L, N)
            T, _, _ = contribs.shape
            split_context_strs = []
            for t in range(T):            
                before = context[:t]
                at = str(context[t])
                after = context[t+1:]
                split_context_strs.append(f"{''.join(before)}*\t{at}\t*{''.join(after)}")

            for t in range(T):
                split_str = split_context_strs[t]
                for l in range(L):
                    resid_mid_t = resids[l][0][t]  
                    resid_post_t = resids[l][1][t]
                    logit_lens_dict = self.draw_token_table(n_top, n_bottom, resid_post_t, resid_mid_t)
                    for n, f_val in enumerate(contribs[t, l].tolist()):
                        h = heaps[l][n]
                        entry = (f_val, cont_resid[l][t], sent_id, t, (resid_mid_t, resid_post_t), split_str, logit_lens_dict)
                        if len(h) < self.top_k:
                            heapq.heappush(h, entry)
                        elif f_val > h[0][0]:
                            heapq.heapreplace(h, entry)

        out_file = os.path.join(self.output_path, 'heap.pkl')
        with open(out_file, "wb") as out:
            pickle.dump(heaps, out)
        print("Done: contexts in", out_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config',  type=str, default="/home/kaiwei/llm-transparency-tool/config/exp_olmo_config.json",)
    p.add_argument('--dataset_path', type=str, default="/mnt/disks/neurondata/dolma_data/dolma_samples_w_trunc_2.pkl")
    p.add_argument('--output_path', type=str, default="/mnt/disks/dolmaresiddata/")
    p.add_argument('--max_samples',  type=int, default=100)
    args = p.parse_args()
    App().run(args)