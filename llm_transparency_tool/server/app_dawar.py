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

@dataclass
class App:
    model_name: str = ""
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    amp_enabled: bool = True
    prepend_bos: bool = False
    contribution_threshold: float = 0.0
    top_k: int = 100
    max_samples: Optional[int] = None
    output_path: str = "./output/"
    dataset_path: str = "./data.pkl"

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
        tokens = self.stateful_model.tokens()[B0]
        # populate contributions
        _, _ = get_contribution_graph(
            self.stateful_model,
            "",
            tokens.tolist(),
            self.contribution_threshold
        )
        # build (T, L, N) tensor using original logic
        ffn_contributions = []
        for layer in range(n_layers):
            resid_mid = self.stateful_model.residual_after_attn(layer)[B0]
            resid_post = self.stateful_model.residual_out(layer)[B0]
            decomposed = self.stateful_model.decomposed_ffn_out(B0, layer, -1)
            layer_res = []
            for t in range(len(tokens)):
                c_ffn, _ = contributions.get_decomposed_mlp_contributions(
                    resid_mid[t], resid_post[t], decomposed[t],
                    renormalizing_threshold=self.contribution_threshold)
                layer_res.append(c_ffn)
            ffn_contributions.append(torch.stack(layer_res))  # (T,)
        # stack to (L, T) then transpose
        return torch.stack(ffn_contributions).transpose(1, 0)  # (T, L, N)

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

        # PASS 1: build heaps of (f_val, sent_id, token_idx)
        heaps: List[List[List[Tuple[float, int, int]]]] = [ [ [] for _ in range(N) ] for _ in range(L) ]
        for sent_id, sent in tqdm(items, desc="Pass 1"):  
            with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
                self.stateful_model.run([sent])
            contribs = self.compute_neuron_contributions(L).cpu()  # (T, L, N)
            T, _, _ = contribs.shape
            for t in range(T):
                for l in range(L):
                    for n, f_val in enumerate(contribs[t, l].tolist()):
                        h = heaps[l][n]
                        entry = (f_val, sent_id, t)
                        if len(h) < self.top_k:
                            heapq.heappush(h, entry)
                        elif f_val > h[0][0]:
                            heapq.heapreplace(h, entry)

        # collect pointers
        pointers: Dict[int, List[Tuple[int,int,int,float]]] = {}
        for l in range(L):
            for n in range(N):
                for f_val, sent_id, t in heaps[l][n]:
                    pointers.setdefault(sent_id, []).append((t, l, n, f_val))

        # PASS 2: only revisit sentences with hits
        out_file = os.path.join(self.output_path, 'contexts2.tsv')
        with open(out_file, 'w') as out:
            out.write("sent_id\ttoken_idx\tlayer\tneuron\tf_val\tcontext\n")
            for sent_id, hits in tqdm(pointers.items(), desc="Pass2"):
                sent = all_samples[sent_id].replace("\n"," ")
                with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
                    self.stateful_model.run([sent])
                # optionally recompute contribs if needed; here f_val from pointers suffices
                for t, l, n, f_val in hits:
                    out.write(f"{sent_id}\t{t}\t{l}\t{n}\t{f_val:.6f}\t{sent}\n")
        print("Done: contexts in", out_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config',  type=str, default="/home/kaiwei/llm-transparency-tool/config/exp_olmo_config.json",)
    p.add_argument('--dataset_path', type=str, default="/mnt/disks/neurondata/dolma_data/dolma_samples_w_trunc_2.pkl")
    p.add_argument('--output_path', type=str, default="/home/kaiwei/llm-transparency-tool/test_out")
    p.add_argument('--max_samples',  type=int, default=100)
    args = p.parse_args()
    App().run(args)