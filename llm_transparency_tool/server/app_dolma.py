import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import time
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Why is it not working without this?
print("sys.argv:", sys.argv)

import networkx as nx
import torch
from jaxtyping import Float
from torch.amp import autocast
from transformers import HfArgumentParser

# import llm_transparency_tool.components
# from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
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
from networkx.classes.digraph import DiGraph


from dataclasses import dataclass
from typing import List
import json
import pickle


@dataclass
class LogLensResult:
    token_idx: int = 0
    top_tokens: List[str] = field(default_factory=list)
    max_logit: float = 0.0
    max_prob: float = 0.0
    entropy: float = 0.0



# @st.cache_resource(
#     hash_funcs={
#         nx.Graph: id,
#         DiGraph: id
#     }
# )
def cached_build_paths_to_predictions(
    graph: nx.Graph,
    n_layers: int,
    n_tokens: int,
    starting_tokens: List[int],
    threshold: float,
):
    return llm_transparency_tool.routes.graph.build_paths_to_predictions(
        graph, n_layers, n_tokens, starting_tokens, threshold
    )

# @st.cache_resource(
#     hash_funcs={
#         TransformerLensTransparentLlm: id
#     }
# )
def cached_run_inference_and_populate_state(
    stateless_model,
    sentences
):
    # stateful_model = stateless_model.copy()
    stateless_model.run(sentences)
    return stateless_model

    

class App:
    _stateful_model: TransparentLlm = None
    _graph: Optional[nx.Graph] = None
    _contribution_threshold: float = 0.0
    _renormalize_after_threshold: bool = False
    _normalize_before_unembedding: bool = False #not normalised to be consistent

    @property
    def stateful_model(self) -> TransparentLlm:
        return self._stateful_model

    def __init__(self):
        pass
        # self._config = config
        # st.set_page_config(layout="wide")
        # st.markdown(margins_css, unsafe_allow_html=True)

    def _get_representation(self, node: Optional[UiGraphNode]) -> Optional[Float[torch.Tensor, "d_model"]]:
        if node is None:
            return None
        fn = {
            NodeType.AFTER_ATTN: self.stateful_model.residual_after_attn,
            NodeType.AFTER_FFN: self.stateful_model.residual_out,
            NodeType.FFN: None,
            NodeType.ORIGINAL: self.stateful_model.residual_in,
        }
        return fn[node.type](node.layer)[B0][node.token]    
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.model_name = config.get("model_name", "allenai/OLMo-7B-0424-hf")
        self.device = config.get("device", "cuda:0")
        self._model_path = config.get("_model_path", None) 
        dtype_str = config.get("dtype", "torch.float16")
        self.dtype = getattr(torch, dtype_str, torch.float16)
        self.amp_enabled = config.get("amp_enabled", True)
        self._renormalize_after_threshold = config.get("renormalize_after_threshold", True)
        self._normalize_before_unembedding = config.get("normalize_before_unembedding", True)
        self._prepend_bos = config.get("prepend_bos", False)
        self._do_neuron_level = config.get("do_neuron_level", True)
        self._do_head_level = config.get("do_head_level", False)
        self._contribution_threshold = config.get("contribution_threshold", 0.01) #set at 0.01 for the moment
        self._logit_lens_topK = config.get("logit_lens_topK", 10)
        self._logit_lens_topK_neurons = config.get("logit_lens_topK_neurons", 10)

    def _unembed(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        return self.stateful_model.unembed(representation, normalize=self._normalize_before_unembedding)

    def draw_graph(self, contribution_threshold: float) -> Optional[GraphSelection]:
        tokens = self.stateful_model.tokens()[B0]
        n_tokens = tokens.shape[0]
        model_info = self.stateful_model.model_info()

        graphs = cached_build_paths_to_predictions(
            self._graph,
            model_info.n_layers,
            n_tokens,
            range(n_tokens),
            contribution_threshold,
        )

        return llm_transparency_tool.components.contribution_graph(
            model_info,
            self.stateful_model.tokens_to_strings(tokens),
            graphs,
            key=f"graph_{hash(self.sentence)}",
        )


    def run_inference(self):

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._stateful_model = cached_run_inference_and_populate_state(self.stateful_model, [self.sentence])

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._graph, self._contributions_dict = get_contribution_graph(
                self.stateful_model,
                "",
                self.stateful_model.tokens()[B0].tolist(),
                (self._contribution_threshold if self._renormalize_after_threshold else 0.0),
            )

    #@profile
    def process_logits(self, logit_scores):
        tokens = self.stateful_model.tokens()[B0]
        #s2 = time.time()
        #tokens = self.stateful_model.tokens()[B0]
        probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
        entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
        sorted_indices = torch.argsort(logit_scores, dim=-1, descending=True)
        results = []
        
        for token_idx in range(len(entropy)):
            result = {}
            result["token_idx"] = token_idx
            # tok = self.stateful_model.tokens_to_strings(sorted_indices[token_idx][:self._logit_lens_topK].cpu())
            result["top_tokens"] = [int(i) for i in sorted_indices[token_idx][:self._logit_lens_topK].cpu()]
            result["max_logit"] = get_val(logit_scores[token_idx].max().cpu())
            result["max_prob"] = get_val(probs[token_idx].max())
            result["entropy"] = get_val(entropy[token_idx])
            results.append(result)
        return results
        

    @torch.no_grad()
    def run_logit_lens_on_resid(self, n_layers):
        full_results = {}
        for layer in range(n_layers):
            for resid_pos in ["pre", "mid"]:
                if resid_pos == "pre":
                    representations = self.stateful_model.residual_in(layer)
                else:
                    representations = self.stateful_model.residual_after_attn(layer)
                logit_scores = self._unembed(representations)[B0]
                hook_name = f"{layer}_{resid_pos}"
                full_results[hook_name] = self.process_logits(logit_scores)
        representations = self.stateful_model.residual_out(n_layers-1)
        logit_scores = self._unembed(representations)[B0]
        full_results["final_post"] = self.process_logits(logit_scores)
        return full_results
    

    @torch.no_grad()
    def run_logit_lens_on_outputs(self, n_layers):
        full_results = {}
        representations = self.stateful_model.residual_in(0)
        logit_scores = self._unembed(representations)[B0]
        full_results["embed"] = self.process_logits(logit_scores)
        for layer in range(n_layers):
            for resid_pos in ["attn", "mlp"]:
                if resid_pos == "attn":
                    representations = self.stateful_model._get_block(layer, "hook_attn_out")
                else:
                    representations = self.stateful_model.ffn_out(layer)
                logit_scores = self._unembed(representations)[B0]
                hook_name = f"{layer}_{resid_pos}_out"
                full_results[hook_name] = self.process_logits(logit_scores)
        return full_results
    

    @torch.no_grad()
    def run_logit_lens_on_heads(self, n_layers, n_heads):
        full_results = {}
        for layer in range(n_layers):
            representations = self.stateful_model._get_block(layer, "attn.hook_result")
            for head in range(n_heads):
                hook_name = f"L{layer}H{head}"
                logit_scores = self._unembed(representations[:, :, head])[B0]
                full_results[hook_name] = self.process_logits(logit_scores)
        return full_results
    
    @torch.no_grad()
    def compute_neuron_contributions(self, n_layers):
        tokens = self.stateful_model.tokens()[B0]

        ffn_contributions = []
        for layer in range(n_layers):
            hook_name = f"L{layer}"
            resid_mid = self.stateful_model.residual_after_attn(layer)[B0]
            resid_post = self.stateful_model.residual_out(layer)[B0]
            decomposed_ffn = self.stateful_model.decomposed_ffn_out(B0, layer, -1)
            results = []
            for token in range(len(tokens)):
                c_ffn, _ = contributions.get_decomposed_mlp_contributions(resid_mid[token], resid_post[token], decomposed_ffn[token], renormalizing_threshold=self._contribution_threshold)
                results.append(c_ffn) #change back to c_ffn
            ffn_contributions.append(torch.stack(results))
        return torch.stack(ffn_contributions).transpose(1, 0)


    @torch.no_grad()
    def run_logit_lens_on_neurons(self, n_layers, sel_neurons_layerwise):
        full_results = {}
        for layer in range(n_layers):
            sel_neurons = sel_neurons_layerwise[layer]
            for neuron in sel_neurons:
                hook_name = f"L{layer}N{neuron}"
                representations = self.stateful_model.neuron_output(layer, int(neuron)).unsqueeze(0).unsqueeze(1)
                logit_scores = self._unembed(representations)[B0][0]
                probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
                entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
                sorted_indices = torch.argsort(logit_scores, dim=-1, descending=True).cpu()

                result = {}
                #result["top_tokens"] = [w for w in self.stateful_model.tokens_to_strings(sorted_indices[:self._logit_lens_topK])]
                result["top_tokens"] = [int(i) for i in sorted_indices[:self._logit_lens_topK].cpu()]
                result["max_logit"] = get_val(logit_scores.max().cpu())
                result["max_prob"] = get_val(probs.max().cpu())
                result["entropy"] = get_val(entropy)

                full_results[hook_name] = result
        return full_results


    def process_sentences(self, sent):
        self.sentence = sent

        # Run inference
        self.run_inference()

        tokens = self.stateful_model.tokens()[B0]

        n_tokens = tokens.shape[0]
        #print(f"Number of Tokens: {n_tokens}")

        model_info = self.stateful_model.model_info()

        # Build contribution graphs
        graphs = cached_build_paths_to_predictions(
            self._graph,
            model_info.n_layers,
            n_tokens,
            range(n_tokens),
            self._contribution_threshold,
        )

        # Run logit lens on various outputs
        #resid_logit_lens_results = self.run_logit_lens_on_resid(model_info.n_layers)
        #output_logit_lens_results = self.run_logit_lens_on_outputs(model_info.n_layers)
        
        # Create sentence analysis dictionary
        sentence_analysis = {
            "sentence": self.sentence,
            "tokens": tokens.tolist(),             
        #    "contributions": self._contributions_dict,
        #    "full_graph": self._graph.copy(),
        #    "token_subgraphs": graphs,
        #    "logit_lens_result": {
        #        "resid": resid_logit_lens_results,
        #        "output": output_logit_lens_results,
        #    }
        }

        if self._do_head_level:
            heads_logit_lens_results = self.run_logit_lens_on_heads(model_info.n_layers, model_info.n_heads, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
            sentence_analysis["logit_lens_result"]["heads"] = heads_logit_lens_results

        # If neuron level analysis is enabled
        if self._do_neuron_level:
            neuron_contributions = self.compute_neuron_contributions(model_info.n_layers)
            top_neuron_contvals, top_neuron_indices = torch.sort(neuron_contributions,descending=True)
            #nonzero_mask = top_neuron_contvals != 0
            #top_neuron_contvals = top_neuron_contvals[nonzero_mask] #cut off non-zero values
            #top_neuron_indices = top_neuron_indices[nonzero_mask]
            #top_neuron_contvals, top_neuron_indices = torch.topk(neuron_contributions, k=self._logit_lens_topK_neurons)
            top_neuron_indices = top_neuron_indices.cpu().numpy()
            top_neuron_contvals = top_neuron_contvals.cpu().to(torch.float16).numpy()

            #n_tokens, n_heads, n_neurons = neuron_contributions.shape
            
            #sel_neurons_layerwise = []
            #for layer in range(model_info.n_layers):
            #    sel_neurons_layerwise.append([])
            #    for token in range(n_tokens):
            #        sel_neurons_layerwise[-1].extend(top_neuron_indices[token, layer].tolist())
            #    sel_neurons_layerwise[-1] = list(set(sel_neurons_layerwise[-1]))

            #n_tokens, n_heads, n_neurons = neuron_contributions.shape
            #all_neuron_indices = torch.arange(n_neurons, device=neuron_contributions.device)#.unsqueeze(0).unsqueeze(0)
            #all_neuron_indices = all_neuron_indices.expand(n_tokens, n_heads, -1)

            # Convert to numpy if needed
            #all_neuron_indices = all_neuron_indices.cpu().numpy()  # Shape: (n_tokens, 32, 11008)


            #run_logit_lens_on_neurons = self.run_logit_lens_on_neurons(
            #    model_info.n_layers,
            #    sel_neurons_layerwise=sel_neurons_layerwise
            #)
            sentence_analysis["neuron_contributions"] = {
                "vals": top_neuron_contvals,
                "ind": top_neuron_indices
            }
            #sentence_analysis["logit_lens_result"]["neurons"] = run_logit_lens_on_neurons

        # Append sentence analysis for this sentence to the relation list

        return sentence_analysis


    def run(self, args):

        self.load_config("/home/kaiwei/llm-transparency-tool/config/exp_olmo_config.json")
        
        self._stateful_model = load_model(
            model_name=self.model_name,
            revision = args.revision,
            prepend_bos=self._prepend_bos,
            _model_path=self._model_path,
            _device=self.device,
            _dtype=self.dtype,
        )
        
        #all_sentence_analyses = []
        
        # Create the base output directory (based on model name)
        base_output_dir = args.output_path
        samples_file = args.dataset_path
        
        # Create the revision-specific directory
        revision_output_dir = os.path.join(base_output_dir, args.revision)

        # samples have format {index: sent}
        processed = [item[2] for item in os.walk(base_output_dir)][0]

        with open(f"{samples_file}","rb") as f:
            all_samples = pickle.load(f)

        for i, sent in tqdm(all_samples.items(),total=1_000_000):
            if f"{i}"+".pkl" in processed:
                print("already processed")
                continue
            else:
                with torch.inference_mode():
                    sentence_analysis = self.process_sentences(sent)
                    with open(f"{base_output_dir}{i}.pkl","wb") as out:
                        pickle.dump(sentence_analysis, out)
                    torch.cuda.empty_cache()
                    del sentence_analysis


            # Skip if the analysis has already been saved
            #if os.path.exists(os.path.join(revision_output_dir, f"{relation}.pkl")):
            #    print(f"Skipping processing for relation '{relation}' as it is already saved.")
            #    continue
            
            # Uncomment the line below if needed to see the count of sentences
            #print(len(sentences))
        # Return or save all_sentence_analyses as needed

        

        

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=str, required=True, help='Model revision to use')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset to use')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for saved pickle file')
    args = parser.parse_args()
    
    
    app = App()
    app.run(args)

