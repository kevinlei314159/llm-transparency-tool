# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import time
from copy import deepcopy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import networkx as nx
# import pandas as pd
# import plotly.express
# import plotly.graph_objects as go
# import streamlit as st
# import streamlit_extras.row as st_row
import torch
from jaxtyping import Float
from torch.amp import autocast
from transformers import HfArgumentParser

# import llm_transparency_tool.components
# from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
import llm_transparency_tool.routes.contributions as contributions
import llm_transparency_tool.routes.graph_off
from llm_transparency_tool.models.transparent_llm import TransparentLlm
from llm_transparency_tool.routes.graph_node import NodeType
from llm_transparency_tool.server.graph_selection import (
    GraphSelection,
    UiGraphEdge,
    UiGraphNode,
)
from llm_transparency_tool.server.utils_off import (
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
    rank_subject: int = 0
    rank_answer: int = 0
    logit_subject: float = 0.0
    logit_answer: float = 0.0
    prob_subject: float = 0.0
    prob_answer: float = 0.0
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
    return llm_transparency_tool.routes.graph_off.build_paths_to_predictions(
        graph, n_layers, n_tokens, starting_tokens, threshold
    )

# @st.cache_resource(
#     hash_funcs={
#         TransformerLensTransparentLlm: id
#     }
# )
def cached_run_inference_and_populate_state(
    stateless_model,
    sentences,
    subject,
    object
):
    # stateful_model = stateless_model.copy()
    stateless_model.run(sentences, subject, object)
    return stateless_model


# @dataclass
# class LlmViewerConfig:
#     debug: bool = field(
#         default=False,
#         metadata={"help": "Show debugging information, like the time profile."},
#     )

#     preloaded_dataset_filename: Optional[str] = field(
#         default=None,
#         metadata={"help": "The name of the text file to load the lines from."},
#     )

#     demo_mode: bool = field(
#         default=False,
#         metadata={"help": "Whether the app should be in the demo mode."},
#     )

#     allow_loading_dataset_files: bool = field(
#         default=True,
#         metadata={"help": "Whether the app should be able to load the dataset files " "on the server side."},
#     )

#     max_user_string_length: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "Limit for the length of user-provided sentences (in characters), " "or None if there is no limit."
#         },
#     )

#     models: Dict[str, str] = field(
#         default_factory=dict,
#         metadata={
#             "help": "Locations of models which are stored locally. Dictionary: official "
#             "HuggingFace name -> path to dir. If None is specified, the model will be"
#             "downloaded from HuggingFace."
#         },
#     )

#     default_model: str = field(
#         default="",
#         metadata={"help": "The model to load once the UI is started."},
#     )


def load_json_files(directory: str, selected_category: str) -> Dict[str, List[Dict]]:
    data = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    category = os.path.basename(root)
                    if category == selected_category or not selected_category:
                        relation = os.path.basename(file).replace('.json', '')
                        #if relation == "country_capital_city":
                        if relation not in data:
                            data[relation] = []
                        data[relation].append(json.load(f))
                    
    return data

def find_subject_token_position(template: str, subject: str) -> int:
    # Format the template with the subject
    fact = template.format(subject)

    # Tokenize the sentence by splitting on whitespace
    tokens = fact.split()

    # Find the position of the subject in the tokenized list (if contained in a token)
    for i, token in enumerate(tokens):
        if subject in token:
            return i
    
    # If subject is not found, raise an error
    raise ValueError(f"Subject '{subject}' not found in the tokenized string.")
    

def parse_samples(data: Dict[str, List[Dict]]) -> Tuple[List[str], List[str], List[str], List[str], List[int]]:
    facts = []
    subjects = []
    targets = []
    sentences = []
    all_subject_positions = []
    indices = []
    subject_object_to_index = {}
    current_index = 0

    for entries in data:
        prompt_templates = entries['prompt_templates'] + entries['prompt_templates_zs']
        samples = entries['samples']
        
        for sample in samples:
            subject = sample['subject']
            obj = " " + sample['object']
            
            # Use the subject-object pair as the unique identifier for indexing
            subject_object_pair = (subject, obj)

            if subject_object_pair not in subject_object_to_index:
                subject_object_to_index[subject_object_pair] = current_index
                current_index += 1
            
            # Get the index for this subject-object pair
            fact_index = subject_object_to_index[subject_object_pair]
            
            for template in prompt_templates:  # Iterate through multiple templates
                subject_pos = find_subject_token_position(template, "{}")
                
                fact = template.format(subject)
                facts.append(fact)
                if subject_pos != 0:
                    subjects.append(" " + subject)
                else:
                    subjects.append(subject)
                targets.append(obj)
                sentences.append(fact + obj)
                all_subject_positions.append(subject_pos)
                indices.append(fact_index)  # Append the index corresponding to this subject-object pair
    
    return sentences, facts, subjects, targets, all_subject_positions, indices

def save_analysis_per_relation(data, relation, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename for each relation
    filename = os.path.join(output_dir, f"{relation}.pkl")
    
    # Save the relation analysis to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Analysis for relation '{relation}' saved to {filename}")
    
# Function to find the span indices in a tensor

def find_span_indices(tokens, subj_tokens):
    # Get the length of the span to match
    span_len = len(subj_tokens)
    # Sliding window over the tokens to find where the subj_tokens match
    for i in range(len(tokens) - span_len + 1):
        if torch.equal(tokens[i:i+span_len], subj_tokens):
            return list(range(i, i + span_len))
    return None
    

class App:
    _stateful_model: TransparentLlm = None
    _graph: Optional[nx.Graph] = None
    _contribution_threshold: float = 0.0
    _renormalize_after_threshold: bool = False
    _normalize_before_unembedding: bool = True

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

    # def draw_model_info(self):
    #     info = self.stateful_model.model_info().__dict__
    #     df = pd.DataFrame(
    #         data=[str(x) for x in info.values()],
    #         index=info.keys(),
    #         columns=["Model parameter"],
    #     )
    #     st.dataframe(df, use_container_width=False)

    # def draw_dataset_selection(self) -> int:
    #     def update_dataset(filename: Optional[str]):
    #         dataset = load_dataset(filename) if filename is not None else []
    #         st.session_state["dataset"] = dataset
    #         st.session_state["dataset_file"] = filename

    #     # if "dataset" not in st.session_state:
    #     #     update_dataset(self._config.preloaded_dataset_filename)


    #     if not self._config.demo_mode:
    #         if self._config.allow_loading_dataset_files:
    #             row_f = st_row.row([2, 1], vertical_align="bottom")
    #             filename = row_f.text_input("Dataset", value=st.session_state.dataset_file or "")
    #             if row_f.button("Load"):
    #                 update_dataset(filename)
    #         row_s = st_row.row([2, 1], vertical_align="bottom")
    #         new_sentence = row_s.text_input("New sentence")
    #         new_sentence_added = False

    #         if row_s.button("Add"):
    #             max_len = self._config.max_user_string_length
    #             n = len(new_sentence)
    #             if max_len is None or n <= max_len:
    #                 st.session_state.dataset.append(new_sentence)
    #                 new_sentence_added = True
    #                 st.session_state.sentence_selector = new_sentence
    #             else:
    #                 st.warning(f"Sentence length {n} is larger than " f"the configured limit of {max_len}")

    #     sentences = st.session_state.dataset
    #     selection = st.selectbox(
    #         "Sentence",
    #         sentences,
    #         index=len(sentences) - 1,
    #         key="sentence_selector",
    #     )
    #     return selection
    
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.model_name = config.get("model_name", "allenai/OLMo-7B-0424-hf")
        self.device = config.get("device", "gpu")
        self._model_path = config.get("_model_path", None) 
        dtype_str = config.get("dtype", "torch.bfloat16")
        self.dtype = getattr(torch, dtype_str, torch.bfloat16)
        self.amp_enabled = config.get("amp_enabled", True)
        self._renormalize_after_threshold = config.get("renormalize_after_threshold", True)
        self._normalize_before_unembedding = config.get("normalize_before_unembedding", True)
        self._prepend_bos = config.get("prepend_bos", False)
        self._do_neuron_level = config.get("do_neuron_level", False)
        self._contribution_threshold = config.get("contribution_threshold", 0.01)
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

    # def draw_token_matrix(
    #     self,
    #     values: Float[torch.Tensor, "t t"],
    #     tokens: List[str],
    #     value_name: str,
    #     title: str,
    # ):
    #     assert values.shape[0] == len(tokens)
    #     labels = {
    #         "x": "<b>src</b>",
    #         "y": "<b>tgt</b>",
    #         "color": value_name,
    #     }

    #     captions = [f"({i}){t}" for i, t in enumerate(tokens)]

    #     fig = plotly.express.imshow(
    #         values.cpu(),
    #         title=f'<b>{title}</b>',
    #         labels=labels,
    #         x=captions,
    #         y=captions,
    #         color_continuous_scale=self.render_settings.attention_color_map,
    #         aspect="equal",
    #     )
    #     fig.update_layout(
    #         autosize=True,
    #         margin=go.layout.Margin(
    #             l=50,  # left margin
    #             r=0,  # right margin
    #             b=100,  # bottom margin
    #             t=100,  # top margin
    #             # pad=10  # padding
    #         )
    #     )
    #     fig.update_xaxes(tickmode="linear")
    #     fig.update_yaxes(tickmode="linear")
    #     fig.update_coloraxes(showscale=False)

    #     st.plotly_chart(fig, use_container_width=True, theme=None)

    # def draw_attn_info(self, edge: UiGraphEdge, container_attention_map) -> Optional[int]:
    #     """
    #     Returns: the index of the selected head.
    #     """

    #     n_heads = self.stateful_model.model_info().n_heads

    #     layer = edge.target.layer

    #     head_contrib, _ = contributions.get_attention_contributions(
    #         resid_pre=self.stateful_model.residual_in(layer)[B0].unsqueeze(0),
    #         resid_mid=self.stateful_model.residual_after_attn(layer)[B0].unsqueeze(0),
    #         decomposed_attn=self.stateful_model.decomposed_attn(B0, layer).unsqueeze(0),
    #     )

    #     # [batch pos key_pos head] -> [head]
    #     flat_contrib = head_contrib[0, edge.target.token, edge.source.token, :]
    #     assert flat_contrib.shape[0] == n_heads, f"{flat_contrib.shape} vs {n_heads}"

    #     selected_head = llm_transparency_tool.components.selector(
    #         items=[f"H{h}" if h >= 0 else "All" for h in range(-1, n_heads)],
    #         indices=range(-1, n_heads),
    #         temperatures=[sum(flat_contrib).item()] + flat_contrib.tolist(),
    #         preselected_index=flat_contrib.argmax().item(),
    #         key=f"head_selector_layer_{layer}" #_from_tok_{edge.source.token}_to_tok_{edge.target.token}",
    #     )
    #     print(f"head_selector_layer_{layer}_from_tok_{edge.source.token}_to_tok_{edge.target.token}")
    #     if selected_head == -1 or selected_head is None:
    #         # selected_head = None
    #         selected_head = flat_contrib.argmax().item()
    #         print('****\n' * 3 + f"selected_head: {selected_head}" + '\n****\n' * 3)

    #     # Draw attention matrix and contributions for the selected head.
    #     if selected_head is not None:
    #         tokens = [
    #             string_to_display(s) for s in self.stateful_model.tokens_to_strings(self.stateful_model.tokens()[B0])
    #         ]

    #         with container_attention_map:
    #             attn_container, contrib_container = st.columns([1, 1])
    #             with attn_container:
    #                 attn = self.stateful_model.attention_matrix(B0, layer, selected_head)
    #                 self.draw_token_matrix(
    #                     attn,
    #                     tokens,
    #                     "attention",
    #                     f"Attention map L{layer} H{selected_head}",
    #                 )
    #             with contrib_container:
    #                 contrib = head_contrib[B0, :, :, selected_head]
    #                 self.draw_token_matrix(
    #                     contrib,
    #                     tokens,
    #                     "contribution",
    #                     f"Contribution map L{layer} H{selected_head}",
    #                 )

    #     return selected_head


    # TODO: Maybe also add neuron-level?
    # def draw_ffn_info(self, node: UiGraphNode) -> Optional[int]:
    #     """
    #     Returns: the index of the selected neuron.
    #     """

    #     resid_mid = self.stateful_model.residual_after_attn(node.layer)[B0][node.token]
    #     resid_post = self.stateful_model.residual_out(node.layer)[B0][node.token]
    #     decomposed_ffn = self.stateful_model.decomposed_ffn_out(B0, node.layer, node.token)
    #     c_ffn, _ = contributions.get_decomposed_mlp_contributions(resid_mid, resid_post, decomposed_ffn)

    #     top_values, top_i = c_ffn.sort(descending=True)
    #     n = min(self.render_settings.n_top_neurons, c_ffn.shape[0])
    #     top_neurons = top_i[0:n].tolist()

    #     selected_neuron = llm_transparency_tool.components.selector(
    #         items=[f"{top_neurons[i]}" if i >= 0 else "All" for i in range(-1, n)],
    #         indices=range(-1, n),
    #         temperatures=[0.0] + top_values[0:n].tolist(),
    #         preselected_index=-1,
    #         key="neuron_selector",
    #     )
    #     if selected_neuron is None:
    #         selected_neuron = -1
    #     selected_neuron = None if selected_neuron == -1 else top_neurons[selected_neuron]

    #     return selected_neuron

    # def _draw_token_table(
    #     self,
    #     n_top: int,
    #     n_bottom: int,
    #     representation: torch.Tensor,
    #     predecessor: Optional[torch.Tensor] = None,
    # ):
    #     n_total = n_top + n_bottom

    #     logits = self._unembed(representation)
    #     n_vocab = logits.shape[0]
    #     scores, indices = torch.topk(logits, n_top, largest=True)
    #     positions = list(range(n_top))

    #     if n_bottom > 0:
    #         low_scores, low_indices = torch.topk(logits, n_bottom, largest=False)
    #         indices = torch.cat((indices, low_indices.flip(0)))
    #         scores = torch.cat((scores, low_scores.flip(0)))
    #         positions += range(n_vocab - n_bottom, n_vocab)

    #     tokens = [string_to_display(w) for w in self.stateful_model.tokens_to_strings(indices)]

    #     if predecessor is not None:
    #         pre_logits = self._unembed(predecessor)
    #         _, sorted_pre_indices = pre_logits.sort(descending=True)
    #         pre_indices_dict = {index: pos for pos, index in enumerate(sorted_pre_indices.tolist())}
    #         old_positions = [pre_indices_dict[i] for i in indices.tolist()]

    #         def pos_gain_string(pos, old_pos):
    #             if pos == old_pos:
    #                 return ""
    #             sign = "↓" if pos > old_pos else "↑"
    #             return f"({sign}{abs(pos - old_pos)})"

    #         position_strings = [f"{i} {pos_gain_string(i, old_i)}" for (i, old_i) in zip(positions, old_positions)]
    #     else:
    #         position_strings = [str(pos) for pos in positions]

    #     def pos_gain_color(s):
    #         color = "black"
    #         if isinstance(s, str):
    #             if "↓" in s:
    #                 color = "red"
    #             if "↑" in s:
    #                 color = "green"
    #         return f"color: {color}"

    #     top_df = pd.DataFrame(
    #         data=zip(position_strings, tokens, scores.tolist()),
    #         columns=["Pos", "Token", "Score"],
    #     )

    #     st.dataframe(
    #         top_df.style.map(pos_gain_color)
    #         .background_gradient(
    #             axis=0,
    #             cmap=logits_color_map(positive_and_negative=n_bottom > 0),
    #         )
    #         .format(precision=3),
    #         hide_index=True,
    #         height=self.render_settings.table_cell_height * (n_total + 1),
    #         use_container_width=True,
    #     )

    # def draw_token_dynamics(self, representation: torch.Tensor, block_name: str) -> None:
    #     st.caption(block_name)
    #     self._draw_token_table(
    #         self.render_settings.n_promoted_tokens,
    #         self.render_settings.n_suppressed_tokens,
    #         representation,
    #         None,
    #     )

    # def draw_top_tokens(
    #     self,
    #     node: UiGraphNode,
    #     container_top_tokens,
    #     container_token_dynamics,
    # ) -> None:
    #     pre_node = node.get_residual_predecessor()
    #     if pre_node is None:
    #         return

    #     representation = self._get_representation(node)
    #     predecessor = self._get_representation(pre_node)

    #     with container_top_tokens:
    #         st.caption(node.get_name())
    #         self._draw_token_table(
    #             self.render_settings.n_top_tokens,
    #             0,
    #             representation,
    #             predecessor,
    #         )
    #     if container_token_dynamics is not None:
    #         with container_token_dynamics:
    #             self.draw_token_dynamics(representation - predecessor, node.get_predecessor_block_name())

    # def draw_attention_dynamics(self, node: UiGraphNode, head: Optional[int]):
    #     block_name = node.get_head_name(head)
    #     block_output = (
    #         self.stateful_model.attention_output_per_head(B0, node.layer, node.token, head)
    #         if head is not None
    #         else self.stateful_model.attention_output(B0, node.layer, node.token)
    #     )
    #     self.draw_token_dynamics(block_output, block_name)

    # def draw_ffn_dynamics(self, node: UiGraphNode, neuron: Optional[int]):
    #     block_name = node.get_neuron_name(neuron)
    #     block_output = (
    #         self.stateful_model.neuron_output(node.layer, neuron)
    #         if neuron is not None
    #         else self.stateful_model.ffn_out(node.layer)[B0][node.token]
    #     )
    #     self.draw_token_dynamics(block_output, block_name)

    # def draw_precision_controls(self, device: str) -> Tuple[torch.dtype, bool]:
    #     """
    #     Draw fp16/fp32 switch and AMP control.

    #     return: The selected precision and whether AMP should be enabled.
    #     """

    #     if device == "cpu":
    #         dtype = torch.float32
    #     else:
    #         dtype = st.selectbox(
    #             "Precision",
    #             [torch.float16, torch.bfloat16, torch.float32],
    #             index=0,
    #         )

    #     amp_enabled = dtype != torch.float32

    #     return dtype, amp_enabled

    # def draw_controls(self):
    #     # model_container, data_container = st.columns([1, 1])
    #     with st.sidebar.expander("Model", expanded=True):
    #         list_of_devices = possible_devices()
    #         if len(list_of_devices) > 1:
    #             self.device = st.selectbox(
    #                 "Device",
    #                 possible_devices(),
    #                 index=0,
    #             )
    #         else:
    #             self.device = list_of_devices[0]

    #         self.dtype, self.amp_enabled = self.draw_precision_controls(self.device)

    #         model_list = list(self._config.models)
    #         default_choice = model_list.index(self._config.default_model)

    #         self.model_name = st.selectbox(
    #             "Model",
    #             model_list,
    #             index=default_choice,
    #         )

    #         if self.model_name:
    #             self._stateful_model = load_model(
    #                 model_name=self.model_name,
    #                 _model_path=self._config.models[self.model_name],
    #                 _device=self.device,
    #                 _dtype=self.dtype,
    #             )
    #             self.model_key = self.model_name  # TODO maybe something else?
    #             self.draw_model_info()

    #     self.sentence = self.draw_dataset_selection()

    #     with st.sidebar.expander("Graph", expanded=True):
    #         self._contribution_threshold = st.slider(
    #             min_value=0.01,
    #             max_value=0.1,
    #             step=0.01,
    #             value=0.04,
    #             format=r"%.3f",
    #             label="Contribution threshold",
    #         )
    #         self._renormalize_after_threshold = st.checkbox("Renormalize after threshold", value=True)
    #         self._normalize_before_unembedding = st.checkbox("Normalize before unembedding", value=True)

    def run_inference(self):

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._stateful_model = cached_run_inference_and_populate_state(self.stateful_model, [self.sentence], self.subj_token, self.obj_token)

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._graph, self._contributions_dict = get_contribution_graph(
                self.stateful_model,
                "",
                self.stateful_model.tokens()[B0].tolist(),
                (self._contribution_threshold if self._renormalize_after_threshold else 0.0),
            )

    # def draw_graph_and_selection(
    #     self,
    # ) -> None:
    #     (
    #         container_graph,
    #         container_tokens,
    #     ) = st.columns(self.render_settings.column_proportions)

    #     container_graph_left, container_graph_right = container_graph.columns([5, 1])

    #     container_graph_left.write('##### Graph')
    #     heads_placeholder = container_graph_right.empty()
    #     heads_placeholder.write('##### Blocks')
    #     container_graph_right_used = False

    #     container_top_tokens, container_token_dynamics = container_tokens.columns([1, 1])
    #     container_top_tokens.write('##### Top Tokens')
    #     container_top_tokens_used = False
    #     container_token_dynamics.write('##### Promoted Tokens')
    #     container_token_dynamics_used = False

    #     try:

    #         if self.sentence is None:
    #             return

    #         with container_graph_left:
    #             selection = self.draw_graph(self._contribution_threshold if not self._renormalize_after_threshold else 0.0)

    #         if selection is None:
    #             return

    #         node = selection.node
    #         edge = selection.edge

    #         if edge is not None and edge.target.type == NodeType.AFTER_ATTN:
    #             with container_graph_right:
    #                 container_graph_right_used = True
    #                 heads_placeholder.write('##### Heads')
    #                 head = self.draw_attn_info(edge, container_graph)
    #             with container_token_dynamics:
    #                 self.draw_attention_dynamics(edge.target, head)
    #                 container_token_dynamics_used = True
    #         elif node is not None and node.type == NodeType.FFN:
    #             with container_graph_right:
    #                 container_graph_right_used = True
    #                 heads_placeholder.write('##### Neurons')
    #                 neuron = self.draw_ffn_info(node)
    #             with container_token_dynamics:
    #                 self.draw_ffn_dynamics(node, neuron)
    #                 container_token_dynamics_used = True

    #         if node is not None and node.is_in_residual_stream():
    #             self.draw_top_tokens(
    #                 node,
    #                 container_top_tokens,
    #                 container_token_dynamics if not container_token_dynamics_used else None,
    #             )
    #             container_top_tokens_used = True
    #             container_token_dynamics_used = True
    #     finally:
    #         if not container_graph_right_used:
    #             st_placeholder('Click on an edge to see head contributions. \n\n'
    #                            'Or click on FFN to see individual neuron contributions.', container_graph_right, height=1100)
    #         if not container_top_tokens_used:
    #             st_placeholder('Select a node from residual stream to see its top tokens.', container_top_tokens, height=1100)
    #         if not container_token_dynamics_used:
    #             st_placeholder('Select a node to see its promoted tokens.', container_token_dynamics, height=1100)

    #@profile
    def process_logits(self, logit_scores, subj_voc_id=None, answer_voc_id=None):
        tokens = self.stateful_model.tokens()[B0]
        #s2 = time.time()
        #tokens = self.stateful_model.tokens()[B0]
        probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
        entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
        sorted_indices = torch.argsort(logit_scores, dim=-1, descending=True)
        rank_subj = (sorted_indices == subj_voc_id).nonzero(as_tuple=True)[1].cpu() # probably one?!
        rank_ans = (sorted_indices == answer_voc_id).nonzero(as_tuple=True)[1].cpu()
        results = []
        
        for token_idx in range(len(entropy)):
            result = {}
            result["token_idx"] = token_idx
            # tok = self.stateful_model.tokens_to_strings(sorted_indices[token_idx][:self._logit_lens_topK].cpu())
            result["top_tokens"] = [int(i) for i in sorted_indices[token_idx][:self._logit_lens_topK].cpu()]
            # result["rank_subject"] = get_val((sorted_indices[token_idx] == subj_voc_id).nonzero(as_tuple=True)[0])
            # result["rank_answer"] = get_val((sorted_indices[token_idx] == answer_voc_id).nonzero(as_tuple=True)[0])
            result["rank_subject"] = get_val(rank_subj[token_idx])
            result["rank_answer"] = get_val(rank_ans[token_idx])
            result["logit_subject"] = get_val(logit_scores[token_idx][subj_voc_id].cpu())
            result["logit_answer"] = get_val(logit_scores[token_idx][answer_voc_id].cpu())
            result["prob_subject"] = get_val(probs[token_idx][subj_voc_id])
            result["prob_answer"] = get_val(probs[token_idx][answer_voc_id])
            result["max_logit"] = get_val(logit_scores[token_idx].max().cpu())
            result["max_prob"] = get_val(probs[token_idx].max())
            result["entropy"] = get_val(entropy[token_idx])
            # result = LogLensResult()
            # result.token_idx = token_idx
            # result.top_tokens = [w for w in self.stateful_model.tokens_to_strings(sorted_indices[token_idx][:self._logit_lens_topK])]
            # result.rank_subject = get_val(torch.argwhere(sorted_indices[token_idx] == subj_voc_id))
            # result.rank_answer = get_val(torch.argwhere(sorted_indices[token_idx] == answer_voc_id))
            # result.logit_subject = get_val(logit_scores[token_idx][subj_voc_id].cpu())
            # result.logit_answer = get_val(logit_scores[token_idx][answer_voc_id].cpu())
            # result.prob_subject = get_val(probs[token_idx][subj_voc_id])
            # result.prob_answer = get_val(probs[token_idx][answer_voc_id])
            # result.max_logit = get_val(logit_scores[token_idx].max().cpu())
            # result.max_prob = get_val(probs[token_idx].max())
            # result.entropy = get_val(entropy[token_idx])
            results.append(result)
        return results
        

    @torch.no_grad()
    def run_logit_lens_on_resid(self, n_layers, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        for layer in range(n_layers):
            for resid_pos in ["pre", "mid"]:
                if resid_pos == "pre":
                    representations = self.stateful_model.residual_in(layer)
                else:
                    representations = self.stateful_model.residual_after_attn(layer)
                logit_scores = self._unembed(representations)[B0]
                hook_name = f"{layer}_{resid_pos}"
                full_results[hook_name] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        representations = self.stateful_model.residual_out(n_layers-1)
        logit_scores = self._unembed(representations)[B0]
        full_results["final_post"] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        return full_results
    

    @torch.no_grad()
    def run_logit_lens_on_outputs(self, n_layers, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        representations = self.stateful_model.residual_in(0)
        logit_scores = self._unembed(representations)[B0]
        full_results["embed"] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        for layer in range(n_layers):
            for resid_pos in ["attn", "mlp"]:
                if resid_pos == "attn":
                    representations = self.stateful_model._get_block(layer, "hook_attn_out")
                else:
                    representations = self.stateful_model.ffn_out(layer)
                logit_scores = self._unembed(representations)[B0]
                hook_name = f"{layer}_{resid_pos}_out"
                full_results[hook_name] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        return full_results
    

    @torch.no_grad()
    def run_logit_lens_on_heads(self, n_layers, n_heads, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        for layer in range(n_layers):
            representations = self.stateful_model._get_block(layer, "attn.hook_result")
            for head in range(n_heads):
                hook_name = f"L{layer}H{head}"
                logit_scores = self._unembed(representations[:, :, head])[B0]
                full_results[hook_name] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        return full_results
    
        # representations = self.stateful_model._get_block(layer, "attn.hook_result")
        # representations_shape = representations.shape
        # # representations: [B0, token_len, head, dim]
        # representations = torch.reshape(representations, (representations_shape[0], -1, representations_shape[-1]))
        # # representations: [B0, token_len * head, dim]
        # logit_scores = self._unembed(representations[:, :])[B0]
        # logit_scores = torch.reshape(logit_scores, (representations_shape[1], representations_shape[2], -1))
        # for head in range(n_heads):
        #     hook_name = f"L{layer}H{head}"
        #     full_results[hook_name] = self.process_logits(logit_scores[:, head], subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
    
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
                c_ffn, _ = contributions.get_decomposed_mlp_contributions(resid_mid[token], resid_post[token], decomposed_ffn[token])
                results.append(c_ffn)
            ffn_contributions.append(torch.stack(results))
        return torch.stack(ffn_contributions).transpose(1, 0)


    @torch.no_grad()
    def run_logit_lens_on_neurons(self, n_layers, sel_neurons_layerwise, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        for layer in range(n_layers):
            sel_neurons = sel_neurons_layerwise[layer]
            for neuron in sel_neurons:
                hook_name = f"L{layer}N{neuron}"
                representations = self.stateful_model.neuron_output(layer, neuron).unsqueeze(0).unsqueeze(1)
                logit_scores = self._unembed(representations)[B0][0]
                probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
                entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
                sorted_indices = torch.argsort(logit_scores, dim=-1, descending=True).cpu()

                result = {}
                #result["top_tokens"] = [w for w in self.stateful_model.tokens_to_strings(sorted_indices[:self._logit_lens_topK])]
                result["top_tokens"] = [int(i) for i in sorted_indices[:self._logit_lens_topK].cpu()]
                result["rank_subject"] = get_val((sorted_indices == subj_voc_id).nonzero(as_tuple=True)[0])
                result["rank_answer"] = get_val((sorted_indices == answer_voc_id).nonzero(as_tuple=True)[0])
                result["logit_subject"] = get_val(logit_scores[subj_voc_id].cpu())
                result["logit_answer"] = get_val(logit_scores[answer_voc_id].cpu())
                result["prob_subject"] = get_val(probs[subj_voc_id])
                result["prob_answer"] = get_val(probs[answer_voc_id])
                result["max_logit"] = get_val(logit_scores.max().cpu())
                result["max_prob"] = get_val(probs.max().cpu())
                result["entropy"] = get_val(entropy)

                full_results[hook_name] = result
        return full_results


    def process_sentences(self, sentences, all_subject_positions, facts, subjects, targets, relation, indices):
        relation_sentence_analyses = []  # List to hold analyses for the current relation
        print(f"Relation {relation}")
        for idx, sent in tqdm(enumerate(sentences)):
            start_time = time.time()
            self.sentence = sent
            self.subj_token = subjects[idx] 
            self.obj_token = targets[idx]

            # Run inference
            self.run_inference()

            tokens = self.stateful_model.tokens()[B0]
            subj_tokens = self.stateful_model.subj_tokens()[B0]
            obj_tokens = self.stateful_model.obj_token()[B0]
            n_tokens = tokens.shape[0]
            #print(f"Number of Tokens: {n_tokens}")
            
            subj_token_span = find_span_indices(tokens, subj_tokens)
            obj_token_span = find_span_indices(tokens, obj_tokens)

            subj_voc_id = subj_tokens[0].cpu() #tokens[all_subject_positions[idx]].cpu()
            answer_voc_id = obj_tokens[0].cpu() #tokens[-1].cpu()

            model_info = self.stateful_model.model_info()

            # Build contribution graphs
            graphs = cached_build_paths_to_predictions(
                self._graph,
                model_info.n_layers,
                n_tokens,
                range(n_tokens),
                self._contribution_threshold,
            )
            #print(f"Done with Graph {time.time() - start_time}")

            # Run logit lens on various outputs
            resid_logit_lens_results = self.run_logit_lens_on_resid(model_info.n_layers, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
            output_logit_lens_results = self.run_logit_lens_on_outputs(model_info.n_layers, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
            #heads_logit_lens_results = self.run_logit_lens_on_heads(model_info.n_layers, model_info.n_heads, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id, subj_token_span=subj_token_span, obj_token_span=obj_token_span)            #print(f"Done with MLP & Attn Out {time.time() - start_time}")
            #print(f"Done with Heads {time.time() - start_time}")
            # Create sentence analysis dictionary
            sentence_analysis = {
                "sentence": self.sentence,
                "tokens": tokens.tolist(),
                "subject_tokens": subj_tokens.tolist(),
                "answer_tokens": obj_tokens.tolist(),
                "data_idx": indices[idx],
                "subj_token_span": subj_token_span,   
                "answer_token_span": obj_token_span,               
                "contributions": self._contributions_dict,
                "full_graph": self._graph.copy(),
                "token_subgraphs": graphs,
                "logit_lens_result": {
                    "resid": resid_logit_lens_results,
                    "output": output_logit_lens_results,
                    #"heads": heads_logit_lens_results,
                }
            }

            # If neuron level analysis is enabled
            if self._do_neuron_level:
                neuron_contributions = self.compute_neuron_contributions(model_info.n_layers)
                top_neuron_contvals, top_neuron_indices = torch.topk(neuron_contributions, k=self._logit_lens_topK_neurons)
                top_neuron_indices = top_neuron_indices.cpu().numpy()
                top_neuron_contvals = top_neuron_contvals.cpu().to(torch.float16).numpy()

                sel_neurons_layerwise = []
                for layer in range(model_info.n_layers):
                    sel_neurons_layerwise.append([])
                    for token in range(n_tokens):
                        sel_neurons_layerwise[-1].extend(top_neuron_indices[token, layer].tolist())
                    sel_neurons_layerwise[-1] = list(set(sel_neurons_layerwise[-1]))

                run_logit_lens_on_neurons = self.run_logit_lens_on_neurons(
                    model_info.n_layers,
                    sel_neurons_layerwise=sel_neurons_layerwise,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id
                )
                #print(f"Done with Neurons {time.time() - start_time}")
                sentence_analysis["neuron_contributions"] = {
                    "vals": top_neuron_contvals,
                    "ind": top_neuron_indices
                }
                sentence_analysis["logit_lens_result"]["neurons"] = run_logit_lens_on_neurons

            # Append sentence analysis for this sentence to the relation list
            relation_sentence_analyses.append(sentence_analysis)

        return relation_sentence_analyses


    def run(self, args):
        # TODO: Initialize the following parameters
        # self.model_name = "allenai/OLMo-7B-0424-hf"
        # self.device = "gpu"
        # self.dtype = torch.bfloat16
        # self._model_path = None

        # self.amp_enabled = self.dtype != torch.float32

        # self._renormalize_after_threshold = True
        # self._normalize_before_unembedding = True
        # self._prepend_bos = False

        # self._do_neuron_level = False

        # self._contribution_threshold = 0.01

        # self._logit_lens_topK = 10
        # self._logit_lens_topK_neurons = 10
        self.load_config("/mounts/data/proj/hypersum/llm-transparency-tool/config/exp_olmo_config.json")
        
        self._stateful_model = load_model(
            model_name=self.model_name,
            revision = args.revision,
            prepend_bos=self._prepend_bos,
            _model_path=self._model_path,
            _device=self.device,
            _dtype=self.dtype,
        )
        
        relations = load_json_files(args.dataset_path, args.category)
        all_sentence_analyses = {}
        
        # Create the base output directory (based on model name)
        base_output_dir = args.output_path
        
        # Create the revision-specific directory
        revision_output_dir = os.path.join(base_output_dir, args.revision)

        with torch.inference_mode():
            # If a specific relation is provided, process only that relation
            if args.relation:
                sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relations[args.relation])
                #print(len(sentences))
                relation_sentence_analyses = self.process_sentences(
                    sentences, all_subject_positions, facts, subjects, targets, args.relation, indices
                )
                all_sentence_analyses[args.relation] = relation_sentence_analyses
                save_analysis_per_relation(relation_sentence_analyses, args.relation, revision_output_dir)
            else:
                # Process all relations
                for relation, relation_data in tqdm(relations.items(), desc="Processing Relations"):
                    #print(f"Relation {relation}")
                    sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relation_data)
                    print(len(sentences))
                    relation_sentence_analyses = self.process_sentences(
                        sentences, all_subject_positions, facts, subjects, targets, relation, indices
                    )
                    all_sentence_analyses[relation] = relation_sentence_analyses
                    
                    save_analysis_per_relation(relation_sentence_analyses, relation, revision_output_dir)

        # Return or save all_sentence_analyses as needed
        return all_sentence_analyses
                    
                    
            # sentence_analysis = {
            #     "contributions": self._contributions_dict,
            #     "full_graph": self._graph.copy(),
            #     "token_subgraphs": graphs,
            #     "neuron_contributions": {"vals": top_neuron_contvals, "ind": top_neuron_indices},
            #     "logit_lens_result": {
            #         "resid": resid_logit_lens_results,
            #         "output": output_logit_lens_results,
            #         "heads": heads_logit_lens_results,
            #         "neurons": run_logit_lens_on_neurons
            #     }
            # }
            
        # with st.sidebar.expander("About", expanded=True):
        #     if self._config.demo_mode:
        #         st.caption("""
        #             The app is deployed in Demo Mode, thus only predefined models and inputs are available.\n
        #             You can still install the app locally and use your own models and inputs.\n
        #             See https://github.com/facebookresearch/llm-transparency-tool for more information.
        #         """)

        # self.draw_controls()

        # if not self.model_name:
        #     st.warning("No model selected")
        #     st.stop()

        # if self.sentence is None:
        #     st.warning("No sentence selected")
        # else:
        #     with torch.inference_mode():
        #         self.run_inference()

        # self.sentence = "Berlin is the capital of Germany"

        # with torch.inference_mode():
        #     self.run_inference()

        #     tokens = self.stateful_model.tokens()[B0]
        #     n_tokens = tokens.shape[0]
        #     model_info = self.stateful_model.model_info()

        #     graphs = cached_build_paths_to_predictions(
        #         self._graph,
        #         model_info.n_layers,
        #         n_tokens,
        #         range(n_tokens),
        #         self._contribution_threshold,
        #     )

        
        
        print("Done")

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=str, required=True, help='Model revision to use')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset to use')
    parser.add_argument('--category', type=str, required=False, help='Select from Main Categories: Factual, Commonsense, Bias, Linguistic')
    parser.add_argument('--relation', type=str, required=False, help='Specific relation from Category')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for saved pickle file')
    args = parser.parse_args()
    

    # top_parser = argparse.ArgumentParser()
    # top_parser.add_argument("config_file")
    # args = top_parser.parse_args()

    # parser = HfArgumentParser([LlmViewerConfig])
    # config = parser.parse_json_file(args.config_file)[0]

    # with SystemMonitor(config.debug) as prof:
    app = App()
    app.run(args)

