# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os

import torch
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from sentence_transformers import SentenceTransformer
import faiss
import queue
import numpy as np

from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import get_indexed_dataset_


try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def get_params(cfg): 
    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
    }
    return length_params, sampling_params

# def process_sentence_chunks(prompts, tokenizer, encoder_seq_length):
#     queue = queue.Queue()
#     for prompt in prompts: 
#         ids = tokenizer.text_to_ids(prompt)
#         if len(ids) > self.cfg.encoder_seq_length: 
#             ids = ids[: self.cfg.encoder_seq_length]

#         for start in range(0, self.cfg.encoder_seq_length, self.cfg.chunk_size):
#             end = start + self.cfg.chunk_size
#             sentences = [self.tokenizer.ids_to_text(id) for id in ids[start:end]]
#             self.queue.put(sentences)
#     self.queue.put(None)

@hydra_runner(config_path="conf", config_name="megatron_retro_inference")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=NLPDDPPlugin(), **cfg.trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.retro_model_file: 
        model = MegatronRetrievalModel.restore_from(restore_path=cfg.retro_model_file, trainer=trainer)
    else:
        raise ValueError("need a nemo file or checkpoint dir")

    if cfg.faiss_index_path: 
        faiss_index = faiss.read_index(cfg.faiss_index_path)
    else: 
        raise ValueError("need a faiss index")
    retrieval_index = get_indexed_dataset_(cfg.model.data.retrieval_prefix, cfg.model.data.data_impl, cfg.model.data.skip_warmup)

    sentence_transformer = SentenceTransformer('bert-base-nli-mean-tokens')

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    try:
        model.frozen_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    length_params, sampling_params = get_params(cfg)
    input_ids = []
    
    for prompt in cfg.prompts: 
        retrieved_docs = []
        ids = model.tokenizer.text_to_ids(prompt)
        batch = []
        if cfg.model.data.neighbors > 0:
            for start in range(0, len(ids), model.cfg.chunk_size):
                end = min(start + model.cfg.chunk_size, len(ids))
                chunk = [model.tokenizer.ids_to_text(ids[start:end])]
                emb = sentence_transformer.encode(sentences=chunk)
                D, I = faiss_index.search(emb, cfg.model.data.neighbors)
                rids = I[0]
                retrieved_doc = [model.tokenizer.ids_to_text(retrieval_index.get_chunk(rid)) for rid in rids]
                batch.append(retrieved_doc)
            if len(ids) % model.cfg.chunk_size: 
                batch.pop(-1)
        retrieved_docs.append(batch)
        # ids_tensor = torch.Tensor([ids]).to(dtype = torch.long, device = 'cuda')
        # ids_att_tensor = torch.Tensor([[1] * len(ids)]).to(dtype = torch.long, device = 'cuda')
        # retrieved_id_tensor = torch.Tensor([batch]).to(dtype = torch.long, device = 'cuda')
        # retrieved_att_tensor = torch.Tensor(np.ones_like(np.asarray([batch], dtype=np.int64))).to(dtype = torch.long, device = 'cuda')
        # print(OmegaConf.to_container(cfg.prompts))
        # print(retrieved_docs)
        response = model.generate([prompt], retrieved_docs, length_params, sampling_params)
        # print(response.keys())
        print(response['sentences'])

    


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
