import torch
from vllm import LLM, SamplingParams
from openai import OpenAI
from utils.crystal_generator import CrystalGenerator
from huggingface_hub import login
from utils.config import OPENAI_API_KEY, HF_TOKEN
# from utils.llm_merge import merge_and_save_weights
import os
from vllm.lora.request import LoRARequest

from peft import PeftModel, PeftConfig, LoraConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import torch
import os
import shutil
from typing import Optional
from huggingface_hub import snapshot_download


class LLMManager:
    def __init__(self, model_name, tensor_parallel_size=1, gpu_memory_utilization=0.8, temperature=1.0, max_tokens=200, seed=42):
        login(token=HF_TOKEN)
        self.model_name = model_name
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm = self._initialize_llm(tensor_parallel_size, gpu_memory_utilization)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # self.lora_request = None
        # if "crystalllm" in self.model_name:
        #     sql_lora_path = snapshot_download(repo_id=self.model_name)
        #     self.lora_request = LoRARequest("3.1-8b-lora", 1, sql_lora_path)
            # tokenizer = AutoTokenizer.from_pretrained(
            #     "meta-llama/Llama-3.1-8B-Instruct",
            #     model_max_length=MAX_LENGTH,
            #     padding_side="right",
            #     use_fast=False,
            # )
            # self.sampling_params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens, stop_token_ids=tokenizer)

    def _initialize_llm(self, tensor_parallel_size, gpu_memory_utilization):
        if "gpt" in self.model_name:
            return OpenAI(api_key=OPENAI_API_KEY)
        elif "crystalllm" in self.model_name:
            return LLM(
                model=self.model_name,
                dtype=torch.float16,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=self._get_max_token_length(),
                seed=self.seed,
                # quantization="bitsandbytes",
                # load_format="bitsandbytes",
                # enforce_eager=True,
                trust_remote_code=True,
                max_num_seqs=8,
            )
        else:
            return LLM(
                model=self.model_name,
                dtype=torch.float16,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=self._get_max_token_length(),
                seed=self.seed,
                trust_remote_code=True,
                max_num_seqs=8,
            )

    def _get_max_token_length(self):
        if '70b' in self.model_name:
            return 11000
        elif 'mistral' in self.model_name:
            return 32000
        return 11000
        
    def generate(self, prompts):
        if "gpt" in self.model_name:
            return [self._generate_gpt(prompt) for prompt in prompts]
        elif "flowmm" in self.model_name:
            results = self.llm.generate(prompts)
        else:
            results = self.llm.generate(prompts, self.sampling_params)
            return [output.text for result in results for output in result.outputs]
        
    def _generate_gpt(self, prompt):
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    

