from pathlib import Path
from typing import Union, List, Optional

import openai
import torch
from transformers import AutoTokenizer

# Try importing vllm, warn if missing
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class LLMManagerError(Exception):
    """Base exception for LLMManager errors."""
    pass


class LLMManager:
    def __init__(self,
                 model_path: Union[str, Path] = './llama3',
                 backend: str = "vllm",  # "vllm" or "ollama"
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.8,
                 temperature: float = 0.95,
                 max_tokens: int = 4096,
                 seed: int = 42,
                 api_base: str = "http://localhost:11434/v1",
                 model_name: Optional[str] = None,
                 chat_template_style: Optional[str] = None
                 ):
        """
        Manages LLM generation using either vLLM (HF Transformers) or Ollama (GGUF + OpenAI API).

        Args:
            model_path: Path to local model directory (for vLLM) or dummy path (for Ollama).
            backend: One of ["vllm", "ollama"].
            tensor_parallel_size: Used for vLLM.
            gpu_memory_utilization: Used for vLLM.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            seed: Random seed (used by vLLM or passed to API).
            api_base: Ollama/OpenAI-compatible API base.
            model_name: Model name in Ollama (defaults to "llama3").
            chat_template_style: Only used for Ollama backend.
             One of ["openai", "chatml", "llama3"] (defaults to "llama3")
        """
        self.model_path = str(Path(model_path).resolve())
        self.backend = backend.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.model_name = model_name or "llama3"
        self.chat_template_style = chat_template_style or "llama3"

        if self.backend == "vllm":
            if not VLLM_AVAILABLE:
                raise LLMManagerError("vLLM backend selected but vllm is not installed.")

            if not Path(self.model_path).exists():
                raise LLMManagerError(f"Model path required by vllm, but does not exist: {self.model_path}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                raise LLMManagerError(f"Failed to load tokenizer: {str(e)}")

            try:
                self.llm = LLM(
                    model=self.model_path,
                    dtype=torch.float16,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=self._get_max_token_length(),
                    seed=self.seed,
                    trust_remote_code=True,
                    max_num_seqs=8,
                )
            except Exception as e:
                raise LLMManagerError(f"Failed to initialize vLLM: {str(e)}")

            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens
            )

        elif self.backend == "ollama":
            try:
                openai.api_key = "ollama"
                openai.api_base = api_base
            except Exception as e:
                raise LLMManagerError(f"Failed to configure OpenAI-compatible API: {str(e)}")

        else:
            raise LLMManagerError(f"Unsupported backend: {self.backend}")

    def _get_max_token_length(self) -> int:
        try:
            config = self.tokenizer.model_max_length
            return config if config != float('inf') else 8192
        except (AttributeError, KeyError):
            return 8192

    def format_chat(self, messages: List[dict]) -> str:
        """
        Formats a list of chat messages into a prompt string.

        Args:
            messages: List of dicts with "role" and "content".

        Returns:
            str: Formatted prompt string.
        """
        if self.backend == "vllm":
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as e:
                raise LLMManagerError(f"Failed to apply chat template: {e}")

        elif self.backend == "ollama":
            style = self.chat_template_style.lower()

            if style == "openai":
                # Simple ChatGPT style prompt.
                prompt = ""
                for m in messages:
                    if m["role"] == "system":
                        prompt += f"[System] {m['content']}\n"
                    elif m["role"] == "user":
                        prompt += f"[User] {m['content']}\n"
                    elif m["role"] == "assistant":
                        prompt += f"[Assistant] {m['content']}\n"
                prompt += "[Assistant] "
                return prompt

            elif style == "chatml":
                # ChatML format：<|im_start|>...
                prompt = ""
                for m in messages:
                    prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"
                return prompt

            elif style == "llama3":
                # Meta LLaMA 3 format.
                prompt = "<|begin_of_text|>"
                for m in messages:
                    role = m["role"]
                    content = m["content"]
                    if role == "system":
                        prompt += f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>\n"
                    elif role == "user":
                        prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>\n"
                    elif role == "assistant":
                        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>\n"
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
                return prompt

            else:
                raise LLMManagerError(f"Unsupported prompt style: {style}")
        else:
            raise LLMManagerError("Invalid backend for formatting chat.")

    def generate(self, prompts: Union[str, List[str]]) -> List[str]:
        """
        Generate text from one or more prompts.

        Args:
            prompts: Single prompt string or list of prompt strings.

        Returns:
            List[str]: Generated responses.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        responses = []

        if self.backend == "vllm":
            results = self.llm.generate(prompts, self.sampling_params)
            responses = [output.text for result in results for output in result.outputs]

        elif self.backend == "ollama":
            for prompt in prompts:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=0.9,
                    )
                    responses.append(response.choices[0].message["content"])
                except Exception as e:
                    responses.append(f"[ERROR] Failed to generate: {e}")

        else:
            raise LLMManagerError("Invalid backend specified during initialization.")

        return responses
