from pathlib import Path
from typing import Union, List
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class LLMManagerError(Exception):
    """Base exception for LLMManager errors."""
    pass


class LLMManager:
    def __init__(self,
                 model_path: Union[str, Path],
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.8,
                 temperature: float = 1.0,
                 max_tokens: int = 2048,
                 seed: int = 42):
        """Manages the LLM model for generating text using vLLM.

        Args:
            model_path: Path to the local model directory.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0).
            temperature: Sampling temperature (higher = more random).
            max_tokens: Maximum number of tokens to generate.
            seed: Random seed for reproducibility.

        Raises:
            LLMManagerError: If model path doesn't exist or model loading fails.
        """
        self.model_path = str(Path(model_path).resolve())
        if not Path(self.model_path).exists():
            raise LLMManagerError(f"Model path does not exist: {self.model_path}")

        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize tokenizer first to access model config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            raise LLMManagerError(f"Failed to load tokenizer: {str(e)}")

        try:
            self.llm = self._initialize_llm(tensor_parallel_size, gpu_memory_utilization)
        except Exception as e:
            raise LLMManagerError(f"Failed to initialize LLM: {str(e)}")

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )

    def _initialize_llm(self, tensor_parallel_size: int, gpu_memory_utilization: float) -> LLM:
        """Initialize the vLLM model.

        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.

        Returns:
            LLM: Initialized vLLM model instance.
        """
        return LLM(
            model=self.model_path,
            dtype=torch.float16,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self._get_max_token_length(),
            seed=self.seed,
            trust_remote_code=True,
            max_num_seqs=8,
        )

    def _get_max_token_length(self) -> int:
        """Get maximum token length from model configuration.

        Returns:
            int: Maximum token length supported by the model.
        """
        try:
            config = self.tokenizer.model_max_length
            return config if config != float('inf') else 8192
        except (AttributeError, KeyError):
            # Fall back to default Llama 3 context window
            return 8192

    def generate(self, prompts: Union[str, List[str]]) -> List[str]:
        """Generate text from one or more prompts.

        Args:
            prompts: Single prompt string or list of prompt strings.

        Returns:
            List of generated text responses.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        results = self.llm.generate(prompts, self.sampling_params)
        return [output.text for result in results for output in result.outputs]