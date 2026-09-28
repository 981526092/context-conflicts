import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class QwenHFBackend:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-32B",
        trust_remote_code: bool = False,
        dist_backend: str = "nccl",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.dist_backend = dist_backend
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._dist = None
        self._closed = False

        import os

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _ensure_dist(self) -> None:
        """Initialize the distributed process group (no model loading)."""
        if self._dist is not None:
            return

        import torch
        import torch.distributed as dist

        self._torch = torch
        self._dist = dist

        if self.distributed:
            if self.dist_backend == "nccl" and not torch.cuda.is_available():
                raise RuntimeError(
                    "NCCL backend requires CUDA but torch.cuda.is_available() is False. "
                    "Use --backend gloo for CPU-only nodes."
                )
            if not dist.is_initialized():
                import datetime, os
                # Nodes on a slow link can exceed the 10-minute default collective
                # timeout; DIST_TIMEOUT_MIN overrides the 60-minute default used here.
                timeout = datetime.timedelta(minutes=int(os.environ.get("DIST_TIMEOUT_MIN", "60")))
                dist.init_process_group(backend=self.dist_backend, timeout=timeout)
                logger.info(
                    "Distributed init: backend=%s rank=%d/%d local_rank=%d",
                    self.dist_backend,
                    self.rank,
                    self.world_size,
                    self.local_rank,
                )
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)

    def _ensure_loaded(self) -> None:
        """Initialize dist (if needed) and load model + tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        load_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit or self.load_in_8bit:
            from transformers import BitsAndBytesConfig
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = {"": self.local_rank}
            logger.info("Loading model with NF4 quantization (bitsandbytes)")
        elif self.load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            # Force single-device map so bitsandbytes can quantize all layers
            # on GPU (device_map="auto" may offload to CPU, which bnb rejects).
            load_kwargs["device_map"] = {"": self.local_rank}
            logger.info("Loading model with INT8 quantization (bitsandbytes)")
        elif self.distributed:
            # Let transformers handle dist init + device_mesh creation.
            # Qwen3's config.base_model_tp_plan already has a complete
            # plan for attention + MLP layers; "auto" picks it up.
            load_kwargs["tp_plan"] = "auto"
        elif torch.cuda.is_available():
            # Put the whole model on this GPU. On unified-memory GPUs
            # device_map="auto" can misjudge free memory and silently offload
            # most of the model to CPU, which runs about a hundred times slower.
            # A model that does not fit should fail loudly here instead.
            load_kwargs["device_map"] = {"": self.local_rank}
        else:
            load_kwargs["device_map"] = "auto"
            load_kwargs["offload_folder"] = "/tmp/hf_offload"

        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        except ValueError:
            # Some models (e.g. Mistral 3 / Ministral 3) use a conditional-
            # generation architecture.  Fall back to AutoModelForImageTextToText
            # which covers vision-language models used in text-only mode.
            from transformers import AutoModelForImageTextToText
            logger.info("AutoModelForCausalLM unsupported; falling back to AutoModelForImageTextToText")
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)

        # After from_pretrained with tp_plan, dist is initialized by
        # transformers — grab the reference for later use.
        if self.distributed:
            import torch.distributed as dist
            self._dist = dist
            if not dist.is_initialized():
                # Fallback: init ourselves if transformers didn't
                self._ensure_dist()

        # --- TP verification (check hooked modules, not DTensors) ---
        if self.distributed:
            hooked = 0
            total_modules = 0
            for name, mod in self._model.named_modules():
                total_modules += 1
                if getattr(mod, "_is_hooked", False):
                    hooked += 1
            if hooked > 0:
                logger.info(
                    "TP active: %d/%d modules have TP hooks (world_size=%d)",
                    hooked,
                    total_modules,
                    self.world_size,
                )
            else:
                logger.warning(
                    "TP may NOT be active: 0/%d modules have TP hooks.",
                    total_modules,
                )

            if torch.cuda.is_available():
                mem_mb = torch.cuda.memory_allocated(self.local_rank) / (1024 * 1024)
                logger.info("Rank %d GPU memory after load: %.0f MB", self.rank, mem_mb)

    def initialize(self) -> None:
        """Explicitly load the model and initialize distributed.

        Call this once at startup before any agent logic runs.
        """
        self._ensure_loaded()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _format_prompt(self, messages: List[Dict[str, str]], enable_thinking: bool) -> str:
        tokenizer = self._tokenizer
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        prompt = self._format_prompt(messages, enable_thinking=enable_thinking)
        model_inputs = self._tokenizer([prompt], return_tensors="pt")
        if self.distributed and self._torch.cuda.is_available():
            model_inputs = {k: v.to(self.local_rank) for k, v in model_inputs.items()}
        elif hasattr(self._model, "device"):
            model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}

        with self._torch.no_grad():
            outputs = self._model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=2048,
                min_new_tokens=0,
                repetition_penalty=1.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs.sequences[0][len(model_inputs["input_ids"][0]) :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Distributed primitives
    # ------------------------------------------------------------------

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast a picklable object from ``src`` to all ranks.

        Returns the broadcast object on every rank.  Falls through as a
        no-op (returning ``obj`` unchanged) when not in distributed mode.
        """
        if not self.distributed:
            return obj
        self._ensure_dist()
        assert self._dist is not None
        container = [obj]
        self._dist.broadcast_object_list(container, src=src)
        return container[0]

    def barrier(self) -> None:
        """Synchronize all ranks.  No-op outside distributed mode."""
        if not self.distributed:
            return
        self._ensure_dist()
        assert self._dist is not None
        self._dist.barrier()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.distributed and self._dist is not None and self._dist.is_initialized():
            self._dist.barrier()
            self._dist.destroy_process_group()
        self._model = None
        self._tokenizer = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
