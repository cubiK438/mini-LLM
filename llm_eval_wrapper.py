
import torch
import torch.nn.functional as F

from lm_eval.api.model import LM
from model.transformer_block import Transformer
from model.attention import causal_mask
from tokenizer.tokenizer import tokenize, decode


class MiniLLM(LM):
    def __init__(self, checkpoint_path="mini-llm-latest.pt", device=None):
        super().__init__()

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self._device = torch.device(device)
        self.max_length = 1024
        self.max_gen_toks = 128
        self.batch_size = 1

        self.model = Transformer(
            vocab_size=35000,
            d_model=768,
            num_layers=5,
            heads=12,
            intermediate_dim=3072,
        ).to(self._device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.checkpoint_step = checkpoint["step"]

        # This tokenizer has no dedicated BOS/EOS token configured.
        # Use a space token as a fallback prefix when context is empty.
        prefix_ids = tokenize(" ")
        if not prefix_ids:
            raise ValueError("The tokenizer cannot encode a prefix token.")
        self.prefix_token_id = prefix_ids[0]
        self.eot_token_id = self.prefix_token_id

        print(
            f"Loaded checkpoint step {self.checkpoint_step} "
            f"on {self._device}"
        )

    @property
    def tokenizer_name(self):
        return "mini-llm-bytelevel-bpe-v1"

    def tok_encode(self, text, **kwargs):
        return tokenize(text)

    def tok_decode(self, tokens, **kwargs):
        return decode(list(tokens))

    @torch.inference_mode()
    def _forward(self, token_ids):
        """Return logits for a 1D list of token IDs."""
        if not token_ids:
            raise ValueError("Cannot run the model on an empty sequence.")

        token_ids = token_ids[-self.max_length:]
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self._device
        )
        mask = causal_mask(input_ids.shape[1]).to(self._device)
        return self.model(input_ids, mask)[0].float()

    def _score_tokens(self, context_ids, continuation_ids):
        """Score continuation tokens, preserving as much context as possible."""
        if not continuation_ids:
            return 0.0, True

        context_ids = list(context_ids)
        continuation_ids = list(continuation_ids)

        # The model needs a preceding position to predict the first token.
        if not context_ids:
            context_ids = [self.prefix_token_id]

        full_ids = context_ids + continuation_ids
        context_len = len(context_ids)
        total_len = len(full_ids)

        total_logprob = 0.0
        is_greedy = True

        # Score chunks to keep every forward pass within the context limit.
        # Each chunk scores at most 512 target tokens.
        chunk_size = min(512, self.max_length - 1)

        for target_start in range(context_len, total_len, chunk_size):
            target_end = min(target_start + chunk_size, total_len)

            # Include the preceding token plus as much earlier context as fits.
            left = max(0, target_start - (self.max_length - (target_end - target_start)))
            input_ids = full_ids[left:target_end]

            logits = self._forward(input_ids)

            # Logits at position k predict token k+1.
            local_start = target_start - left - 1
            local_end = target_end - left - 1

            prediction_logits = logits[local_start:local_end]
            targets = torch.tensor(
                full_ids[target_start:target_end],
                dtype=torch.long,
                device=self._device,
            )

            log_probs = F.log_softmax(prediction_logits, dim=-1)
            selected_log_probs = log_probs.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)

            total_logprob += selected_log_probs.sum().item()
            is_greedy = is_greedy and (
                prediction_logits.argmax(dim=-1) == targets
            ).all().item()

        return total_logprob, bool(is_greedy)

    def loglikelihood(self, requests):
        results = []

        for request in requests:
            context, continuation = request.args

            # Tokenize the combined text to avoid blindly assuming that BPE
            # tokenization is additive across the context/continuation boundary.
            context_ids = tokenize(context)
            combined_ids = tokenize(context + continuation)

            # Find the unchanged token prefix. Tokens merged across the text
            # boundary are assigned to the continuation side.
            common = 0
            while (
                common < len(context_ids)
                and common < len(combined_ids)
                and context_ids[common] == combined_ids[common]
            ):
                common += 1

            effective_context = combined_ids[:common]
            continuation_ids = combined_ids[common:]

            score = self._score_tokens(effective_context, continuation_ids)
            results.append(score)

            self.cache_hook.add_partial(
                "loglikelihood", (context, continuation), score
            )

        return results

    def loglikelihood_rolling(self, requests):
        results = []

        for request in requests:
            (text,) = request.args
            token_ids = tokenize(text)

            # Score the entire document in chunks with maximal available
            # left context. The first token uses the fallback prefix.
            score, _ = self._score_tokens([], token_ids)
            results.append(score)

            self.cache_hook.add_partial(
                "loglikelihood_rolling", (text,), score
            )

        return results

    @torch.inference_mode()
    def generate_until(self, requests):
        results = []

        for request in requests:
            context, gen_kwargs = request.args
            gen_kwargs = dict(gen_kwargs)

            until = gen_kwargs.pop("until", [])
            if isinstance(until, str):
                until = [until]

            max_new_tokens = int(
                gen_kwargs.pop("max_gen_toks", self.max_gen_toks)
            )
            temperature = float(gen_kwargs.pop("temperature", 0.0))
            top_k = gen_kwargs.pop("top_k", None)
            top_p = float(gen_kwargs.pop("top_p", 1.0))
            do_sample = bool(gen_kwargs.pop("do_sample", temperature > 0))

            # Remaining generation options are intentionally not silently
            # forwarded to this custom Transformer.
            unsupported = set(gen_kwargs) - {"until"}
            if unsupported:
                raise ValueError(
                    f"Unsupported generation arguments: {sorted(unsupported)}"
                )

            prompt_ids = tokenize(context)
            if not prompt_ids:
                prompt_ids = [self.prefix_token_id]

            generated_ids = []
            output_text = ""

            for _ in range(max_new_tokens):
                logits = self._forward(prompt_ids + generated_ids)[-1].clone()

                if do_sample and temperature > 0:
                    logits = logits / temperature

                    if top_k is not None:
                        k = max(1, min(int(top_k), logits.numel()))
                        threshold = torch.topk(logits, k).values[-1]
                        logits[logits < threshold] = -float("inf")

                    if 0 < top_p < 1:
                        sorted_logits, sorted_indices = torch.sort(
                            logits, descending=True
                        )
                        cumulative_probs = torch.softmax(
                            sorted_logits, dim=-1
                        ).cumsum(dim=-1)
                        remove = cumulative_probs > top_p
                        remove[1:] = remove[:-1].clone()
                        remove[0] = False
                        logits[sorted_indices[remove]] = -float("inf")

                    next_token = torch.multinomial(
                        torch.softmax(logits, dim=-1), 1
                    ).item()
                else:
                    next_token = logits.argmax().item()

                generated_ids.append(next_token)
                output_text = decode(generated_ids)

                if any(stop and stop in output_text for stop in until):
                    positions = [
                        output_text.find(stop)
                        for stop in until
                        if stop and stop in output_text
                    ]
                    output_text = output_text[:min(positions)]
                    break

            results.append(output_text)

            self.cache_hook.add_partial(
                "generate_until", (context, request.args[1]), output_text
            )

        return results
