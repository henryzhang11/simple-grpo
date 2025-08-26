import os
import re
import socket
import time
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset

from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

# Set environment variable for Ray to pass parameters between processes.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
# --- Configurations ---
MODEL_NAME = "/root/qwen2.5-0.5b-instruct-local"
VLLM_TENSOR_PARALLEL_SIZE = 1
TRAINING_WORLD_SIZE = 1
VLLM_NUM_ENGINES = 1
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 16
N_ROLLOUTS = 4
GRPO_EPOCHS = 4
MAX_PROMPT_LEN = 512
MAX_COMPLETION_LEN = 256
NUM_EPOCHS = 3
LR = 1e-6
CLIP_EPS = 0.1
SEED = 42
UPDATE_VLLM_EVERY_N_STEPS = 1
GRADIENT_CHECKPOINTING = False

# ======================================================================================
#  Utilities & Worker Extension (from vllm_worker_wrap.py)
# ======================================================================================

def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl

class WorkerWrap:
    def init_process_group(self, master_addr: str, master_port: str, rank_offset: int, world_size: int):
        rank = torch.distributed.get_rank() + rank_offset
        self.device = torch.cuda.current_device()
        self._model_update_group = stateless_init_process_group(master_addr, master_port, rank, world_size, self.device)

    def update_weight(self, name: str, dtype: torch.dtype, shape: torch.Size):
        weight_tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self._model_update_group.broadcast(weight_tensor, src=0, stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=[(name, weight_tensor)])
        del weight_tensor

# ======================================================================================
#  vLLM Engine Actor (from vllm_engine.py)
# ======================================================================================

@ray.remote
class VLLMActor:
    def __init__(self, model_name: str, tensor_parallel_size: int, seed: int):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            seed=seed,
            worker_extension_cls="dapo.WorkerWrap",
            dtype=torch.bfloat16, 
        )

    def generate(self, batch_prompts: List[str], sampling_params: SamplingParams):
        outs = self.llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        texts, prompt_ids, completion_ids, final_logps = [], [], [], []
        for o in outs:
            texts.append([seq.text for seq in o.outputs])
            prompt_ids.append([o.prompt_token_ids] * len(o.outputs))
            completion_ids.append([seq.token_ids for seq in o.outputs])
            sub_logps = [
                [logprob_dict[token_id].logprob for token_id, logprob_dict in zip(seq.token_ids, seq.logprobs)] if seq.logprobs else []
                for seq in o.outputs
            ]
            final_logps.append(sub_logps)
        return texts, prompt_ids, completion_ids, final_logps

    def init_process_group(self, master_addr: str, master_port: str, rank_offset: int, world_size: int):
        return self.llm.collective_rpc("init_process_group", args=(master_addr, master_port, rank_offset, world_size))

    def update_weight(self, name: str, dtype: torch.dtype, shape: torch.Size):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape))

def create_vllm_engines(num_engines: int, tensor_parallel_size: int, pretrain: str, seed: int):
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
    pg = placement_group(bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())
    
    vllm_engines = []
    for i in range(num_engines):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size
        )
        vllm_engines.append(
            VLLMActor.options(
                num_gpus=tensor_parallel_size,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model_name=pretrain,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
            )
        )
    return vllm_engines

# ======================================================================================
#  PPO Training Actor (from ppo_actor.py)
# ======================================================================================

@ray.remote
class PolicyModelActor:
    def __init__(self, world_size: int, rank: int, master_addr: str, master_port: int = None):
        if not master_port:
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        self._model_update_group = None

    def _setup_distributed(self, seed: int):
        torch.distributed.init_process_group(backend="nccl")
        set_seed(seed)

    def init_model_from_pretrained(self, deepspeed_config: Dict, hyperparameters: Dict, pretrain: str, max_steps: int, vllm_engines: List[VLLMActor]):
        import deepspeed
        self.args = hyperparameters
        self.vllm_engines = vllm_engines
        self._setup_distributed(seed=42)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain)
        model = AutoModelForCausalLM.from_pretrained(pretrain, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")
        self.model_config = model.config
        for _, param in model.named_parameters():
            param.requires_grad = True
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(model=model, config=deepspeed_config)
        if self.vllm_engines and torch.distributed.get_rank() == 0:
            master_address = "127.0.0.1"
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = 1 + self.args["vllm_tensor_parallel_size"]
            vllm_init_future = [engine.init_process_group.remote(master_address, master_port, 1, world_size) for engine in self.vllm_engines]
            self._model_update_group = stateless_init_process_group(master_address, master_port, 0, world_size, torch.cuda.current_device())
            ray.get(vllm_init_future)
        
    def train_on_batch(self, experience: Dict[str, torch.Tensor]) -> Dict[str, float]:
        device = torch.cuda.current_device()
        status_list = []        
        perm = torch.randperm(experience["ids"].size(0))
        for _ in range(self.args["grpo_epochs"]):
            for i in range(0, len(perm), self.args["micro_batch_size"]):

                batch_indices = perm[i : i + self.args["micro_batch_size"]]
                micro_batch = {key: val[batch_indices].to(device) for key, val in experience.items()}
                ids, msk, lbls = micro_batch["ids"], micro_batch["msk"], micro_batch["lbls"]
                logp_old, adv = micro_batch["logp_old"], micro_batch["adv"]
                
                out = self.model_engine(input_ids=ids, attention_mask=msk)
                logits = out.logits[:, :-1, :]  
                labels = lbls[:, 1:]            
                logp_old_shifted = logp_old[:, 1:]
                adv_shifted = adv.repeat(1, labels.shape[1]) 
                
                logp_new = F.log_softmax(logits.float(), dim=-1)
                gather_labels = labels.clone()
                gather_labels[labels == -100] = 0
                logp_new_token = torch.gather(logp_new, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
                
                loss_mask = (labels != -100).float()
                logp_new_token = logp_new_token * loss_mask
                logp_old_shifted = logp_old_shifted * loss_mask                
                log_ratio = logp_new_token - logp_old_shifted.detach()
                log_ratio_seq = (log_ratio * loss_mask).sum(dim=1)
                ratio_seq = torch.exp(log_ratio_seq)
                adv_seq = micro_batch["adv"].squeeze(-1).detach()
                surr1 = ratio_seq * adv_seq
                surr2 = torch.clamp(ratio_seq, 1.0 - self.args["eps_clip"], 1.0 + self.args["eps_clip"]) * adv_seq
                policy_loss = - torch.min(surr1, surr2)
                loss = policy_loss.mean()            
                self.model_engine.backward(loss)
                self.model_engine.step()
                micro_batch_index = i // self.args["micro_batch_size"]
                status = {"policy_loss": loss.item()}
                status_list.append(status)
        torch.cuda.synchronize()
        if not status_list: return {}
        status_mean = {k: sum(d[k] for d in status_list) / len(status_list) for k in status_list[0]}
        return status_mean
      
    def broadcast_to_vllm(self):
        import deepspeed
        if torch.distributed.get_rank() != 0:
            return
        model_to_sync = self.model_engine.module
        for name, param in model_to_sync.named_parameters():
            is_zero3 = hasattr(self.model_engine, 'zero_optimization_stage') and self.model_engine.zero_optimization_stage() == 3
            with deepspeed.zero.GatheredParameters([param], enabled=is_zero3):
                if torch.distributed.get_rank() == 0:
                    refs = [engine.update_weight.remote(name, dtype=param.dtype, shape=param.shape) for engine in self.vllm_engines]
                    self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())                    
                    ray.get(refs)

# ======================================================================================
#  Main Orchestration Logic
# ======================================================================================

def preprocess(examples, tokenizer):
    prompts = [q + " Let's think step by step until we get an answer and copy the answer (without '\boxed{}' or '$') after a '####' sign." for q in examples["question"]]
    return {"prompt": prompts, "answer": examples["answer"]}

def build_batch_and_compute_reward(tokenizer, prompts, answers, gens, prompt_ids, completion_ids, logps_old):
    flat_prompt_ids = [ids for sublist in prompt_ids for ids in sublist]
    flat_completion_ids = [ids for sublist in completion_ids for ids in sublist]
    flat_logps = [l for sublist in logps_old for l in sublist]

    input_ids = [p + c for p, c in zip(flat_prompt_ids, flat_completion_ids)]
    padded = tokenizer.pad({"input_ids": input_ids}, padding="longest", max_length=MAX_PROMPT_LEN + MAX_COMPLETION_LEN, return_tensors="pt")
    ids, msk = padded["input_ids"], padded["attention_mask"]

    lbls = ids.clone()
    lbls[lbls == tokenizer.pad_token_id] = -100
    for i, p_ids in enumerate(flat_prompt_ids):
        lbls[i, :len(p_ids)] = -100

    logp_old = torch.zeros_like(lbls, dtype=torch.float32)
    for i, logp_list in enumerate(flat_logps):
        start = len(flat_prompt_ids[i])
        end = start + len(logp_list)
        if logp_list and end <= logp_old.shape[1]:
            logp_old[i, start:end] = torch.tensor(logp_list)

    rewards_list = []
    printed_sample_this_batch = 0

    for sub_gens, answer in zip(gens, answers):
        ans_num = re.findall(r"#### (\-?[0-9\.\,]+)", answer)[-1].replace(",", "")
        sub_rewards = [1.0 if (match := re.findall(r"#### (\-?[0-9\.\,]+)", gen_text)) and match[-1].replace(",", "") == ans_num else 0.0 for gen_text in sub_gens]
        for i, gen_text in enumerate(sub_gens):
            gen_match = re.findall(r"#### (\-?[0-9\.\,]+)", gen_text)
            gen_ans = gen_match[-1].replace(",", "") if gen_match else "N/A"
            reward = 1.0 if gen_ans != "N/A" and gen_ans == ans_num else 0.0
            if printed_sample_this_batch < 1 and i < 1:
                print("="*70)
                print(f" REWARD DEBUGGER (Sample {i+1})")
                print(f"  - Generated Snippet    : {gen_text}")
                print(f"  - Regex Expected Number: '{ans_num}'")
                print(f"  - Regex Found Number   : '{gen_ans}'")
                print("="*70 + "\n")
                printed_sample_this_batch += 1
        rewards_list.append(sub_rewards)

    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    adv = (rewards - rewards.mean(dim=-1, keepdim=True)).flatten().unsqueeze(1)
    return ids, msk, lbls, logp_old, adv, rewards.mean().item()

def evaluate_on_gsm8k(vllm_engines, test_dataset, step):
    vllm_actor = vllm_engines[0]
    eval_sampling_params = SamplingParams(n=1, temperature=0, top_p=0.95, max_tokens=MAX_COMPLETION_LEN)
    
    correct_predictions, total_predictions = 0, 0
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=lambda b: b)

    for batch in test_loader:
        prompts = [item['prompt'] for item in batch]
        ground_truths = [item['answer'] for item in batch]
        
        generated_responses_list, _, _, _ = ray.get(vllm_actor.generate.remote(prompts, eval_sampling_params))
        generated_texts = [resp[0] for resp in generated_responses_list]
    
        if step == 10:
            print("\nSAVING GSM8K TEST COMPLETIONS TO FILE (STEP 10)\n")
            with open("validation_completions_log.txt", "w", encoding="utf-8") as f:
                f.write("="*50 + "\n")
                f.write(f"GSM8K Test Set Completions (from Training Step {step})\n")
                f.write("="*50 + "\n\n")
                for p, g, truth in zip(prompts, generated_texts, ground_truths):
                    groun_truth = re.findall(r"#### (\-?[0-9\.\,]+)", truth)
                    groun_truth = groun_truth[-1].replace(",", "")
                    f.write(f"--- PROMPT ---\n{p}\n\n")
                    f.write(f"--- CORRECT ANSWER ---\n{groun_truth}\n\n")
                    f.write(f"--- GENERATED COMPLETION ---\n{g}\n\n")
                    f.write("-" * 25 + "\n\n")

        for gen_text, truth_text in zip(generated_texts, ground_truths):
            total_predictions += 1
            true_ans, gen_ans = None, None
            if (true_ans_match := re.findall(r"#### (\-?[0-9\.\,]+)", truth_text)):
                true_ans = true_ans_match[-1].replace(",", "")
            if (gen_ans_match := re.findall(r"#### (\-?[0-9\.\,]+)", gen_text)):
                gen_ans = gen_ans_match[-1].replace(",", "")
            if true_ans is not None and gen_ans is not None and gen_ans == true_ans:
                correct_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    print(f"GSM8K Test Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    ray.init(runtime_env={"working_dir": "."})
    set_seed(SEED)
    
    vllm_engines = create_vllm_engines(VLLM_NUM_ENGINES, VLLM_TENSOR_PARALLEL_SIZE, MODEL_NAME, SEED)
    actor_model = PolicyModelActor.options(num_gpus=1).remote(TRAINING_WORLD_SIZE, 0, "127.0.0.1")
    
    train_batch_size = BATCH_SIZE * N_ROLLOUTS
    grad_accumulation_steps = train_batch_size // (MICRO_BATCH_SIZE * TRAINING_WORLD_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    train_ds = load_dataset("openai/gsm8k", "main", split="train").map(preprocess, batched=True, num_proc=4, fn_kwargs={"tokenizer": tokenizer})
    test_ds = load_dataset("openai/gsm8k", "main", split="test").map(preprocess, batched=True, num_proc=4, fn_kwargs={"tokenizer": tokenizer})
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: b, drop_last=True)
    train_ds_len = len(train_ds)
    max_steps = (train_ds_len // BATCH_SIZE) * NUM_EPOCHS

    deepspeed_config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 1,
        "optimizer": {"type": "AdamW", "params": {"lr": LR}},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 1, 
        },
        "gradient_accumulation_steps": grad_accumulation_steps,
        "gradient_clipping": 1.0
    }
    hyperparameters = {
        "lr": LR, "eps_clip": CLIP_EPS, "grpo_epochs": GRPO_EPOCHS,
        "micro_batch_size": MICRO_BATCH_SIZE, "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
    }
    
    ray.get(actor_model.init_model_from_pretrained.remote(
        deepspeed_config=deepspeed_config, hyperparameters=hyperparameters,
        pretrain=MODEL_NAME, max_steps=max_steps, vllm_engines=vllm_engines))

    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} Step {step} ---\n")
            if step % 5 == 0:
                evaluate_on_gsm8k(vllm_engines, test_ds, step)
            step_start_time = time.perf_counter()
            prompts = [e["prompt"] for e in batch]
            answers = [e["answer"] for e in batch]
            sampling_params = SamplingParams(n=N_ROLLOUTS, repetition_penalty=1, temperature=1.0, top_p=0.95, max_tokens=MAX_COMPLETION_LEN, logprobs=True)
            gens, prompt_ids, completion_ids, logps_old = ray.get(vllm_engines[0].generate.remote(prompts, sampling_params))
            ids, msk, lbls, logp_old, adv, mean_reward = build_batch_and_compute_reward(tokenizer, prompts, answers, gens, prompt_ids, completion_ids, logps_old)
            if step == 10:
                print("\nSAVING TRAINING BATCH COMPLETIONS TO FILE (STEP 10)\n")
                with open("rollout_completions_log.txt", "w", encoding="utf-8") as f:
                    f.write("="*50 + "\n")
                    f.write(f"Training Batch Completions (Step {step})\n")
                    f.write("="*50 + "\n\n")
                    for i, (prompt, answer, gen_list) in enumerate(zip(prompts, answers, gens)):
                        groun_truth = re.findall(r"#### (\-?[0-9\.\,]+)", answer)
                        groun_truth = groun_truth[-1].replace(",", "")
                        f.write(f"--- PROMPT ---\n{prompt}\n\n")
                        f.write(f"--- CORRECT ANSWER ---\n{groun_truth}\n\n")
                        for j, completion in enumerate(gen_list):
                            f.write(f"--- GENERATED COMPLETION {j+1}/{N_ROLLOUTS} ---\n{completion}\n\n")
                        f.write("-" * 25 + "\n\n")
            status = ray.get(actor_model.train_on_batch.remote({"ids": ids, "msk": msk, "lbls": lbls, "logp_old": logp_old, "adv": adv}))
            if (step + 1) % UPDATE_VLLM_EVERY_N_STEPS == 0:
                ray.get(actor_model.broadcast_to_vllm.remote()) 
            step_elapsed = time.perf_counter() - step_start_time
            print(f"[E{epoch} S{step}] Mean reward: {mean_reward:.4f} | GRPO Loss: {status.get('policy_loss'):.4f} | Time: {step_elapsed:.4f}s")
            step += 1
    ray.shutdown()

if __name__ == "__main__":
    main()
