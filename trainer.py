import pdb

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import AdditionDataset
from eval_metrics import compute_generation_metrics
from hparams import (
    COSINE_ETA_MIN,
    COSINE_TMAX,
    EXP_NAME,
    LINEAR_START_FACTOR,
    LINEAR_TOTAL_ITERS,
    LR,
    MAX_INT,
    MODEL_DIM,
    MODEL_LOAD_PATH,
    MODEL_NHEADS,
    MODEL_NLAYERS,
    NUM_DATA,
    RL_NUM_SAMPLES,
    SAVE_EVERY_NSTEPS,
    SCHEDULER_MILESTONES,
    TEST_SPLIT,
    TRAIN_BATCH_SIZE,
    TRAIN_SPLIT,
    TRAINING_OBJECTIVE,
    VAL_BATCH_SIZE,
    VAL_EVERY_NSTEPS,
    VAL_SPLIT,
)
from logger import logger
from rl_utils import compute_rewards, compute_rl_objective
from transformer_implementation import Transformer
from writer import writer

"""
Gotchas:

Optimization:
  - much higher LR needed when training from scratch
  - higher batch size! (worked wonders)
  - was on the right track with schedule
  - gradient clipping (optional, sometimes works better without)

Dataset size:
  - Had overestimated the learnability of transformers and their sample efficiency
"""

torch.manual_seed(0)


def generate_dataset():
    full_dataset = AdditionDataset(num_data=NUM_DATA, max_int=MAX_INT)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT], generator=generator
    )

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset)

    return full_dataset, train_dataloader, val_dataloader, test_dataloader


full_dataset, train_dataloader, val_dataloader, test_dataloader = generate_dataset()
pad_idx = full_dataset.pad_idx
eos_idx = full_dataset.eos_idx
max_length = full_dataset.max_length

if torch.backends.mps.is_available:
    logger.info("Using MPS")
    device = torch.device("mps")
else:
    logger.info("Using CPU")
    device = torch.device("cpu")


model = Transformer(
    vocab_size=full_dataset.vocab_size(),
    max_length=max_length,
    eos_idx=eos_idx,
    pad_idx=pad_idx,
    dim=MODEL_DIM,
    nheads=MODEL_NHEADS,
    nlayers=MODEL_NLAYERS,
)
model = model.to(device)
if MODEL_LOAD_PATH:
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=True))
    logger.info(f"Loading model from {MODEL_LOAD_PATH}")
criterion = CrossEntropyLoss(ignore_index=pad_idx)
optimizer = Adam(model.parameters(), lr=LR)

scheduler1 = LinearLR(optimizer, start_factor=LINEAR_START_FACTOR, total_iters=LINEAR_TOTAL_ITERS)
scheduler2 = CosineAnnealingLR(optimizer, T_max=COSINE_TMAX, eta_min=COSINE_ETA_MIN)
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=SCHEDULER_MILESTONES)


def validate_generate(seq, src_masked):
    input_src_lengths = torch.count_nonzero(src_masked != model.pad_idx, dim=-1)
    output = model.generate(src_masked)
    pred_tensor = output["preds"]

    seq = seq.tolist()
    pred_list = pred_tensor.tolist()
    prompts, golds, preds = [], [], []
    for i in range(len(pred_list)):
        prompt = full_dataset.tensor_to_text(seq[i])
        gold = full_dataset.tensor_to_text(seq[i][input_src_lengths[i] :])
        pred = full_dataset.tensor_to_text(pred_list[i][input_src_lengths[i] :])
        prompts.append(prompt)
        golds.append(gold)
        preds.append(pred)
    return prompts, golds, preds


def validate(step):
    avg_loss = 0
    batches = 0
    prompts, golds, preds = [], [], []
    with torch.no_grad():
        for seq, masked_tgt, src_masked in tqdm(val_dataloader):
            seq = seq.to(device)
            src_masked = src_masked.to(device)
            logits = model(seq)
            logits = logits.reshape([-1, logits.shape[-1]])
            masked_tgt = masked_tgt.reshape([-1])
            masked_tgt = masked_tgt.to(device)
            loss = criterion(logits, masked_tgt)
            avg_loss += loss

            _prompts, _golds, _preds = validate_generate(seq, src_masked)
            prompts += _prompts
            golds += _golds
            preds += _preds

            batches += 1
    compute_generation_metrics(prompts, golds, preds, step)
    writer.add_scalar("loss/val", avg_loss / batches, step)
    logger.info(f"VALIDATE = {avg_loss / batches}")


step = 0
num_parameters = sum(p.numel() for p in model.parameters())
logger.info("Number of model parameters = ")
model.train()
for epoch in range(20):
    logger.info(f"Epoch: {epoch}")
    for seq, masked_tgt, masked_src in train_dataloader:
        optimizer.zero_grad()

        seq = seq.to(device)
        masked_tgt = masked_tgt.to(device)
        masked_src = masked_src.to(device)

        if TRAINING_OBJECTIVE == "sft":
            # seq: [batch_size (B), num_tokens (N)]
            # logits: [B, N, vocabulary size (C)]
            logits = model(seq)
            logits = logits.reshape([-1, logits.shape[-1]])
            masked_tgt = masked_tgt.reshape([-1])
            loss = criterion(logits, masked_tgt)
        else:
            B, N = masked_src.shape
            masked_src = masked_src.repeat_interleave(RL_NUM_SAMPLES, dim=0)
            with torch.no_grad():
                model.eval()
                outputs = model.generate(masked_src, require_probs=True)
                model.train()
            preds = outputs["preds"]
            pred_probs = outputs["pred_probs"]
            masked_pred_probs = torch.where(masked_src == pad_idx, pred_probs, 0.0)

            # compute probabilities for the predictions
            logits = model(preds)
            BxSamples, N, V = logits.shape
            fp_probs = log_softmax(logits, dim=-1)
            aligned_fp_probs = torch.cat(
                [torch.zeros((BxSamples, 1, V), device=fp_probs.device), fp_probs[:, 0 : N - 1, :]], dim=1
            )
            unsqueezed_preds = preds.unsqueeze(-1)
            aligned_fp_probs = torch.gather(input=aligned_fp_probs, dim=-1, index=unsqueezed_preds).squeeze(-1)
            # zero out log probabilities for padding
            valid_preds = torch.logical_and(masked_src == pad_idx, preds != pad_idx)
            pred_fp_probs = torch.where(valid_preds, aligned_fp_probs, 0.0)

            # assert forward pass and prediction probabilities are the same
            if not torch.allclose(masked_pred_probs, pred_fp_probs, atol=1e-2):
                pdb.set_trace()

            seq = seq.repeat_interleave(RL_NUM_SAMPLES, dim=0)
            masked_tgt = masked_tgt.repeat_interleave(RL_NUM_SAMPLES, dim=0)

            rewards = compute_rewards(seq, masked_tgt, masked_src, preds, full_dataset)
            pred_fp_probs = pred_fp_probs.unflatten(0, (B, RL_NUM_SAMPLES))
            rewards = rewards.unflatten(0, (B, RL_NUM_SAMPLES))
            loss = compute_rl_objective(pred_fp_probs, rewards)
        writer.add_scalar("loss/train", loss, step)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        writer.add_scalar("grad_norm", total_norm, global_step=step)

        optimizer.step()
        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        logger.info(f"loss: {loss}")
        if step % VAL_EVERY_NSTEPS == 0:
            logger.info(f"Epoch: {epoch}, Step: {step}")
            model.eval()
            validate(step)
            model.train()

        if step % SAVE_EVERY_NSTEPS == 0:
            logger.info(f"Saving model to checkpoints/{EXP_NAME}_{step}")
            torch.save(model.state_dict(), f"checkpoints/{EXP_NAME}_{step}")
        step += 1


writer.flush()
