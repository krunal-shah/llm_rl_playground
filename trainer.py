import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import AdditionDataset
from eval_metrics import compute_generation_metrics
from hparams import (
    COSINE_ETA_MIN,
    COSINE_TMAX,
    LINEAR_START_FACTOR,
    LINEAR_TOTAL_ITERS,
    LR,
    MAX_INT,
    MODEL_DIM,
    MODEL_NHEADS,
    MODEL_NLAYERS,
    NUM_DATA,
    SCHEDULER_MILESTONES,
    TEST_SPLIT,
    TRAIN_BATCH_SIZE,
    TRAIN_SPLIT,
    VAL_BATCH_SIZE,
    VAL_SPLIT,
)
from logger import logger
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
    eos_idx=full_dataset.eos_idx,
    dim=MODEL_DIM,
    nheads=MODEL_NHEADS,
    nlayers=MODEL_NLAYERS,
)
model = model.to(device)
criterion = CrossEntropyLoss(ignore_index=full_dataset.pad_idx)
optimizer = Adam(model.parameters(), lr=LR)

scheduler1 = LinearLR(optimizer, start_factor=LINEAR_START_FACTOR, total_iters=LINEAR_TOTAL_ITERS)
scheduler2 = CosineAnnealingLR(optimizer, T_max=COSINE_TMAX, eta_min=COSINE_ETA_MIN)
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=SCHEDULER_MILESTONES)


def validate_generate(seq, src_masked):
    input_src_lengths = torch.count_nonzero(src_masked != model.pad_idx, dim=-1)
    pred_tensor, pred_probs = model.generate(src_masked)

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
    for seq, masked_tgt, _ in train_dataloader:
        optimizer.zero_grad()

        seq = seq.to(device)
        masked_tgt = masked_tgt.to(device)

        # seq: [batch_size (B), num_tokens (N)]
        # logits: [B, N, vocabulary size (C)]
        logits = model(seq)

        logits = logits.reshape([-1, logits.shape[-1]])

        masked_tgt = masked_tgt.reshape([-1])

        loss = criterion(logits, masked_tgt)
        writer.add_scalar("loss/train", loss, step)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        writer.add_scalar("grad_norm", total_norm, global_step=step)

        optimizer.step()
        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        logger.info(f"loss: {loss}")
        step += 1
        if step % 50 == 0:
            logger.info(f"Epoch: {epoch}, Step: {step}")
            model.eval()
            validate(step)
            model.train()

writer.flush()
