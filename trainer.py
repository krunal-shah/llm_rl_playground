from loguru import logger
import sys
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import editdistance
from torch.utils.data import DataLoader, random_split
from datasets import AdditionDataset
from transformer_implementation import Transformer
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

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

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)


def generate_dataset():
    full_dataset = AdditionDataset()
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [0.9, 0.05, 0.05], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset)

    return full_dataset, train_dataloader, val_dataloader, test_dataloader


full_dataset, train_dataloader, val_dataloader, test_dataloader = generate_dataset()
max_length = full_dataset.max_length

if torch.backends.mps.is_available:
    logger.info("Using MPS")
    device = torch.device('mps')
else:
    logger.info("Using CPU")
    device = torch.device('cpu')


writer = SummaryWriter(log_dir=f"runs/active/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
model = Transformer(vocab_size=full_dataset.vocab_size(), max_length=max_length, eos_idx=full_dataset.eos_idx)
model = model.to(device)
criterion = CrossEntropyLoss(ignore_index=full_dataset.pad_idx)
optimizer = Adam(model.parameters(), lr=5e-4)

scheduler1 = LinearLR(optimizer, start_factor=0.05, total_iters=30)
scheduler2 = CosineAnnealingLR(optimizer, T_max=800, eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[30])

# logger.level("INFO")


def compute_generation_metrics(prompts, golds, preds):
    num_samples = len(prompts)
    accuracy = 0
    edit_distance = 0
    for prompt, gold, pred in zip(prompts, golds, preds):
        if gold == pred:
            accuracy += 1
        edit_distance += editdistance.eval(gold, pred)
        logger.debug(f"{prompt=} {gold=} {pred=}")
    logger.info(f"accuracy = {accuracy/num_samples}, edit_distance = {edit_distance/num_samples}, {num_samples=}")
    writer.add_scalar("accuracy", accuracy/num_samples, step)
    writer.add_scalar("edit_distance", edit_distance/num_samples, step)
    writer.add_scalar("num_samples", num_samples, step)


def validate_generate(seq, src_masked):
    input_src_lengths = torch.count_nonzero(src_masked != model.pad_idx, dim=-1)
    pred_tensor = model.generate(src_masked)

    seq = seq.tolist()
    pred_list = pred_tensor.tolist()
    prompts, golds, preds = [], [], []
    for i in range(len(pred_list)):
        prompt = full_dataset.tensor_to_text(seq[i])
        gold = full_dataset.tensor_to_text(seq[i][input_src_lengths[i]:])
        pred = full_dataset.tensor_to_text(pred_list[i][input_src_lengths[i]:])
        prompts.append(prompt)
        golds.append(gold)
        preds.append(pred)
    return prompts, golds, preds


def validate(step):
    avg_loss = 0
    batches = 0
    prompts, golds, preds = [], [], []
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
    compute_generation_metrics(prompts, golds, preds)
    writer.add_scalar("loss/val", avg_loss/batches, step)
    logger.info(f"VALIDATE = {avg_loss/batches}")


# torch.set_printoptions(threshold=10_000)
step = 0
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

        # logger.info(loss)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        logger.info(f"loss: {loss}")
        step += 1
        if step % 20 == 0:
            logger.info(f"Epoch: {epoch}, Step: {step}")
            model.eval()
            validate(step)
            model.train()

writer.flush()
