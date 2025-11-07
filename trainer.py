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

full_dataset = AdditionDataset()
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [0.95, 0.025, 0.025], generator=generator)
max_length = full_dataset.max_length

train_dataloader = DataLoader(train_dataset, batch_size=64)
val_dataloader = DataLoader(val_dataset)
test_dataloader = DataLoader(test_dataset)

if torch.backends.mps.is_available:
    logger.info("Using MPS")
    device = torch.device('mps')
else:
    logger.info("Using CPU")
    device = torch.device('cpu')

writer = SummaryWriter()
model = Transformer(vocab_size=full_dataset.vocab_size(), max_length=max_length)
model = model.to(device)
criterion = CrossEntropyLoss(ignore_index=full_dataset.pad_idx)
optimizer = Adam(model.parameters(), lr=5e-4)

scheduler1 = LinearLR(optimizer, start_factor=0.05, total_iters=30)
scheduler2 = CosineAnnealingLR(optimizer, T_max=1000, eta_min=3e-4)
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[30])

# logger.level("INFO")


def validate_generate(step):
    num_correct = 0
    total_edit_distance = 0
    total_diff = 0
    total = 0
    for seq, masked_tgt, src_masked, src_lengths in val_dataloader:
        src_masked = src_masked.to(device)
        initial_src_masked_text = full_dataset.tensor_to_text(src_masked[0])
        src_lengths = src_lengths.to(device)

        max_length = src_masked.shape[-1]
        eos_predictions = torch.tensor([False], device=device)
        lengths_maxed = torch.tensor([False], device=device)
        while (not eos_predictions._is_all_true()) and (not lengths_maxed._is_all_true()):
            # logger.info(f"{src_masked=}")
            logits = model(src_masked)
            # logger.info(f"{logits=}")
            logits = logits[:, src_lengths - 1, :].squeeze(1)

            predictions = torch.argmax(logits, dim=-1, keepdim=False)
            # logger.info(f"{logits=} {predictions=}")

            src_masked[:, src_lengths] = predictions
            eos_predictions = (predictions == full_dataset.eos_idx)
            lengths_maxed = (src_lengths == (max_length - 1))
            src_lengths = torch.where((~eos_predictions) & (~lengths_maxed), src_lengths + 1, src_lengths)
            # logger.info(f"{src_lengths=} {lengths_maxed=} {src_masked=} {eos_predictions=}")
        gold = full_dataset.tensor_to_text(seq[0])[len(initial_src_masked_text):]
        pred = full_dataset.tensor_to_text(src_masked[0])[len(initial_src_masked_text):]
        if gold == pred:
            num_correct += 1
        else:
            logger.info(f"{initial_src_masked_text=} {pred=} {gold=} {seq=} {src_masked=}")
        total_edit_distance += editdistance.eval(gold, pred)
        total_diff += abs(int(gold) - (int(pred) if pred else 0))
        total += 1
    writer.add_scalar("accuracy/val", num_correct/total, step)
    logger.info(f"accuracy={num_correct/total} avg_edit_distance={total_edit_distance/total} {num_correct=} {total_edit_distance=} {total_diff=}")


def validate(step):
    avg_loss = 0
    batches = 0
    for seq, masked_tgt, _, _ in val_dataloader:
        seq = seq.to(device)
        logits = model(seq)
        logits = logits.reshape([-1, logits.shape[-1]])
        masked_tgt = masked_tgt.reshape([-1])
        masked_tgt = masked_tgt.to(device)
        loss = criterion(logits, masked_tgt)
        avg_loss += loss
        batches += 1
    writer.add_scalar("loss/val", avg_loss/batches, step)
    logger.info(f"VALIDATE = {avg_loss/batches}")


# torch.set_printoptions(threshold=10_000)
step = 0
model.train()
for epoch in range(20):
    logger.info(f"Epoch: {epoch}")
    for seq, masked_tgt, _, _ in train_dataloader:
        optimizer.zero_grad()

        logger.debug(seq)
        seq = seq.to(device)
        masked_tgt = masked_tgt.to(device)
        
        logger.debug(seq)
        logger.debug(masked_tgt)
        
        # seq: [batch_size (B), num_tokens (N)]
        # logits: [B, N, vocabulary size (C)]
        logits = model(seq)
        
        logger.debug(logits.shape)
        
        logits = logits.reshape([-1, logits.shape[-1]])
        
        logger.debug(logits.shape)
        
        masked_tgt = masked_tgt.reshape([-1])
        
        logger.debug(masked_tgt.shape)
        logger.debug(f"{seq=} {seq.shape=}")
        logger.debug(f"{logits=} {logits.shape=}")
        logger.debug(f"{masked_tgt=} {masked_tgt.shape=}")
        logger.debug(f"{torch.argmax(logits, dim=-1)}")
        logger.debug(f"{seq.reshape([-1])}")
        
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
            validate_generate(step)
            model.train()

writer.flush()