from loguru import logger
import sys
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from datasets import AdditionDataset
from transformer_implementation import Transformer
from torch.optim import Adam

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
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [0.8, 0.1, 0.1], generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=2)
val_dataloader = DataLoader(val_dataset)
test_dataloader = DataLoader(test_dataset)

if torch.backends.mps.is_available:
    logger.info("Using MPS")
    device = torch.device('mps')
else:
    logger.info("Using CPU")
    device = torch.device('cpu')

model = Transformer(vocab_size=full_dataset.vocab_size())
model = model.to(device)
criterion = CrossEntropyLoss(ignore_index=full_dataset.pad_idx)
optimizer = Adam(model.parameters(), lr=5e-6)

# logger.level("INFO")

for epoch in range(3):
    for seq, masked_tgt in train_dataloader:
        optimizer.zero_grad()

        logger.debug(seq)
        seq = seq.to(device)
        masked_tgt = masked_tgt.to(device)
        logger.debug(seq)
        logger.debug(masked_tgt)
        logits = model(seq)
        logger.debug(logits.shape)
        logits = logits.reshape([-1, logits.shape[-1]])
        logger.debug(logits.shape)
        masked_tgt = masked_tgt.reshape([-1])
        logger.debug(masked_tgt.shape)
        logger.debug(logits)
        logger.debug(masked_tgt)
        loss = criterion(logits, masked_tgt)
        logger.info(loss)
        loss.backward()
        optimizer.step()
