from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from hparams import EXP_NAME

writer = SummaryWriter(log_dir=f"runs/active/{datetime.now().strftime('%m_%d_%H_%M_%S')}_{EXP_NAME}")
