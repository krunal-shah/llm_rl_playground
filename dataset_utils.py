import torch
# from loguru import logger


def process_src_tgt_tensors(src, tgt, max_length, pad_idx):
    src_length, tgt_length = src.shape[0], tgt.shape[0]
    # logger.debug(f"{src_length=}, {tgt_length=}")
    seq_length = src_length + tgt_length
    pad_length = max_length - seq_length
    pad = torch.ones([pad_length], dtype=src.dtype) * pad_idx

    # concatenate all three
    seq = torch.cat([src, tgt, pad], axis=0)
    # mask = torch.cat([torch.ones([seq_length]), torch.zeros([pad_length])], axis=0)
    # everything but tgt is masked
    masked_tgt = torch.cat([
                                torch.ones([src_length - 1], dtype=src.dtype) * pad_idx,
                                tgt,
                                torch.ones([pad_length + 1], dtype=src.dtype) * pad_idx
                            ], axis=0)

    # everything but src is masked
    src_masked = torch.cat([
                                src,
                                torch.ones([pad_length + tgt_length], dtype=src.dtype) * pad_idx
                            ], axis=0)

    return seq, masked_tgt, src_masked
