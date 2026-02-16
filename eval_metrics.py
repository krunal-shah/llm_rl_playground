import editdistance
from logger import logger
from writer import writer


def compute_generation_metrics(prompts, golds, preds, step):
    num_samples = len(prompts)
    accuracy = 0
    edit_distance = 0
    incorrects = []
    for prompt, gold, pred in zip(prompts, golds, preds):
        if gold == pred:
            accuracy += 1
        else:
            incorrects.append(f"{prompt=} {gold=} {pred=}")
        edit_distance += editdistance.eval(gold, pred)
        logger.debug(f"{prompt=} {gold=} {pred=}")
    incorrects = "\n".join(incorrects)
    writer.add_text("Incorrects", f"{incorrects}", step)
    logger.info(f"accuracy = {accuracy / num_samples}, edit_distance = {edit_distance / num_samples}, {num_samples=}")
    writer.add_scalar("accuracy", accuracy / num_samples, step)
    writer.add_scalar("edit_distance", edit_distance / num_samples, step)
    writer.add_scalar("num_samples", num_samples, step)
