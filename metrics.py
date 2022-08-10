import torch


def get_metrics(criterion):
    from ignite.metrics import Loss

    return {
        "loss": Loss(criterion),
    }
