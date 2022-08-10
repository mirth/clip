from tensorboardX import SummaryWriter
from ignite.engine import create_supervised_evaluator, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import RunningAverage

def attach_loggers(
    model,
    trainer,
    train_loader,
    val_loader,
    metrics,
    log_dir='./log',
):
    # model_device = next(model.parameters()).device
#    evaluator = create_supervised_evaluator(model, metrics=metrics, device=model_device)
    summary_writer = SummaryWriter(log_dir=log_dir)

    log_interval = 1

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])
    #pbar = ProgressBar()
    #pbar.attach(evaluator)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        summary_writer.add_scalar('training/loss', engine.state.output, engine.state.iteration)

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_training_results(engine):
#         evaluator.run(train_loader)

#         metrics = evaluator.state.metrics
#         add_scalars(summary_writer, 'training', metrics, engine.state.epoch)
#         print(f'Training Results - Epoch: {engine.state.epoch} {compose_metrics_string(metrics)}')

#    @trainer.on(Events.EPOCH_COMPLETED)
#    def log_validation_results(engine):
#        evaluator.run(val_loader)
#        metrics = evaluator.state.metrics

#        add_scalars(summary_writer, 'validation', metrics, engine.state.epoch)
#        print(f'Validation Results - Epoch: {engine.state.epoch} {compose_metrics_string(metrics)}')

def compose_metrics_string(metrics):
    parts = []
    for metric_name, metric_value in metrics.items():
        parts.append(f'Avg {metric_name}: {metric_value:.2f}')

    return ', '.join(parts)

def add_scalars(summary_writer, phase, metrics, epoch):
    for metric_name, metric_value in metrics.items():
        summary_writer.add_scalar(f'{phase,}/avg_{metric_name}', metric_value, epoch)
