import baker
import torch
from torch import nn

from ignite.engine import create_supervised_trainer

from attach_loggers import attach_loggers
from data_loaders import *
from augmentations import *
from metrics import *
from clip import CLIP


@baker.command
def run(
    dataset_path='/home/tolik/data/open-images-v6_64',
    train_batch_size=64,
    num_workers=12,
    accelerator='cuda',
    max_epochs=100,
    temperature=1.0,
):
    max_text_length = 64
    train_val_augmentations = (
        (aug1(), make_text_transforms(max_length=max_text_length)),
        (aug1(), make_text_transforms(max_length=max_text_length)),
    )

    train_loader, val_loader = get_openimages6_dataloaders(
        dataset_path,
        train_val_augmentations,
        train_batch_size,
        num_workers=num_workers,
    )

    image_encoder = make_image_encoder()
    text_encoder = make_text_encoder(max_text_length)

    model = CLIP(image_encoder, text_encoder, temperature=temperature).to(accelerator)

    checkpoint = new_checkpoint(model, train_loader)
    metrics = get_metrics(checkpoint['criterion'])

    trainer = checkpoint['trainer']
    attach_loggers(
        model=model,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=metrics
    )

    trainer.run(train_loader, max_epochs=max_epochs)

def make_text_transforms(max_length):
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def fn(text):
        text = tokenizer.encode(text, max_length=max_length, padding='max_length', truncation=True)
        text = torch.from_numpy(np.array(text))

        return text
    return fn

def make_image_encoder():
    import torchvision

    model = torchvision.models.resnet50(pretrained=False)
    model.fc = Identity()

    return model

def make_text_encoder(max_length):
    from transformers import BertModel, BertConfig

    model = BertModel(BertConfig(max_position_embeddings=max_length)) #.from_pretrained('bert-base-uncased')

    return model

def new_checkpoint(model, train_loader, device='cuda'):
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=0.001) #,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    criterion = SimmetricCrossEntropy()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    return dict(
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        criterion=criterion,
    )

class SimmetricCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, y):
        ground_truth = torch.arange(len(logits)).long().to(logits.device)
        loss = (self.criterion(logits, ground_truth) + self.criterion(logits.T, ground_truth)) / 2
        loss = loss.mean()

        return loss

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == '__main__':
    baker.run()
