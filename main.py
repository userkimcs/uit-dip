import argparse
import torchvision

from helpers import image_show, DatasetLoader
import logging

from model import ModelTrainer
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Data path", required=True)
    parser.add_argument("--batch_size", help="Training batch size", required=False, default=32, type=int)
    parser.add_argument("--workers", required=False, default=4, type=int)
    parser.add_argument("--epochs", required=False, default=5, type=int)
    parser.add_argument("--sample", action='store_true', default=False)

    args = parser.parse_args()

    dataloader = DatasetLoader(data_path=args.data_path, batch_size=args.batch_size, workers=args.workers).load()

    logger.info("Hello")
    # sample
    if args.sample:
        inputs, classes = dataloader.next('train')
        out = torchvision.utils.make_grid(inputs, nrow=8)
        image_show(out)

    list_models = [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
    ]

    for model in list_models:
        trainer = ModelTrainer(dataset=dataloader, model_name=model, num_epochs=args.epochs, es={})
        trainer.fit()

    logger.info("Done")
