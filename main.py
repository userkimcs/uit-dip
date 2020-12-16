import argparse
import torchvision

from helpers import image_show, DatasetLoader
import logging
from glob import glob
from model import ModelTrainer, ModelEval, ImageVisualize
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Data path", required=True)
    parser.add_argument("--model_path", required=False)
    parser.add_argument("--batch_size", help="Training batch size", required=False, default=32, type=int)
    parser.add_argument("--workers", required=False, default=4, type=int)
    parser.add_argument("--epochs", required=False, default=5, type=int)
    parser.add_argument("--sample", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False)

    args = parser.parse_args()
    logger.info("Hello")

    if args.eval or args.train:
        dataloader = DatasetLoader(data_path=args.data_path, batch_size=args.batch_size, workers=args.workers).load()

        # sample
        if args.sample:
            inputs, classes = dataloader.next('train')
            out = torchvision.utils.make_grid(inputs, nrow=8)
            image_show(out)

        if args.train:
            list_models = [
                'resnet18',
                'densenet121',
                'mnasnet0_75',
                'vgg11',
            ]

            logger.info("Training mode")
            for model in list_models:
                trainer = ModelTrainer(dataset=dataloader, model_name=model, num_epochs=args.epochs, es={})
                trainer.run()

        if args.eval:
            for model_path in glob("models/*.pt"):
                logger.info("Eval model " + model_path)
                evaluator = ModelEval(dataset=dataloader, model_path=model_path)
                evaluator.run()

    if args.demo:
        vs = ImageVisualize(model_path=args.model_path)
        vs.run(args.data_path)

    logger.info("Exit 1")
