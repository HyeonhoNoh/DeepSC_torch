import torch
import socket, datetime
from config import Config
from DeepSC_torch.trainer import train, test

if __name__ == '__main__':
    print("=" * 30)
    print("Load configuration...")
    config = Config()
    config.print_options()

    print("Load Complete!")
    print("=" * 30)

    print('torch version: {}'.format(torch.__version__))
    print(config.device)
    print("=" * 30)

    if config.param['test']:
        test(config)

    else:
        train(config)
    print("Done!")
    print("=" * 30)
