import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocess', default = False ,action='store_true', help = 'preprocess')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch_size')


    return parser.parse_args()