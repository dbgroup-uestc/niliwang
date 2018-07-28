from models.ModelA import Model as A

import sys


def get_model(args):
    if args.nn_model == 'A':
        return A(args)
    print('no such model:', args.nn_model)
    sys.exit(1)
