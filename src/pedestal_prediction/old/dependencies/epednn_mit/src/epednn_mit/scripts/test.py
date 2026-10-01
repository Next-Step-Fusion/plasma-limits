import argparse
import importlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of model to load and test')
    parser.add_argument('backend', type=str, help='Name of backend to use for evaluating model',
        choices=['tensorflow', 'pytorch', 'jax'],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        module = importlib.import_module(f'epednn_mit.models.{args.model_name}.{args.backend}_model')
        model_out, test_out = getattr(module, f'test_epednn_mit_{args.model_name}_{args.backend}')()
    except ImportError:
        raise NotImplementedError(f'Test script for {args.model_name} using {args.backend} backend not yet implemented')
    print('Evaluation:', model_out)
    assert model_out.shape == test_out.shape
    assert np.all(np.isclose(model_out, test_out))


if __name__ == '__main__':
    main()
