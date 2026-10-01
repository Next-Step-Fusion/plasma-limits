import copy
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf

from ...utils.load import load_weights

default_dtype = tf.keras.backend.floatx()


def recursive_set_weights_tensorflow(model, weights_dict):
    if hasattr(model, 'get_layer'):
        for key in weights_dict:
            layer = model.get_layer(key)
            recursive_set_weights_tensorflow(layer, weights_dict[key])
    elif hasattr(model, 'set_weights'):
        weights = model.get_weights()
        for i, variable in enumerate(model.weights):
            var = variable.name.split('/')[-1].split(':')[0]
            if var in weights_dict:
                weights[i] = np.array(weights_dict[var], dtype=default_dtype)
        model.set_weights(weights)


def configure_weights_for_tensorflow(weights_dict):
    config = {}
    params = {}
    layers = []
    for k, v in weights_dict.items():
        ks = k.split('.')
        if ks[0] == 'input':
            tag = '_'.join(ks)
            config[f'{tag}'] = copy.deepcopy(v)
        if ks[0] == 'output':
            tag = '_'.join(ks)
            config[f'{tag}'] = copy.deepcopy(v)
        if ks[0].startswith('hidden_layer'):
            tag = 'hidden' + ks[0][12:]
            if tag not in params:
                params[f'{tag}'] = {}
            if ks[1] == 'weight':
                params[f'{tag}']['kernel'] = copy.deepcopy(v)
                if tag == 'hidden0':
                    config['n_input'] = int(v.shape[0])
                layers.append(int(v.shape[1]))
            if ks[1] == 'bias':
                params[f'{tag}']['bias'] = copy.deepcopy(v)
        if ks[0].startswith('output_layer'):
            tag = 'output'
            if tag not in params:
                params[f'{tag}'] = {}
            if ks[1] == 'weight':
                params[f'{tag}']['kernel'] = copy.deepcopy(v)
                config['n_output'] = int(v.shape[1])
            if ks[1] == 'bias':
                params[f'{tag}']['bias'] = copy.deepcopy(v)
    config['common_nodes'] = copy.deepcopy(layers)
    return config, params


def generate_tensorflow_sparc_model(config, params):
    layers = []
    if len(config.get('input_names', [])) > 0:
        layers.append(tf.keras.layers.Normalization(axis=-1, mean=config.get('input_mean', []), variance=config.get('input_variance', [])))
    for i, neurons in enumerate(config.get('common_nodes', [])):
        layers.append(tf.keras.layers.Dense(neurons, activation='relu', name=f'hidden{i:d}'))
    layers.append(tf.keras.layers.Dense(config.get('n_output', 1), activation=None, name=f'output'))
    if len(config.get('output_names', [])) > 0:
        layers.append(tf.keras.layers.Normalization(axis=-1, mean=config.get('output_mean', []), variance=config.get('output_variance', []), invert=True))
    model = tf.keras.models.Sequential(layers)
    model.build((None, config.get('n_input', 1)))
    recursive_set_weights_tensorflow(model, params)
    return model


def generate_epednn_mit_sparc_tensorflow(weights_list):
    n_inputs = 1
    model_list = []
    for i, weights_dict in enumerate(weights_list):
        config, params = configure_weights_for_tensorflow(weights_dict)
        model_list.append(generate_tensorflow_sparc_model(config, params))
        n_inputs = config.get('n_input', 1)
    input_shape = tf.keras.layers.Input(shape=(n_inputs, ))
    x = [m(input_shape) for m in model_list]
    output_shape = tf.keras.layers.Average()(x)
    return tf.keras.models.Model(inputs=input_shape, outputs=output_shape)


def test_epednn_mit_sparc_tensorflow():
    test_in = np.atleast_2d([10.0, 9.0, 1.85, 0.57, 2.0, 0.45, 30.0, 1.2, 2.0])
    test_out = np.atleast_2d([3.19934668e+02, 4.11556657e-02])
    root = Path(__file__).resolve().parent
    weights_files = [Path(f).resolve() for f in sorted(root.glob('*sparc*.pkl'))]
    weights_list = load_weights(weights_files)
    model = generate_epednn_mit_sparc_tensorflow(weights_list)
    model_out = model.predict(test_in)
    return model_out, test_out
