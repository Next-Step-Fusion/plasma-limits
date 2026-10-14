import copy
import pickle
from pathlib import Path
import numpy as np
import torch

from ...utils.load import load_weights

default_dtype = torch.get_default_dtype()
np_dtype = np.float64 if default_dtype == torch.float64 else np.float32


class EPEDNN_SPARC(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n_inputs = kwargs.get('n_input', 1)
        self.n_outputs = kwargs.get('n_output', 1)
        self._base_activation = torch.nn.ReLU()
        self._input_mean_tensor = torch.tensor(np.atleast_2d(kwargs.get('input.mean', [0.0])).astype(np_dtype), dtype=default_dtype)
        self._input_variance_tensor = torch.tensor(np.atleast_2d(kwargs.get('input.variance', [1.0])).astype(np_dtype), dtype=default_dtype)
        self._output_mean_tensor = torch.tensor(np.atleast_2d(kwargs.get('output.mean', [0.0])).astype(np_dtype), dtype=default_dtype)
        self._output_variance_tensor = torch.tensor(np.atleast_2d(kwargs.get('output.variance', [1.0])).astype(np_dtype), dtype=default_dtype)
        self._model = args[0] if len(args) > 0 else torch.nn.Identity()

    def forward(self, x):
        y = (x - self._input_mean_tensor) / torch.sqrt(self._input_variance_tensor)
        for i in range(len(self._model) - 1):
            y = self._model[f'hidden{i:d}'](y)
            y = self._base_activation(y)
        ynorm = self._model['output'](y)
        return ynorm * torch.sqrt(self._output_variance_tensor) + self._output_mean_tensor


class Average(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n_inputs = kwargs.get('n_input', 1)
        self.n_outputs = kwargs.get('n_output', 1)
        self._model_list = args[0] if len(args) > 0 else [torch.nn.Identity()]

    def forward(self, x):
        y_stack = [model(x) for model in self._model_list]
        return torch.mean(torch.stack(y_stack), dim=0)


def configure_weights_for_pytorch(weights_dict):
    config = {}
    params = {}
    layers = []
    for k, v in weights_dict.items():
        ks = k.split('.')
        if ks[0] in ['input', 'output']:
            config[k] = copy.deepcopy(weights_dict[k])
        if ks[0].startswith('hidden_layer'):
            tag = 'hidden' + ks[0][12:]
            var = '.'.join(['_model', f'{tag}'] + ks[1:])
            if ks[1] == 'weight':
                params[f'{var}'] = torch.transpose(torch.tensor(np.array(v).astype(np_dtype), dtype=default_dtype), 0, 1)
                if tag == 'hidden0':
                    config['n_input'] = int(v.shape[0])
                layers.append(int(v.shape[1]))
            if ks[1] == 'bias':
                params[f'{var}'] = torch.tensor(np.array(v).astype(np_dtype), dtype=default_dtype)
        if ks[0].startswith('output_layer'):
            tag = 'output'
            var = '.'.join(['_model', f'{tag}'] + ks[1:])
            if ks[1] == 'weight':
                params[f'{var}'] = torch.transpose(torch.tensor(np.array(v).astype(np_dtype), dtype=default_dtype), 0, 1)
                config['n_output'] = int(v.shape[1])
            if ks[1] == 'bias':
                params[f'{var}'] = torch.tensor(np.array(v).astype(np_dtype), dtype=default_dtype)
    config['common_nodes'] = copy.deepcopy(layers)
    return config, params


def generate_pytorch_sparc_model(config, params):
    base_activation = torch.nn.ReLU()
    layers = torch.nn.ModuleDict()
    n_previous_layer = config.get('n_input', 1)
    for i, neurons in enumerate(config.get('common_nodes', [])):
        layers.update({f'hidden{i:d}': torch.nn.Linear(n_previous_layer, neurons, dtype=default_dtype)})
        n_previous_layer = neurons
    layers.update({f'output': torch.nn.Linear(n_previous_layer, config.get('n_output', 1), dtype=default_dtype)})
    model = EPEDNN_SPARC(layers, **config)
    if params:
        with torch.no_grad():
            model.load_state_dict(params)
    return model


def generate_epednn_mit_sparc_pytorch(weights_list):
    n_inputs = 1
    n_outputs = 1
    model_list = []
    for i, weights_dict in enumerate(weights_list):
        config, params = configure_weights_for_pytorch(weights_dict)
        model_list.append(generate_pytorch_sparc_model(config, params))
        n_inputs = config.get('n_input', 1)
        n_outputs = config.get('n_output', 1)
    return Average(model_list, n_input=n_inputs, n_output=n_outputs)


def test_epednn_mit_sparc_pytorch():
    test_in = np.atleast_2d([10.0, 9.0, 1.85, 0.57, 2.0, 0.45, 30.0, 1.2, 2.0]).astype(np_dtype)
    test_out = np.atleast_2d([3.19934668e+02, 4.11556657e-02]).astype(np_dtype)
    root = Path(__file__).resolve().parent
    weights_files = [Path(f).resolve() for f in sorted(root.glob('*sparc*.pkl'))]
    weights_list = load_weights(weights_files)
    model = generate_epednn_mit_sparc_pytorch(weights_list)
    model_out_tensor = model(torch.from_numpy(test_in))
    model_out = model_out_tensor.detach().cpu().numpy()
    return model_out, test_out
