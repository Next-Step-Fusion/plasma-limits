"""
JAX versions of the EPEDNN-MIT model

For questions, bugs, or feedback, contact @theo-brown.
"""

from os import PathLike
from pathlib import Path
import pickle
from typing import Any, Final, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp


def load_params_from_pickle(
    pickle_file: PathLike | str,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Loads EPEDNN-mit model parameters and normalization statistics from a pickle file.

    Args:
      pickle_file: Path to the pickle file containing the model data.

    Returns:
      A tuple containing two dictionaries:
        - stats: Normalization statistics (input/output mean and variance).
        - params: Model parameters remapped to Flax Linen layer names.
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    stats = {
        "input_mean": jnp.asarray(data["input.mean"]),
        "input_variance": jnp.asarray(data["input.variance"]),
        "output_mean": jnp.asarray(data["output.mean"]),
        "output_variance": jnp.asarray(data["output.variance"]),
    }

    # Remap the weight names
    flax_params = {}
    pkl_to_flax_labels = {
        "hidden_layer0": "Dense_0",
        "hidden_layer1": "Dense_1",
        "output_layer": "Dense_2",
    }
    for pkl_label, flax_label in pkl_to_flax_labels.items():
        flax_params[flax_label] = {
            "kernel": data[f"{pkl_label}.weight"],
            "bias": data[f"{pkl_label}.bias"],
        }

    return stats, {"params": flax_params}


def load_ensemble_params_from_pickle(
    pickle_files: Sequence[PathLike | str],
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Loads EPEDNN-mit parameters and statistics for an ensemble of models from pickle files.

    Args:
      pickle_files: A sequence of paths to pickle files, each containing data for
        one ensemble member.

    Returns:
      A tuple containing two dictionaries:
        - ensemble_stats: Stacked normalization statistics for the ensemble.
        - ensemble_weights: Stacked model parameters for the ensemble, nested
          under "params": {"ensemble": ...}.
    """
    all_stats = []
    all_weights = []

    for f in pickle_files:
        stats, weights = load_params_from_pickle(f)
        all_stats.append(stats)
        all_weights.append(weights)

    ensemble_stats = jax.tree.map(lambda *args: jnp.stack(args), *all_stats)
    ensemble_weights = jax.tree.map(lambda *args: jnp.stack(args), *all_weights)

    return ensemble_stats, {"params": {"ensemble": ensemble_weights["params"]}}


def normalize(data: jax.Array, *, mean: jax.Array, stddev: jax.Array) -> jax.Array:
    """Normalizes data to have mean 0 and stddev 1.

    Args:
      data: The input data array to be normalized.
      mean: The mean of the data distribution.
      stddev: The standard deviation of the data distribution.

    Returns:
      The normalized data array.
    """
    return (data - mean) / jnp.where(stddev == 0, 1, stddev)


def unnormalize(data: jax.Array, *, mean: jax.Array, stddev: jax.Array) -> jax.Array:
    """Unnormalizes data from mean 0 and stddev 1 to the original distribution.

    Args:
      data: The normalized data array.
      mean: The mean of the original data distribution.
      stddev: The standard deviation of the original data distribution.

    Returns:
      The unnormalized data array.
    """
    return data * jnp.where(stddev == 0, 1, stddev) + mean


class EPEDNNmit(nn.Module):
    """A single member of the EPEDNN-mit model ensemble.

    Attributes:
      hidden_dims: Dimensions of the hidden layers.
      output_dim: Dimension of the output layer.
    """

    hidden_dims: Final[Sequence[int]] = (32, 32)
    output_dim: Final[int] = 2

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        input_mean: jax.Array,
        input_variance: jax.Array,
        output_mean: jax.Array,
        output_variance: jax.Array,
    ) -> jax.Array:
        """Forward pass of the model.

        Args:
          x: Input data.
          input_mean: Mean of the input features for normalization.
          input_variance: Variance of the input features for normalization.
          output_mean: Mean of the output targets for unnormalization.
          output_variance: Variance of the output targets for unnormalization.

        Returns:
          The model's prediction.
        """
        x = normalize(x, mean=input_mean, stddev=jnp.sqrt(input_variance))

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        x = nn.Dense(self.output_dim)(x)

        x = unnormalize(x, mean=output_mean, stddev=jnp.sqrt(output_variance))

        return x


class EPEDNNmitEnsemble(nn.Module):
    """An ensemble of EPEDNN-mit models.

    Attributes:
      n_ensemble: The number of models in the ensemble.
      hidden_dims: Dimensions of the hidden layers for each model.
      output_dim: Dimension of the output layer for each model.
    """

    n_ensemble: Final[int] = 10
    hidden_dims: Final[Sequence[int]] = (32, 32)
    output_dim: Final[int] = 2

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        input_mean: jax.Array,
        input_variance: jax.Array,
        output_mean: jax.Array,
        output_variance: jax.Array,
    ) -> jax.Array:
        """Forward pass of the ensemble model.

        Args:
          x: Input data.
          input_mean: Mean of the input features for each ensemble member.
          input_variance: Variance of the input features for each ensemble member.
          output_mean: Mean of the output targets for each ensemble member.
          output_variance: Variance of the output targets for each ensemble member.

        Returns:
          The mean prediction across all ensemble members.
        """
        ensemble = nn.vmap(
            EPEDNNmit,
            variable_axes={"params": 0},  # Separate params per member
            split_rngs={"params": True},  # Separate RNGs per member
            in_axes=(None, 0, 0, 0, 0),  # Shared input x, separate stats per member
            axis_size=self.n_ensemble,
        )(self.hidden_dims, self.output_dim, name="ensemble")
        ensemble_predictions = ensemble(
            x, input_mean, input_variance, output_mean, output_variance
        )
        return jnp.mean(ensemble_predictions, axis=0)


def test_epednn_mit_sparc_jax():
    """Tests the EPEDNNmitEnsemble model loading and inference.

    Loads ensemble parameters from pickle files found in the same directory
    and runs a test inference.

    Returns:
      A tuple containing the model output and the expected test output.
    """
    test_in = jnp.atleast_2d(jnp.array([10.0, 9.0, 1.85, 0.57, 2.0, 0.45, 30.0, 1.2, 2.0]))
    test_out = jnp.atleast_2d(jnp.array([3.19934668e02, 4.11556657e-02]))
    root = Path(__file__).resolve().parent
    weights_files = [Path(f).resolve() for f in sorted(root.glob("*sparc*.pkl"))]
    stats, params = load_ensemble_params_from_pickle(weights_files)
    model = EPEDNNmitEnsemble()
    model_out = model.apply(params, test_in, **stats)
    return model_out, test_out
