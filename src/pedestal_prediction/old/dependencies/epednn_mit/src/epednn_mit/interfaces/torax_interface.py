"""TORAX interface for the EPEDNN-mit pedestal model.

Please cite [M. Muraca et al. 2025 Nucl. Fusion 65
096010](https://doi.org/10.1088/1741-4326/adf656) in any works using this model.
Currently, this model is only valid for the SPARC parameter space, as specified
in https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/sparc.

This file defines the four necessary classes to use the model in TORAX:
1. A wrapper class, which closes over the EPEDNN-mit model, parameters, and
   statistics. This is necessary to make the model hashable.
2. A TORAX pedestal model, which calls the wrapper within its
   _call_implementation. This is what will be called by TORAX.
3. A TORAX pedestal model Pydantic config, which is used to build the pedestal
   model and runtime params. This allows us to use the model from a TORAX
   config.
4. A runtime params class, which contains any input parameters specific to the
   EPEDNN-mit model. This will be passed to the pedestal model by TORAX at each
   time step.
"""

import dataclasses
import pathlib
from typing import Annotated, Any, Final, Literal, TypeAlias

import chex
from epednn_mit.models.sparc import jax_model as epednn_mit_jax_model
import jax
from jax import numpy as jnp
import jaxtyping as jt
import torax
from torax._src import constants
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.physics import formulas
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

# pylint: disable=invalid-name


EPEDNNmitStats: TypeAlias = dict[str, jax.Array]
EPEDNNmitParams: TypeAlias = dict[str, Any]
EPEDNNmitMachine: TypeAlias = Literal["sparc"]

# For definitions of these parameters, see the EPEDNN-mit SPARC README:
# https://github.com/aaronkho/epednn_mit/blob/main/src/epednn_mit/models/sparc/README.txt
_SPARC_INPUT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "Ip": (1.6, 14.3),
    "Bt": (7.2, 12.2),
    "R": (1.85, 1.85),
    "a": (0.57, 0.57),
    "kappa": (1.53, 2.29),
    "delta": (0.39, 0.59),
    "neped": (2.84, 90.235),
    "betan": (0.8, 1.6),
    "zeff": (1.3, 2.5),
}
_SPARC_DEVICE_MAJOR_RADIUS: Final[float] = 1.85
_SPARC_DEVICE_MINOR_RADIUS: Final[float] = 0.57


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax.pedestal.RuntimeParams):
  """Runtime params for the EPEDNNmitPedestalModel."""

  n_e_ped: jt.Float[jt.Scalar, ""]  # [m^-3]
  T_i_T_e_ratio: jt.Float[jt.Scalar, ""]
  P_ped_multiplier: jt.Float[jt.Scalar, ""]


class EPEDNNmitPedestalModelWrapper:
  """Wrapper for the EPEDNN-mit pedestal model.

  Captures the EPEDNN-mit model, parameters, and statistics as attributes of
  the class. Hashes by value (machine name), not id.
  """

  def __init__(self, machine: EPEDNNmitMachine):
    self.machine = machine

    # Freeze in parameters of the model specific to the machine.
    match self.machine:
      case "sparc":
        model_dir = pathlib.Path(epednn_mit_jax_model.__file__).parent
        model_weights = sorted(model_dir.glob("epednn_mit_sparc_*.pkl"))
        self._stats, self._params = (
            epednn_mit_jax_model.load_ensemble_params_from_pickle(model_weights)
        )
        self.model = epednn_mit_jax_model.EPEDNNmitEnsemble()
        self.input_lower_bounds = jnp.array(
            [_SPARC_INPUT_BOUNDS[key][0] for key in _SPARC_INPUT_BOUNDS]
        )
        self.input_upper_bounds = jnp.array(
            [_SPARC_INPUT_BOUNDS[key][1] for key in _SPARC_INPUT_BOUNDS]
        )
        self.R_0 = _SPARC_DEVICE_MAJOR_RADIUS
        self.a_0 = _SPARC_DEVICE_MINOR_RADIUS
      case _:
        raise ValueError(
            f"Unsupported machine: {machine}. Only SPARC is supported."
        )

  def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    P_ped_kPa, pedestal_width_psi_norm = self.model.apply(
        self._params, inputs, **self._stats
    )
    return P_ped_kPa, pedestal_width_psi_norm

  def __hash__(self):
    return hash(self.machine)


@dataclasses.dataclass(frozen=True, eq=False)
class EPEDNNmitPedestalModel(pedestal_model_lib.PedestalModel):
  """TORAX pedestal model using EPEDNN-mit to predict pressure and width."""

  machine: EPEDNNmitMachine = "sparc"
  # The following fields are set by __post_init__.
  model: EPEDNNmitPedestalModelWrapper = dataclasses.field(init=False)

  def __post_init__(self):
    # Need to use __setattr__  to install attributes as this is a frozen
    # dataclass.
    object.__setattr__(
        self, "model", EPEDNNmitPedestalModelWrapper(self.machine)
    )
    super().__post_init__()

  def _prepare_inputs(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      previous_rho_norm_ped_top: jax.Array | None = None,
  ) -> jt.Float[jt.Array, "9"]:
    """Prepares the inputs for EPEDNN-mit.

    When ``previous_rho_norm_ped_top`` is provided (and is not the
    placeholder ``inf``), Z_eff and n_e_ped are evaluated at the mtanh
    "pedestal" location (ψ_ped = 1 - Δ). n_e_top is read from the
    evolved ``core_profiles.n_e`` at rho_norm_top, then corrected to ψ_ped
    using the assumed mtanh relationship below.

    mtanh pedestal density profile (Snyder et al., PPCF 46, 2004):

    Neglecting the contribution of the core density profile:

      n(ψ) = n_sep + a₀·[tanh(1) - tanh(2(ψ - ψ_mid)/Δ)]

    where ψ_mid = 1 - Δ/2.  Key locations:

      ψ_top = 1 - 1.5Δ  →  tanh arg = -2  →  EPED-NN output / TORAX top
      ψ_ped = 1 - Δ      →  tanh arg = -1  →  EPED-NN input location
      ψ_mid = 1 - Δ/2    →  tanh arg =  0  →  inflection point

    Profile values at these locations::

      n_top = n_sep + a₀·(tanh 1 + tanh 2)
      n_ped = n_sep + a₀·2·tanh 1

    Eliminating a₀:

      n_ped = n_sep + (n_top - n_sep)·C
      C = 2·tanh(1) / (tanh(1) + tanh(2)) ≈ 0.883

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core plasma profiles.
      previous_rho_norm_ped_top: Previous timestep's rho_norm at the pedestal
        top. ``jnp.inf`` signals first timestep (placeholder).

    Returns:
      9-element float32 array of clipped EPED-NN inputs.
    """
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    _, _, beta_N = formulas.calculate_betas(core_profiles, geo)

    # -- Convert ρ_top (pedestal top) to ρ_ped (mtanh pedestal location) --
    # EPED-NN expects inputs evaluated at ψ_ped = 1 − Δ, but TORAX tracks
    # ρ_top (= ψ_top = 1 − 1.5Δ). This section maps ρ_top → ψ_top → Δ →
    # ψ_ped → ρ_ped so we can sample profiles at the correct location.

    # Default ρ_top to 0.9 when no previous pedestal output is available.
    if previous_rho_norm_ped_top is not None:
      safe_rho_top = jnp.where(
          jnp.isinf(previous_rho_norm_ped_top),
          jnp.float32(0.9),
          previous_rho_norm_ped_top,
      )
    else:
      safe_rho_top = jnp.float32(0.9)

    # Map ρ → ψ using the normalised poloidal flux profile.
    psi_face = core_profiles.psi.face_value()
    psi_norm = (core_profiles.psi.value - psi_face[0]) / (
        psi_face[-1] - psi_face[0]
    )
    psi_top = jnp.interp(safe_rho_top, geo.rho_norm, psi_norm)

    # Δ = (1 − ψ_top) / 1.5  ;  ψ_ped = 1 − Δ
    delta_psi = (1.0 - psi_top) / 1.5
    psi_ped = 1.0 - delta_psi

    # Map ψ_ped back to ρ_ped.
    rho_ped = jnp.interp(psi_ped, psi_norm, geo.rho_norm)

    # -- n_e from profile at ρ_top, then mtanh correction to ψ_ped --
    n_e_top = jnp.interp(safe_rho_top, geo.rho_norm, core_profiles.n_e.value)
    _C = 2.0 * jnp.tanh(1.0) / (jnp.tanh(1.0) + jnp.tanh(2.0))
    n_e_sep = core_profiles.n_e.face_value()[-1]
    n_e_ped = n_e_sep + (n_e_top - n_e_sep) * _C

    # -- Z_eff at the pedestal location --
    ped_idx = jnp.argmin(jnp.abs(geo.rho_norm - rho_ped))
    Z_eff_ped = core_profiles.Z_eff[ped_idx]

    raw_inputs = jnp.array(
        [
            core_profiles.Ip_profile_face[-1] * 1e-6,  # [MA]
            geo.B_0,  # [T]
            self.model.R_0,  # [m]
            self.model.a_0,  # [m]
            geo.elongation_face[-1],  # []
            geo.delta_face[-1],  # []
            n_e_ped * 1e-19,  # [10^19 m^-3]
            beta_N,  # [%]
            Z_eff_ped,  # []
        ],
        # Network was trained with float32
        dtype=jnp.float32,
    )

    clipped_inputs = jnp.clip(
        raw_inputs,
        self.model.input_lower_bounds,
        self.model.input_upper_bounds,
    )

    return clipped_inputs

  @override
  def _call_implementation(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      pedestal_transition_state: (
          pedestal_transition_state_lib.PedestalTransitionState
      ),
  ) -> pedestal_model_output_lib.PedestalModelOutput:
    """Computes pedestal-top values from EPEDNN-mit pressure and width.

    Naming convention: In the EPED literature, "pedestal" (ψ_ped = 1 - Δ)
    and "pedestal top" (ψ_top = 1 - 1.5Δ) are distinct mtanh profile
    locations.  EPEDNN-mit outputs P_top and the width to the pedestal
    top, while TORAX I/O also uses "ped" to mean pedestal *top*
    (e.g. ``n_e_ped``, ``T_e_ped``, ``rho_norm_ped_top`` in
    ``PedestalModelOutput``).

    Within this function, local variables use ``_top`` suffixes when
    evaluated at the pedestal-top location (rho_norm_top) to distinguish from
    EPEDNN-mit inputs evaluated at ψ_ped.

    Args:
      runtime_params: Runtime parameters including pedestal config.
      geo: Geometry.
      core_profiles: Core plasma profiles.
      pedestal_transition_state: Pedestal transition state containing
        the previous pedestal model output.
    """
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    # Get pedestal pressure and width from EPEDNN-mit.
    inputs = self._prepare_inputs(
        runtime_params,
        geo,
        core_profiles,
        previous_rho_norm_ped_top=pedestal_transition_state.previous_pedestal_model_output.rho_norm_ped_top,
    )
    # P_top_kPa is pressure at the pedestal top.
    # pedestal_width_psi_norm is Δψ_N from ψ=1 to the pedestal top (ψ_top).
    P_top_kPa, pedestal_width_psi_norm = self.model(inputs)

    # Convert pedestal-top width from ψ_N to ρ_N.
    psi_face = core_profiles.psi.face_value()
    psi_norm = (core_profiles.psi.value - psi_face[0]) / (
        psi_face[-1] - psi_face[0]
    )
    psi_norm_ped_top = 1.0 - pedestal_width_psi_norm
    rho_norm_ped_top = jnp.interp(psi_norm_ped_top, psi_norm, geo.rho_norm)
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - rho_norm_ped_top)
    )

    # Convert pressure from kPa to Pa.
    P_top = P_top_kPa * 1e3

    # n_e_top: pedestal-top electron density (TORAX API names this "n_e_ped").
    n_e_top = runtime_params.pedestal.n_e_ped

    # -- Calculate T_e and T_i at the pedestal top from P_top --
    # Evaluate composition at the pedestal-top location.
    temperature_ratio = runtime_params.pedestal.T_i_T_e_ratio
    Z_eff_top = core_profiles.Z_eff[rho_norm_ped_top_idx]
    Z_i_top = core_profiles.Z_i[rho_norm_ped_top_idx]
    Z_impurity_top = core_profiles.Z_impurity[rho_norm_ped_top_idx]
    dilution_factor_top = jnp.where(
        Z_eff_top == 1.0,
        1.0,
        formulas.calculate_main_ion_dilution_factor(
            Z_i_top, Z_impurity_top, Z_eff_top
        ),
    )
    n_i_top = dilution_factor_top * n_e_top
    safe_Z_impurity_top = jnp.where(
        Z_eff_top == 1.0, 1.0, Z_impurity_top
    )
    n_impurity_top = jnp.where(
        Z_eff_top == 1.0,
        0.0,
        (n_e_top - Z_i_top * n_i_top) / safe_Z_impurity_top,
    )
    # P = T_e*n_e + T_i*n_i + T_i*n_imp  (assuming T_imp = T_i)
    T_e_top = (
        P_top * runtime_params.pedestal.P_ped_multiplier
    ) / (
        n_e_top
        + temperature_ratio * n_i_top
        + temperature_ratio * n_impurity_top
    ) / constants.CONSTANTS.keV_to_J
    T_i_top = temperature_ratio * T_e_top

    # TORAX API uses "ped" to mean pedestal top.
    return pedestal_model_output_lib.PedestalModelOutput(
        n_e_ped=n_e_top,
        T_i_ped=T_i_top,
        T_e_ped=T_e_top,
        rho_norm_ped_top=rho_norm_ped_top,
    )


# TODO: Keep aligned with upcoming TORAX V2 API changes where n_e_ped will be
# renamed to n_e_top for consistency with EPED naming conventions.
class EPEDNNmitConfig(torax.pedestal.BasePedestal):
  """TORAX pedestal model config using EPEDNN-mit.

  Attributes:
    n_e_ped: The electron density at the pedestal top [m^-3].
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      [dimensionless].
    P_ped_multiplier: Multiplier for the pedestal pressure (mostly used for
      sensitivity analysis) [dimensionless].
  """

  model_name: Annotated[Literal["epednn_mit"], torax_pydantic.JAX_STATIC] = (
      "epednn_mit"
  )
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  P_ped_multiplier: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  def build_pedestal_model(
      self,
  ) -> EPEDNNmitPedestalModel:
    return EPEDNNmitPedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        use_formation_model_with_internal_boundary_condition=base_runtime_params.use_formation_model_with_internal_boundary_condition,
        transition_time_width=base_runtime_params.transition_time_width,
        P_LH_hysteresis_factor=base_runtime_params.P_LH_hysteresis_factor,
        include_dW_dt_in_P_SOL=base_runtime_params.include_dW_dt_in_P_SOL,
        explicit_pedestal=base_runtime_params.explicit_pedestal,
        pedestal_profile_form=base_runtime_params.pedestal_profile_form,
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        chi_max=base_runtime_params.chi_max,
        D_e_max=base_runtime_params.D_e_max,
        V_e_max=base_runtime_params.V_e_max,
        V_e_min=base_runtime_params.V_e_min,
        pedestal_top_smoothing_width=base_runtime_params.pedestal_top_smoothing_width,
        P_ped_multiplier=self.P_ped_multiplier.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
    )
