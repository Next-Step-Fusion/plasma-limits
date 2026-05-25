import numpy as np
from dataclasses import dataclass

ELEMENTARY_CHARGE = 1.60217662e-19  # C
VACUUM_PERMEABILITY = 4 * np.pi * 1e-7  # H/m AKA mu0, the magnetic constant
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m AKA epsilon0


def calc_collisionality_edge(
    ne_edge: float, Te_edge: float, q_star: float, R0: float, epsilon: float
) -> float:
    """Compute normalized edge collisionality.

    Input:
        - ne_edge: [m^-3] edge electron density
        - Te_edge: [eV] edge electron temperature
        - q_star: [-] normalized safety factor
        - R0: [m] major radius
        - epsilon: [-] inverse aspect ratio (a0/R0)
    Output:
        - collisionality_edge: [-] normalized edge collisionality
    """
    coulomb_log = calc_coulomb_log_edge(ne_edge, Te_edge)
    return (
        (ELEMENTARY_CHARGE**4 / (4 * np.pi * VACUUM_PERMITTIVITY**2))
        * coulomb_log
        * ne_edge
        * q_star
        * R0
        * epsilon ** (-3 / 2)
        * (2 * ELEMENTARY_CHARGE * Te_edge) ** (-2)
    )


def calc_coulomb_log_edge(ne_edge: float, Te_edge: float) -> float:
    """Compute Coulomb logarithm at the plasma edge.

    Input:
        - ne_edge: [m^-3] edge electron density
        - Te_edge: [eV] edge electron temperature
    Output:
        - Coulomb_log_edge: [-] Coulomb logarithm
    """
    debye_length = 7430 * np.sqrt(Te_edge / ne_edge)
    plasma_parameter = 12 * np.pi * ne_edge * debye_length**3
    return np.log(plasma_parameter)


def calc_q_star(
    Bt0: float, R0: float, epsilon: float, kappa: float, Ip: float
) -> float:
    """Compute normalized safety factor.

    Definition from Verdoolaege et al. 2021,
    with kappa_area taken to be approximately (1 + kappa^2) / 2.

    Input:
        - Bt0: [T] toroidal magnetic field at R0
        - R0: [m] major radius
        - epsilon: [-] inverse aspect ratio (a0/R0)
        - kappa: [-] elongation
        - Ip: [A] plasma current
    Output:
        - q_star: [-] normalized safety factor
    """
    return np.abs(
        (2 * np.pi / VACUUM_PERMEABILITY)
        * Bt0
        * R0
        * epsilon**2
        * ((1 + kappa**2) / 2)
        / Ip
    )


def calc_beta_edge(
    ne_edge: float, Te_edge: float, Bt0: float, R0: float, a0: float, Ip: float
) -> float:
    """Compute normalized edge plasma pressure (beta).

    Input:
        - ne_edge: [m^-3] edge electron density
        - Te_edge: [eV] edge electron temperature
        - Bt0: [T] toroidal magnetic field at R0
        - R0: [m] major radius
        - a0: [m] minor radius
        - Ip: [A] plasma current
    Output:
        - beta_edge: [-] normalized edge pressure
    """
    b_poloidal_edge = VACUUM_PERMEABILITY * Ip / (2 * np.pi * a0)
    bt_edge_outboard = Bt0 * R0 / (R0 + a0)
    b_tot_edge_outboard = np.sqrt(bt_edge_outboard**2 + b_poloidal_edge**2)
    return (
        4 * VACUUM_PERMEABILITY * ne_edge * Te_edge
        * ELEMENTARY_CHARGE * b_tot_edge_outboard**(-2)
    )


@dataclass
class PlasmaState:
    """Container for plasma equilibrium parameters at a single time point."""

    R0: float  # [m] major radius
    a0: float  # [m] minor radius
    Bt0: float  # [T] toroidal magnetic field
    kappa: float  # [-] elongation
    Ip: float  # [A] plasma current
    Te_edge: float  # [eV] edge electron temperature
    ne_edge: float  # [m^-3] edge electron density
    ne_mean: float  # [m^-3] line-averaged electron density


class DL26:
    """DL26 density limit metric.

    Computes the instability metric as the product of normalized edge
    collisionality and edge beta. A normalized_limit approaching 1 indicates
    proximity to the density limit.
    """

    normalized_limit: float
    instability_metric: float
    collisionality_edge: float
    beta_edge: float
    q_star: float

    def __init__(self, plasma_state: PlasmaState, warning_threshold: float = 0.1):
        epsilon = plasma_state.a0 / plasma_state.R0
        self.q_star = calc_q_star(
            Bt0=plasma_state.Bt0,
            R0=plasma_state.R0,
            epsilon=epsilon,
            kappa=plasma_state.kappa,
            Ip=plasma_state.Ip,
        )
        self.collisionality_edge = calc_collisionality_edge(
            ne_edge=plasma_state.ne_edge,
            Te_edge=plasma_state.Te_edge,
            q_star=self.q_star,
            R0=plasma_state.R0,
            epsilon=epsilon,
        )
        self.beta_edge = calc_beta_edge(
            ne_edge=plasma_state.ne_edge,
            Te_edge=plasma_state.Te_edge,
            Bt0=plasma_state.Bt0,
            R0=plasma_state.R0,
            a0=plasma_state.a0,
            Ip=plasma_state.Ip,
        )
        self.warning_threshold = warning_threshold
        self.instability_metric = self.collisionality_edge * self.beta_edge
        self.normalized_limit = self.instability_metric / self.warning_threshold

    def to_density_value(self) -> float:
        """Convert instability metric to an equivalent density value."""
        pass


class Greenwald:
    """Greenwald density limit.

    Computes the Greenwald density limit and the Greenwald fraction
    (ratio of mean density to the limit).
    """

    n_Gw: float  # [m^-3] Greenwald density limit
    f_Gw: float  # [-] Greenwald fraction (ne_mean / n_Gw)

    def __init__(self, plasma_state: PlasmaState):
        self.n_Gw = plasma_state.Ip / (np.pi * plasma_state.a0**2) * 1e14
        self.f_Gw = plasma_state.ne_mean / self.n_Gw
