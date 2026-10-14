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
        4 * VACUUM_PERMEABILITY * ne_edge * Te_edge * ELEMENTARY_CHARGE * b_tot_edge_outboard**(-2)
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
    state: PlasmaState
    warning_threshold: float

    normalized_metric: float
    instability_metric: float


    def __init__(self, plasma_state: PlasmaState, warning_threshold: float = 0.1):
        self.state = plasma_state
        self.warning_threshold = warning_threshold

        self.instability_metric = self._instability_at(self.state.ne_edge)
        self.normalized_metric = self.instability_metric / self.warning_threshold

    def to_limit_density(self, max_iter: int = 10, tol: float = 1e-6) -> float:
        """Return edge electron density [m^-3] at which normalized_metric = 1.0.

            ν*_edge ∝ n_e · lnΛ(n_e, T_e)
            β_edge  ∝ n_e · T_e
            => instability_metric ∝ n_e^2 · T_e · lnΛ(n_e, T_e)

        Fixed-point iteration: treat lnΛ as frozen during each step and
        rescale n by sqrt(target / current); lnΛ is recomputed inside
        _instability_at on the next pass. Converges in a few iterations
        because lnΛ depends only logarithmically on n.
        """
        target = self.warning_threshold
        n = self.state.ne_edge
        for _ in range(max_iter):
            m_trial = self._instability_at(n)
            ratio = np.sqrt(target / m_trial)
            n *= ratio
            if abs(ratio - 1.0) < tol:
                break
        return n

    def _instability_at(self, n: float) -> float:
        """Calculate instability metric at a given density n."""
        epsilon = self.state.a0 / self.state.R0
        q_star = calc_q_star(
            Bt0=self.state.Bt0,
            R0=self.state.R0,
            epsilon=epsilon,
            kappa=self.state.kappa,
            Ip=self.state.Ip,
        )
        collisionality_edge = calc_collisionality_edge(
            ne_edge=n,
            Te_edge=self.state.Te_edge,
            q_star=q_star,
            R0=self.state.R0,
            epsilon=epsilon,
        )
        beta_edge = calc_beta_edge(
            ne_edge=n,
            Te_edge=self.state.Te_edge,
            Bt0=self.state.Bt0,
            R0=self.state.R0,
            a0=self.state.a0,
            Ip=self.state.Ip,
        )
        return collisionality_edge * beta_edge * q_star


class Greenwald:
    """Greenwald density limit.

    Computes the Greenwald density limit and the Greenwald fraction
    (ratio of mean density to the limit).
    """

    n_Gw: float  # [m^-3] Greenwald density limit
    f_Gw: float  # [-] Greenwald fraction (ne_mean / n_Gw)

    def __init__(self, plasma_state: PlasmaState):
        self.n_Gw = (plasma_state.Ip * 1e-6) / (np.pi * plasma_state.a0**2) * 1e20
        self.f_Gw = plasma_state.ne_mean / self.n_Gw
