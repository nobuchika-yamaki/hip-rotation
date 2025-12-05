"""hip_rotation_analysis_full.py

Complete analysis code corresponding to the manuscript:

    "Morphological Constraints on Axial Hip Rotation: A Comparative Mechanical Analysis Across Mammals"

This script implements, as closely as possible from the Methods description, the three-stage
analysis pipeline:

    Stage 1: Species-level parameter handling (comparative anatomical dataset)
    Stage 2: Geometric axial ROM and stiffness mapping
    Stage 3: Reduced-order passive dynamic simulations and morphological substitution tests

Notes
-----
* The manuscript describes an ellipse-based geometric congruency model driven by
  detailed morphometric measurements (femoral head / acetabular dimensions).
  Those raw morphometric inputs are not included here, so this script treats
  geometric axial ROM (aROM_geo) as given inputs (e.g., from prior geometric
  analysis or measurement).
* This code therefore reproduces the *mechanical* analyses (stiffness mapping,
  dynamics, substitution) and provides a clean hook to plug in species-specific
  geometric ROM.
* All numerical values (ROM, stiffness, etc.) are easily editable at the top
  of the file.

Usage
-----
You can either import the functions in another script / notebook:

    from hip_rotation_analysis_full import (
        SPECIES_PARAMS,
        compute_stiffness_from_rom,
        run_all_dynamic_simulations,
        run_morphological_substitution
    )

or run this file directly from the command line:

    python hip_rotation_analysis_full.py

which will:
    * print species parameters
    * compute stiffness from geometric ROM
    * run passive dynamic simulations
    * run simple morphological substitution experiments
    * print summary tables to stdout

Dependencies
------------
    numpy

You may optionally use pandas/matplotlib in a separate notebook for plotting
or tabular output. This script itself stays minimal.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np


# ================================================================
# Stage 1: Species-level parameters
# ================================================================

@dataclass
class SpeciesParams:
    """Container for species-level hip parameters used in the analysis.

    Parameters
    ----------
    name : str
        Species name (e.g., 'Human').
    aROM_internal_deg : float
        Geometric internal axial rotation (degrees).
    aROM_external_deg : float
        Geometric external axial rotation (degrees).
    stiffness_Nm_per_rad : float | None
        Hip rotational stiffness (Nm/rad). If None, it will be estimated
        from ROM using compute_stiffness_from_rom().
    inertia_trunk_kgm2 : float
        Effective trunk/pelvic rotational inertia around the hip axis.
        These values should be set using anthropometric tables and
        cross-species scaling (Winter, 2009; Kilbourne & Hoffman, 2013).
        Here we provide example relative values.
    damping_ratio : float
        Target damping ratio for the single-DOF rotational model.
    """

    name: str
    aROM_internal_deg: float
    aROM_external_deg: float
    stiffness_Nm_per_rad: float | None
    inertia_trunk_kgm2: float
    damping_ratio: float = 0.1

    @property
    def aROM_total_deg(self) -> float:
        """Total geometric axial ROM (internal + external) in degrees."""
        return self.aROM_internal_deg + self.aROM_external_deg


# Default species parameters consistent with the manuscript text.
# Internal / external values match the ranges quoted in the Results section.
# Stiffness values are filled in later via compute_stiffness_from_rom() unless
# you want to override them manually.

SPECIES_PARAMS: Dict[str, SpeciesParams] = {
    "Human": SpeciesParams(
        name="Human",
        aROM_internal_deg=42.5,  # 40–45
        aROM_external_deg=47.5,  # 45–50
        stiffness_Nm_per_rad=None,  # will be mapped from ROM
        inertia_trunk_kgm2=1.0,  # reference value (relative units)
    ),
    "Orangutan": SpeciesParams(
        name="Orangutan",
        aROM_internal_deg=25.0,  # 20–30
        aROM_external_deg=32.5,  # 30–35
        stiffness_Nm_per_rad=None,
        inertia_trunk_kgm2=0.8,  # example: somewhat lower trunk inertia
    ),
    "Dog": SpeciesParams(
        name="Dog",
        aROM_internal_deg=9.0,   # 8–10
        aROM_external_deg=16.5,  # 15–18
        stiffness_Nm_per_rad=None,
        inertia_trunk_kgm2=0.6,
    ),
    "Horse": SpeciesParams(
        name="Horse",
        aROM_internal_deg=9.0,   # <10
        aROM_external_deg=9.0,   # <10
        stiffness_Nm_per_rad=None,
        inertia_trunk_kgm2=1.5,  # larger trunk/pelvic inertia
    ),
}


# ================================================================
# Stage 2: Geometric ROM -> Stiffness mapping
# ================================================================

def compute_stiffness_from_rom(
    species: SpeciesParams,
    k_reference_Nm_per_rad: float = 120.0,
    reference_total_rom_deg: float = 90.0,
) -> float:
    """Estimate rotational stiffness from geometric ROM.

    Implements:
        K_species = K_reference * (aROM_reference / aROM_species)

    as described in the Methods, with human hip as the baseline.

    Parameters
    ----------
    species : SpeciesParams
        Species parameters with aROM_total_deg already defined.
    k_reference_Nm_per_rad : float, optional
        Reference stiffness (human), by default 120 Nm/rad.
    reference_total_rom_deg : float, optional
        Reference total ROM (human), by default 90 deg.

    Returns
    -------
    float
        Estimated stiffness K (Nm/rad).
    """
    total_rom = species.aROM_total_deg
    if total_rom <= 0:
        raise ValueError(f"Total ROM for {species.name} must be > 0.")
    k_species = k_reference_Nm_per_rad * (reference_total_rom_deg / total_rom)
    return k_species


def assign_stiffness_to_all_species(
    species_params: Dict[str, SpeciesParams],
    k_reference_Nm_per_rad: float = 120.0,
    reference_total_rom_deg: float = 90.0,
) -> None:
    """Fill in stiffness_Nm_per_rad for each species in-place if missing."""
    for sp in species_params.values():
        if sp.stiffness_Nm_per_rad is None:
            sp.stiffness_Nm_per_rad = compute_stiffness_from_rom(
                sp,
                k_reference_Nm_per_rad=k_reference_Nm_per_rad,
                reference_total_rom_deg=reference_total_rom_deg,
            )


# ================================================================
# Stage 3: Reduced-order passive dynamics
# ================================================================

def compute_damping_coefficient(
    inertia: float,
    stiffness: float,
    damping_ratio: float,
) -> float:
    """Compute viscous damping coefficient C for a target damping ratio.

    For a single-DOF system:

        I * theta¨ + C * theta˙ + K * theta = tau(t)

    The undamped natural frequency is:

        omega_n = sqrt(K / I)

    For a given damping ratio zeta:

        C = 2 * zeta * sqrt(I * K)

    Parameters
    ----------
    inertia : float
        Rotational inertia I.
    stiffness : float
        Rotational stiffness K.
    damping_ratio : float
        Desired damping ratio (e.g., 0.1).

    Returns
    -------
    float
        Damping coefficient C.
    """
    omega_n = np.sqrt(stiffness / inertia)
    C = 2.0 * damping_ratio * np.sqrt(inertia * stiffness)
    # omega_n not used directly, but kept for clarity
    return C


def simulate_passive_dynamics(
    inertia: float,
    stiffness: float,
    damping_ratio: float = 0.1,
    torque_std_Nm: float = 1.0,
    dt: float = 0.001,
    t_final: float = 600.0,
    t_transient: float = 60.0,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate passive axial hip rotation under stochastic torque.

    Uses a simple Euler–Maruyama scheme for:

        I * theta¨ + C * theta˙ + K * theta = tau(t)

    where tau(t) is Gaussian white noise.

    Parameters
    ----------
    inertia : float
        Rotational inertia I.
    stiffness : float
        Rotational stiffness K.
    damping_ratio : float, optional
        Target damping ratio, by default 0.1.
    torque_std_Nm : float, optional
        Standard deviation of torque noise, by default 1.0.
    dt : float, optional
        Time step (s), by default 0.001.
    t_final : float, optional
        Total simulation time (s), by default 600.0.
    t_transient : float, optional
        Initial transient period to discard (s), by default 60.0.
    random_seed : int | None, optional
        Seed for reproducibility, by default None.

    Returns
    -------
    np.ndarray
        Array of theta (rad) over time after transient removal.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_steps = int(t_final / dt)
    n_transient = int(t_transient / dt)

    C = compute_damping_coefficient(inertia, stiffness, damping_ratio)

    # State variables: theta (angle), omega (angular velocity)
    theta = 0.0
    omega = 0.0

    thetas = np.empty(n_steps, dtype=float)

    for i in range(n_steps):
        # Gaussian torque noise
        tau = np.random.normal(loc=0.0, scale=torque_std_Nm)

        # theta¨ = (tau - C*theta˙ - K*theta) / I
        theta_ddot = (tau - C * omega - stiffness * theta) / inertia

        # simple explicit integration (Euler)
        omega += theta_ddot * dt
        theta += omega * dt

        thetas[i] = theta

    # Discard transient
    if n_transient >= n_steps:
        return thetas * 0.0  # edge case, but should not occur in practice
    return thetas[n_transient:]


def compute_rms_and_95_range(theta_rad: np.ndarray) -> Tuple[float, float]:
    """Return RMS (deg) and 95% range (deg) of theta time series."""
    if theta_rad.size == 0:
        return 0.0, 0.0
    theta_deg = np.degrees(theta_rad)
    rms = np.sqrt(np.mean(theta_deg ** 2))
    lower, upper = np.percentile(theta_deg, [2.5, 97.5])
    range_95 = upper - lower
    return rms, range_95


def run_all_dynamic_simulations(
    species_params: Dict[str, SpeciesParams],
    torque_std_Nm: float = 1.0,
    dt: float = 0.001,
    t_final: float = 600.0,
    t_transient: float = 60.0,
    random_seed: int | None = 123,
) -> Dict[str, Dict[str, float]]:
    """Run passive dynamic simulations for all species.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping from species name to metrics:
        {
            'RMS_deg': ...,
            'range95_deg': ...,
            'stiffness_Nm_per_rad': ...,
            'aROM_total_deg': ...,
        }
    """
    results: Dict[str, Dict[str, float]] = {}

    for name, sp in species_params.items():
        if sp.stiffness_Nm_per_rad is None:
            raise ValueError(f"Stiffness not assigned for species {name}.")
        theta_series = simulate_passive_dynamics(
            inertia=sp.inertia_trunk_kgm2,
            stiffness=sp.stiffness_Nm_per_rad,
            damping_ratio=sp.damping_ratio,
            torque_std_Nm=torque_std_Nm,
            dt=dt,
            t_final=t_final,
            t_transient=t_transient,
            random_seed=random_seed,
        )
        rms_deg, range95_deg = compute_rms_and_95_range(theta_series)
        results[name] = {
            "RMS_deg": rms_deg,
            "range95_deg": range95_deg,
            "stiffness_Nm_per_rad": sp.stiffness_Nm_per_rad,
            "aROM_total_deg": sp.aROM_total_deg,
        }

    return results


# ================================================================
# Stage 3 Supplement: Morphological substitution
# ================================================================

def run_morphological_substitution(
    base_species: SpeciesParams,
    alt_species: SpeciesParams,
    torque_std_Nm: float = 1.0,
    dt: float = 0.001,
    t_final: float = 600.0,
    t_transient: float = 60.0,
    random_seed: int | None = 123,
) -> Dict[str, float]:
    """Simulate a "hybrid" configuration: base inertia + alternative hip stiffness.

    This corresponds to the manuscript's substitution tests, e.g.:

        Human trunk inertia + Equine hip morphology
        Human trunk inertia + Orangutan hip morphology

    Parameters
    ----------
    base_species : SpeciesParams
        Species providing inertia (e.g., Human).
    alt_species : SpeciesParams
        Species providing hip stiffness (e.g., Horse).
    torque_std_Nm : float, optional
        Torque noise standard deviation, by default 1.0.
    dt : float, optional
        Time step, by default 0.001.
    t_final : float, optional
        Total simulation time in seconds, by default 600.0.
    t_transient : float, optional
        Transient time to discard, by default 60.0.
    random_seed : int | None, optional
        Random seed, by default 123.

    Returns
    -------
    Dict[str, float]
        Metrics for the hybrid configuration.
    """
    if alt_species.stiffness_Nm_per_rad is None:
        raise ValueError(f"Alt species stiffness not assigned: {alt_species.name}")

    theta_series = simulate_passive_dynamics(
        inertia=base_species.inertia_trunk_kgm2,
        stiffness=alt_species.stiffness_Nm_per_rad,
        damping_ratio=base_species.damping_ratio,
        torque_std_Nm=torque_std_Nm,
        dt=dt,
        t_final=t_final,
        t_transient=t_transient,
        random_seed=random_seed,
    )
    rms_deg, range95_deg = compute_rms_and_95_range(theta_series)
    return {
        "base_inertia_species": base_species.name,
        "alt_morph_species": alt_species.name,
        "RMS_deg": rms_deg,
        "range95_deg": range95_deg,
        "stiffness_Nm_per_rad": alt_species.stiffness_Nm_per_rad,
    }


# ================================================================
# Utility: pretty-print tables
# ================================================================

def print_species_table(species_params: Dict[str, SpeciesParams]) -> None:
    """Print a simple table of species parameters."""
    header = (
        f"{'Species':<10} {'aROM_int(deg)':>14} {'aROM_ext(deg)':>14} "
        f"{'aROM_total(deg)':>16} {'K (Nm/rad)':>12} {'Inertia':>10}"
    )
    print(header)
    print("-" * len(header))
    for sp in species_params.values():
        k = sp.stiffness_Nm_per_rad if sp.stiffness_Nm_per_rad is not None else float('nan')
        print(
            f"{sp.name:<10} {sp.aROM_internal_deg:14.2f} {sp.aROM_external_deg:14.2f} "
            f"{sp.aROM_total_deg:16.2f} {k:12.2f} {sp.inertia_trunk_kgm2:10.2f}"
        )


def print_dynamic_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print RMS and 95% range for each species."""
    header = (
        f"{'Species':<10} {'RMS(deg)':>10} {'95%range(deg)':>15} "
        f"{'K (Nm/rad)':>12} {'aROM_total(deg)':>16}"
    )
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<10} {metrics['RMS_deg']:10.3f} {metrics['range95_deg']:15.3f} "
            f"{metrics['stiffness_Nm_per_rad']:12.2f} {metrics['aROM_total_deg']:16.2f}"
        )


def print_substitution_results_table(sub_results: List[Dict[str, float]]) -> None:
    """Print a table of morphological substitution results."""
    header = (
        f"{'Base inertia':<14} {'Alt morph':<12} {'RMS(deg)':>10} "
        f"{'95%range(deg)':>15} {'K (Nm/rad)':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in sub_results:
        print(
            f"{r['base_inertia_species']:<14} {r['alt_morph_species']:<12} "
            f"{r['RMS_deg']:10.3f} {r['range95_deg']:15.3f} {r['stiffness_Nm_per_rad']:12.2f}"
        )


# ================================================================
# Main entry point
# ================================================================

def main() -> None:
    # Stage 2: map geometric ROM to stiffness for each species
    assign_stiffness_to_all_species(SPECIES_PARAMS)

    print("\n=== Species parameters and stiffness mapping ===\n")
    print_species_table(SPECIES_PARAMS)

    # Stage 3: run passive dynamic simulations
    print("\n=== Passive dynamic simulations (RMS & 95% range) ===\n")
    dyn_results = run_all_dynamic_simulations(
        SPECIES_PARAMS,
        torque_std_Nm=1.0,
        dt=0.001,
        t_final=600.0,
        t_transient=60.0,
        random_seed=123,
    )
    print_dynamic_results_table(dyn_results)

    # Stage 3 Supplement: morphological substitution
    print("\n=== Morphological substitution experiments ===\n")

    human = SPECIES_PARAMS["Human"]
    horse = SPECIES_PARAMS["Horse"]
    orangutan = SPECIES_PARAMS["Orangutan"]
    dog = SPECIES_PARAMS["Dog"]

    sub_results: List[Dict[str, float]] = []

    # Human trunk + Equine hip
    sub_results.append(run_morphological_substitution(human, horse))

    # Human trunk + Orangutan hip
    sub_results.append(run_morphological_substitution(human, orangutan))

    # (Optionally) Human trunk + Dog hip
    sub_results.append(run_morphological_substitution(human, dog))

    print_substitution_results_table(sub_results)


if __name__ == "__main__":
    main()
