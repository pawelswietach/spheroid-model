from __future__ import annotations
from scipy.integrate import trapezoid

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SimulationResult:
    x_um: np.ndarray
    Uend_M: np.ndarray
    Uave_M: np.ndarray
    ATP_avg_Mps: float
    converged: bool
    max_abs_dudt_final: float
    t_final_s: float
    success: bool
    message: str

    def profiles(self) -> dict[str, Any]:
        """Return profiles in the same units/layout as the MATLAB helper."""
        ve = self.metadata["ve"]
        vi = 1.0 - ve
        U = self.Uend_M
        return {
            "x_um": self.x_um,
            "O2_mM": 1000.0 * U[:, 0],
            "CO2_mM": 1000.0 * U[:, 1],
            "HCO3e_mM": 1000.0 * U[:, 2],
            "pHe": -np.log10(np.clip(U[:, 3], 1e-30, None)),
            "Lace_mM": 1000.0 * U[:, 4],
            "HLac_mM": 1000.0 * U[:, 5],
            "Glu_mM": 1000.0 * U[:, 6],
            "HCO3i_mM": 1000.0 * U[:, 7],
            "pHi": -np.log10(np.clip(U[:, 8], 1e-30, None)),
            "Laci_mM": 1000.0 * U[:, 9],
            "Carbon_total_mM": 1000.0 * (ve * U[:, 2] + vi * U[:, 7] + U[:, 1]),
            "Uave": self.Uave_M.copy(),
            "ATP_avg_Mps": self.ATP_avg_Mps,
            "converged": self.converged,
            "max_abs_dudt_final": self.max_abs_dudt_final,
            "t_final_s": self.t_final_s,
            "success": self.success,
            "message": self.message,
        }

    metadata: dict[str, Any]


def _parse_nhe(nhe: Any) -> float:
    """
    Convert user input into the MATLAB-compatible scalar multiplier.

    MATLAB uses NHE numerically inside:
        Jnhe = (NHE / 1000 / 60) * (...)

    So:
      - yes/true/on -> 1.0
      - no/false/off -> 0.0
      - numeric input is passed through as float
    """
        
    if isinstance(nhe, str):
        key = nhe.strip().lower()
        if key in {"yes", "y", "true", "1", "on"}:
            return 1.0
        if key in {"no", "n", "false", "0", "off"}:
            return 0.0
        raise ValueError(f"Unrecognized NHE value: {nhe!r}")

    if isinstance(nhe, (bool, np.bool_)):
        return 1.0 if nhe else 0.0

    return float(nhe)


class DiffusionSteadyStateModel:
    """
    Python translation of the MATLAB pdepe model.

    Important note:
    The original MATLAB routine integrates from t=0 to 5 hours and returns the
    final state as the practical steady state. This class reproduces that logic
    with a method-of-lines discretization and SciPy's stiff ODE solver.
    """

    def __init__(self,
        R: float,
        RR: float,
        GR: float,
        ve: float,
        startO2: float,
        startCO2: float,
        startHCO3: float,
        startGlucose: float,
        NHE: Any,
        CA: float = 100.0,
        n_points: int = 500,
    ) -> None:
        if R <= 0:
            raise ValueError("R must be positive.")
        if not (0 < ve < 1):
            raise ValueError("ve must be between 0 and 1.")
        if n_points < 5:
            raise ValueError("n_points must be at least 5.")

        self.R = float(R)
        self.RR = float(RR)
        self.GR = float(GR)
        self.ve = float(ve)
        self.vi = 1.0 - self.ve
        self.startO2 = float(startO2)
        self.startCO2 = float(startCO2)
        self.startHCO3 = float(startHCO3)
        self.startGlucose = float(startGlucose)
        self.NHE = _parse_nhe(NHE)
        self.n_points = int(n_points)

        self.m = 2.0

        D_free = np.array([2600.0, 2100.0, 1300.0, 10.0, 1000.0, 1000.0, 960.0, 0.0, 0.0, 0.0])
        self.D = np.concatenate([D_free[:2], D_free[2:] * self.ve])

        self.CA = float(CA)
        self.kh = 0.14
        self.kr = self.kh / (10.0 ** -6.1)
        self.kf = 1.0e6
        self.kb = self.kf / (10.0 ** -3.9)

        self.Km = 1.0e-6
        self.Kg = 1.0e-3
        self.Href = 10.0 ** -7.2
        self.Knhe = 10.0 ** -6.5

        self.x = np.linspace(0.0, self.R, self.n_points)
        self.dx = self.x[1] - self.x[0]

        startO2_M = self.startO2 / 1000.0
        startCO2_M = self.startCO2 / 1000.0
        startHCO3_M = self.startHCO3 / 1000.0
        startGlucose_M = self.startGlucose / 1000.0

        He_blood = startCO2_M * (10.0 ** -6.1) / startHCO3_M
        self.c_blood = np.array([startO2_M, startCO2_M, startHCO3_M, He_blood, 0.0, 0.0, startGlucose_M])

        b_pHi = 7.2
        self.u0_one_node = np.array(
            [
                *self.c_blood,
                startCO2_M * (10.0 ** (b_pHi - 6.1)),
                10.0 ** (-b_pHi),
                0.0,
            ],
            dtype=float,
        )

    def JR(self, r: np.ndarray) -> np.ndarray:
        return np.full_like(r, (self.RR / 1000.0) / 60.0, dtype=float)

    def JG(self, r: np.ndarray) -> np.ndarray:
        return np.full_like(r, (self.GR / 1000.0) / 60.0, dtype=float)

    def initial_state(self) -> np.ndarray:
        U0 = np.tile(self.u0_one_node, (self.n_points, 1))
        U0[-1, :7] = self.c_blood
        return U0.ravel()

    def _reaction_terms(self, U: np.ndarray) -> np.ndarray:
        O2 = U[:, 0]
        CO2 = U[:, 1]
        HCO3e = U[:, 2]
        He = U[:, 3]
        Lace = U[:, 4]
        HLac = U[:, 5]
        Glu = U[:, 6]
        HCO3i = U[:, 7]
        Hi = U[:, 8]
        Laci = U[:, 9]

        r_CO2e = self.CA * (self.kr * HCO3e * He - self.kh * CO2)
        r_CO2i = self.CA * (self.kr * HCO3i * Hi - self.kh * CO2)
        r_HLace = self.kb * Lace * He - self.kf * HLac
        r_HLaci = self.kb * Laci * Hi - self.kf * HLac

        JRval = self.JR(self.x) * Glu / (Glu + self.Kg) * O2 / (O2 + self.Km)
        JGval = self.JG(self.x) * Glu / (Glu + self.Kg)

        nhe_activation = Hi**2 / (Hi**2 + self.Knhe**2)
        nhe_reference = self.Href**2 / (self.Href**2 + self.Knhe**2)
        Jnhe = (self.NHE / 1000.0 / 60.0) * (nhe_activation - nhe_reference)

        s = np.zeros_like(U)
        s[:, 0] = -6.0 * self.vi * JRval
        s[:, 1] = 6.0 * self.vi * JRval + self.ve * r_CO2e + self.vi * r_CO2i
        s[:, 2] = self.ve * (-r_CO2e)
        s[:, 3] = self.ve * (-r_CO2e - r_HLace + Jnhe)
        s[:, 4] = self.ve * (-r_HLace)
        s[:, 5] = 2.0 * self.vi * JGval + self.ve * r_HLace + self.vi * r_HLaci
        s[:, 6] = -self.vi * (JGval + JRval)
        s[:, 7] = self.vi * (-r_CO2i)
        s[:, 8] = self.vi * (-r_CO2i - r_HLaci - Jnhe)
        s[:, 9] = self.vi * (-r_HLaci)
        return s

    def _diffusion_term_one_species(self, u: np.ndarray, D: float, dirichlet_value: float | None) -> np.ndarray:
        term = np.zeros_like(u)
        if D == 0.0:
            return term

        dx = self.dx

        # r = 0 symmetry in spherical geometry: Laplacian(u)(0) = 6 * (u1 - u0) / dx^2
        term[0] = D * (6.0 * (u[1] - u[0]) / dx**2)

        # interior nodes
        r = self.x[1:-1]
        term[1:-1] = D * (
            (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
            + (2.0 / r) * (u[2:] - u[:-2]) / (2.0 * dx)
        )

        # outer boundary
        if dirichlet_value is None:
            # zero-flux at outer edge via mirrored ghost point
            u_ghost = u[-2]
            urr = (u_ghost - 2.0 * u[-1] + u[-2]) / dx**2
            ur = 0.0
            term[-1] = D * (urr + (2.0 / self.x[-1]) * ur)
        else:
            # keep boundary node pinned to the blood value
            term[-1] = 0.0

        return term

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        U = y.reshape(self.n_points, 10).copy()

        # Enforce the MATLAB right boundary exactly for the extracellular species.
        U[-1, :7] = self.c_blood

        s = self._reaction_terms(U)
        rhs = np.zeros_like(U)

        c_vec = np.array([1.0, 1.0, self.ve, self.ve, self.ve, 1.0, 1.0, self.vi, self.vi, self.vi])

        for j in range(10):
            dirichlet_value = self.c_blood[j] if j < 7 else None
            diff_term = self._diffusion_term_one_species(U[:, j], self.D[j], dirichlet_value)
            rhs[:, j] = (diff_term + s[:, j]) / c_vec[j]

        # Pin extracellular boundary values at r = R.
        rhs[-1, :7] = 0.0

        return rhs.ravel()

    def solve(
        self,
        t_final_s: float = 5.0 * 3600.0,
        steady_tol: float = 1.0e-10,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-10,
        max_step: float = 120.0,
    ) -> SimulationResult:
        y0 = self.initial_state()

        sol = solve_ivp(
            self.rhs,
            t_span=(0.0, float(t_final_s)),
            y0=y0,
            method="BDF",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        y_end = sol.y[:, -1]
        Uend = y_end.reshape(self.n_points, 10).copy()
        Uend[-1, :7] = self.c_blood

        dudt_final = self.rhs(sol.t[-1], y_end).reshape(self.n_points, 10)
        max_abs_dudt_final = float(np.max(np.abs(dudt_final)))
        converged = bool(max_abs_dudt_final < steady_tol)

        r = self.x[:, None]
        Uave = (3.0 / self.R**3) * trapezoid(Uend * (r**2), self.x, axis=0)

        O2 = Uend[:, 0]
        Glu = Uend[:, 6]
        Hi = Uend[:, 8]
        JRv = self.JR(self.x) * Glu / (Glu + self.Kg) * O2 / (O2 + self.Km)
        JGv = self.JG(self.x) * Glu / (Glu + self.Kg) * ((10.0 ** -7.1) ** 2.25) / (Hi**2.25 + (10.0 ** -7.1) ** 2.25)
        ATP_avg_Mps = float((3.0 / self.R**3) * trapezoid((2.0 * JGv + 30.0 * JRv) * (self.x**2), self.x))

        return SimulationResult(
            x_um=self.x.copy(),
            Uend_M=Uend,
            Uave_M=Uave,
            ATP_avg_Mps=ATP_avg_Mps,
            converged=converged,
            max_abs_dudt_final=max_abs_dudt_final,
            t_final_s=float(sol.t[-1]),
            success=bool(sol.success),
            message=str(sol.message),
            metadata={
                "R": self.R,
                "RR": self.RR,
                "GR": self.GR,
                "ve": self.ve,
                "startO2": self.startO2,
                "startCO2": self.startCO2,
                "startHCO3": self.startHCO3,
                "startGlucose": self.startGlucose,
                "NHE": self.NHE,
                "n_points": self.n_points,
                "CA": self.CA,
            },
        )


def diffusion_pdepe_profiles_python(
    R: float,
    RR: float,
    GR: float,
    ve: float,
    startO2: float,
    startCO2: float,
    startHCO3: float,
    startGlucose: float,
    NHE: Any,
    CA: float = 100.0,
    n_points: int = 500,
    t_final_s: float = 5.0 * 3600.0,
) -> dict[str, Any]:
    """Drop-in convenience wrapper returning MATLAB-like named outputs."""
    model = DiffusionSteadyStateModel(
        R=R,
        RR=RR,
        GR=GR,
        ve=ve,
        startO2=startO2,
        startCO2=startCO2,
        startHCO3=startHCO3,
        startGlucose=startGlucose,
        NHE=NHE,
        CA=CA,
        n_points=n_points,
    )
    return model.solve(t_final_s=t_final_s).profiles()