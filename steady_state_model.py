from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.integrate import solve_ivp, trapezoid


def _parse_nhe(nhe: Any) -> float:
    if isinstance(nhe, str):
        key = nhe.strip().lower()
        if key in {"yes", "y", "true", "1", "on"}:
            return 1.0
        if key in {"no", "n", "false", "0", "off"}:
            return 0.0
        raise ValueError(f"Unrecognized NHE value: {nhe}")
    if isinstance(nhe, (bool, np.bool_)):
        return 1.0 if nhe else 0.0
    return float(nhe)


@dataclass
class SimulationResult:
    x_um: np.ndarray
    Uend_M: np.ndarray
    ATP_avg_Mps: float
    converged: bool
    success: bool

    def profiles(self):
        U = self.Uend_M
        return {
            "x_um": self.x_um,
            "O2_mM": 1000 * U[:, 0],
            "CO2_mM": 1000 * U[:, 1],
            "HCO3e_mM": 1000 * U[:, 2],
            "pHe": -np.log10(np.clip(U[:, 3], 1e-30, None)),
            "Lace_mM": 1000 * U[:, 4],
            "HLac_mM": 1000 * U[:, 5],
            "Glu_mM": 1000 * U[:, 6],
            "HCO3i_mM": 1000 * U[:, 7],
            "pHi": -np.log10(np.clip(U[:, 8], 1e-30, None)),
            "Laci_mM": 1000 * U[:, 9],
            "ATP_avg_Mps": self.ATP_avg_Mps,
            "converged": self.converged,
            "solver_success": self.success,
        }


class DiffusionModel:

    def __init__(
        self,
        R, RR, GR, ve,
        startO2, startCO2, startHCO3, startGlucose,
        NHE,
        n_points=100,
        CA=300
    ):
        self.R = R
        self.RR = RR
        self.GR = GR
        self.ve = ve
        self.vi = 1 - ve
        self.NHE = _parse_nhe(NHE)
        self.CA = CA

        self.x = np.linspace(0, R, n_points)
        self.dx = self.x[1] - self.x[0]

        self.Km = 1e-6
        self.Kg = 1e-3
        self.Href = 10**-7.2
        self.Knhe = 10**-6.5

        self.kh = 0.14
        self.kr = self.kh / (10**-6.1)
        self.kf = 1e6
        self.kb = self.kf / (10**-3.9)

        self.c_blood = np.array([
            startO2/1000,
            startCO2/1000,
            startHCO3/1000,
            (startCO2/1000)*(10**-6.1)/(startHCO3/1000),
            0, 0, startGlucose/1000
        ])

    def JR(self):
        return (self.RR / 1000) / 60

    def JG(self):
        return (self.GR / 1000) / 60

    def initial(self):
        base = np.concatenate([self.c_blood, [0, 10**-7.2, 0]])
        return np.tile(base, (len(self.x), 1)).ravel()

    def rhs(self, t, y):
        U = y.reshape(len(self.x), 10)

        O2, CO2, HCO3e, He, Lace, HLac, Glu, HCO3i, Hi, Laci = U.T

        r_CO2e = self.CA * (self.kr * HCO3e * He - self.kh * CO2)
        r_CO2i = self.CA * (self.kr * HCO3i * Hi - self.kh * CO2)
        r_HLace = self.kb * Lace * He - self.kf * HLac
        r_HLaci = self.kb * Laci * Hi - self.kf * HLac

        JR = self.JR() * Glu/(Glu+self.Kg) * O2/(O2+self.Km)
        JG = self.JG() * Glu/(Glu+self.Kg)

        Jnhe = (self.NHE/1000/60) * (
            (Hi**2/(Hi**2+self.Knhe**2)) -
            (self.Href**2/(self.Href**2+self.Knhe**2))
        )

        s = np.zeros_like(U)

        s[:,0] = -6*self.vi*JR
        s[:,1] = 6*self.vi*JR + self.ve*r_CO2e + self.vi*r_CO2i
        s[:,2] = -self.ve*r_CO2e
        s[:,3] = self.ve*(-r_CO2e - r_HLace + Jnhe)
        s[:,4] = -self.ve*r_HLace
        s[:,5] = 2*self.vi*JG + self.ve*r_HLace + self.vi*r_HLaci
        s[:,6] = -self.vi*(JG + JR)
        s[:,7] = -self.vi*r_CO2i
        s[:,8] = self.vi*(-r_CO2i - r_HLaci - Jnhe)
        s[:,9] = -self.vi*r_HLaci

        return s.ravel()

    def solve(self):

        def event(t, y):
            return np.max(np.abs(self.rhs(t, y))) - 1e-9

        event.terminal = True
        event.direction = -1

        sol = solve_ivp(
            self.rhs,
            (0, 3*3600),
            self.initial(),
            method="BDF",
            rtol=1e-5,
            atol=1e-8,
            max_step=300,
            events=event
        )

        U = sol.y[:, -1].reshape(len(self.x), 10)

        O2 = U[:,0]
        Glu = U[:,6]

        JR = self.JR() * Glu/(Glu+self.Kg) * O2/(O2+self.Km)
        JG = self.JG() * Glu/(Glu+self.Kg)

        ATP = (3/self.R**3) * trapezoid((2*JG + 30*JR)*(self.x**2), self.x)

        return SimulationResult(self.x, U, ATP, True, sol.success)


def diffusion_pdepe_profiles_python(**kwargs):
    model = DiffusionModel(**kwargs)
    return model.solve().profiles()