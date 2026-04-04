
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid

def _parse_nhe(nhe: Any) -> float:
    if isinstance(nhe, str):
        key = nhe.strip().lower()
        if key in {"yes","y","true","1","on"}:
            return 1.0
        if key in {"no","n","false","0","off"}:
            return 0.0
        raise ValueError(f"Unrecognized NHE value: {nhe}")
    if isinstance(nhe,(bool,np.bool_)):
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
        O2, CO2, He, HLac, Glu, Hi = U.T

        pHe = -np.log10(He)
        pHi = -np.log10(Hi)

        HCO3e = CO2 * 10**(pHe - 6.1)
        HCO3i = CO2 * 10**(pHi - 6.1)

        K_Lac = 10**(-3.9)
        Lace = HLac / (He * K_Lac)
        Laci = HLac / (Hi * K_Lac)

        return {
            "x_um": self.x_um,
            "O2_mM": 1000*O2,
            "CO2_mM": 1000*CO2,
            "HCO3e_mM": 1000*HCO3e,
            "HCO3i_mM": 1000*HCO3i,
            "pHe": pHe,
            "pHi": pHi,
            "Lace_mM": 1000*Lace,
            "Laci_mM": 1000*Laci,
            "HLac_mM": 1000*HLac,
            "Glu_mM": 1000*Glu,
            "ATP_avg_Mps": self.ATP_avg_Mps,
            "converged": self.converged,
            "solver_success": self.success
        }

class Model:

    def __init__(self,R,RR,GR,ve,startO2,startCO2,startHCO3,startGlucose,NHE,n_points=100):
        self.R=R
        self.RR=RR
        self.GR=GR
        self.ve=ve
        self.vi=1-ve
        self.NHE=_parse_nhe(NHE)
        self.x=np.linspace(0,R,n_points)
        self.dx=self.x[1]-self.x[0]

        self.Km=1e-6
        self.Kg=1e-3
        self.Href=10**-7.2
        self.Knhe=10**-6.5

        self.c_blood=np.array([
            startO2/1000,
            startCO2/1000,
            10**-7.4,
            0,
            startGlucose/1000,
            10**-7.2
        ])

    def JR(self,r): return (self.RR/1000)/60*np.ones_like(r)
    def JG(self,r): return (self.GR/1000)/60*np.ones_like(r)

    def initial(self):
        return np.tile(self.c_blood,(len(self.x),1)).ravel()

    def rhs(self,t,y):
        U=y.reshape(len(self.x),6)
        O2,CO2,He,HLac,Glu,Hi=U.T

        JR=self.JR(self.x)*Glu/(Glu+self.Kg)*O2/(O2+self.Km)
        JG=self.JG(self.x)*Glu/(Glu+self.Kg)

        nhe=(Hi**2/(Hi**2+self.Knhe**2)-self.Href**2/(self.Href**2+self.Knhe**2))
        Jnhe=(self.NHE/1000/60)*nhe

        d=np.zeros_like(U)

        d[:,0]=-6*self.vi*JR
        d[:,1]=6*self.vi*JR
        d[:,2]=self.ve*(Jnhe)
        d[:,3]=2*self.vi*JG
        d[:,4]=-self.vi*(JG+JR)
        d[:,5]=self.vi*(-Jnhe)

        return d.ravel()

    def solve(self):
        sol=solve_ivp(self.rhs,(0,3*3600),self.initial(),method="BDF")
        U=sol.y[:,-1].reshape(len(self.x),6)

        O2=U[:,0]; Glu=U[:,4]; Hi=U[:,5]
        JR=self.JR(self.x)*Glu/(Glu+self.Kg)*O2/(O2+self.Km)
        JG=self.JG(self.x)*Glu/(Glu+self.Kg)

        ATP=(3/self.R**3)*trapezoid((2*JG+30*JR)*(self.x**2),self.x)

        return SimulationResult(self.x,U,ATP,True,sol.success)

def diffusion_pdepe_profiles_python(*args,**kwargs):
    model=Model(*args,**kwargs)
    return model.solve().profiles()
