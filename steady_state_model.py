from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.integrate import solve_ivp, trapezoid


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
    success: bool

    def profiles(self):
        U = self.Uend_M
        return {
            "x_um": self.x_um,
            "O2_mM": 1000*U[:,0],
            "CO2_mM": 1000*U[:,1],
            "HCO3e_mM": 1000*U[:,2],
            "pHe": -np.log10(np.clip(U[:,3],1e-30,None)),
            "Lace_mM": 1000*U[:,4],
            "HLac_mM": 1000*U[:,5],
            "Glu_mM": 1000*U[:,6],
            "HCO3i_mM": 1000*U[:,7],
            "pHi": -np.log10(np.clip(U[:,8],1e-30,None)),
            "Laci_mM": 1000*U[:,9],
            "ATP_avg_Mps": self.ATP_avg_Mps,
            "solver_success": self.success
        }


class Model:

    def __init__(self,R,RR,GR,ve,startO2,startCO2,startHCO3,startGlucose,NHE,n_points=80):

        self.R=R
        self.RR=RR
        self.GR=GR
        self.ve=ve
        self.vi=1-ve
        self.NHE=_parse_nhe(NHE)
        self.CA=100  # corrected

        self.x=np.linspace(0,R,n_points)
        self.dx=self.x[1]-self.x[0]

        self.inv_r = np.zeros_like(self.x)
        self.inv_r[1:] = 2/self.x[1:]

        D_free=np.array([2600,2100,1300,10,1000,1000,960,0,0,0])
        self.D=np.concatenate([D_free[:2], D_free[2:]*self.ve])

        self.Km=1e-6
        self.Kg=1e-3
        self.Href=10**-7.2
        self.Knhe=10**-6.5

        self.kh=0.14
        self.kr=self.kh/(10**-6.1)
        self.kf=1e6
        self.kb=self.kf/(10**-3.9)

        self.JR_const = (RR/1000)/60
        self.JG_const = (GR/1000)/60

        startO2/=1000
        startCO2/=1000
        startHCO3/=1000
        startGlucose/=1000

        He_blood = startCO2*(10**-6.1)/startHCO3

        self.c_blood=np.array([
            startO2,startCO2,startHCO3,He_blood,0,0,startGlucose
        ])

    def initial(self):
        HCO3i = self.c_blood[1] * 10**(7.2 - 6.1)
        base = np.concatenate([self.c_blood,[HCO3i,10**-7.2,0]])
        U = np.tile(base,(len(self.x),1))
        U[-1,:7] = self.c_blood
        return U.ravel()

    def rhs(self,t,y):

        n = len(self.x)
        U=y.reshape(n,10)
        U[-1,:7]=self.c_blood

        O2,CO2,HCO3e,He,Lace,HLac,Glu,HCO3i,Hi,Laci=U.T

        r_CO2e=self.CA*(self.kr*HCO3e*He-self.kh*CO2)
        r_CO2i=self.CA*(self.kr*HCO3i*Hi-self.kh*CO2)
        r_HLace=self.kb*Lace*He-self.kf*HLac
        r_HLaci=self.kb*Laci*Hi-self.kf*HLac

        JR=self.JR_const*Glu/(Glu+self.Kg)*O2/(O2+self.Km)
        JG=self.JG_const*Glu/(Glu+self.Kg)

        Jnhe=(self.NHE/1000/60)*(
            (Hi**2/(Hi**2+self.Knhe**2)) -
            (self.Href**2/(self.Href**2+self.Knhe**2))
        )

        s=np.zeros_like(U)

        s[:,0]=-6*self.vi*JR
        s[:,1]=6*self.vi*JR+self.ve*r_CO2e+self.vi*r_CO2i
        s[:,2]=-self.ve*r_CO2e
        s[:,3]=self.ve*(-r_CO2e-r_HLace+Jnhe)
        s[:,4]=-self.ve*r_HLace
        s[:,5]=2*self.vi*JG+self.ve*r_HLace+self.vi*r_HLaci
        s[:,6]=-self.vi*(JG+JR)
        s[:,7]=-self.vi*r_CO2i
        s[:,8]=self.vi*(-r_CO2i-r_HLaci-Jnhe)
        s[:,9]=-self.vi*r_HLaci

        U_pad = np.pad(U, ((1,1),(0,0)), mode='edge')
        d2 = (U_pad[2:] - 2*U + U_pad[:-2]) / self.dx**2
        d1 = (U_pad[2:] - U_pad[:-2]) / (2*self.dx)

        dU = self.D * (d2 + self.inv_r[:,None]*d1)
        dU[0,:] = self.D * (3*(U[1,:]-U[0,:]) / self.dx**2)
        dU[-1,:] = 0

        c_vec=np.array([1,1,self.ve,self.ve,self.ve,1,1,self.vi,self.vi,self.vi])
        rhs=(s+dU)/c_vec
        rhs[-1,:7]=0

        return rhs.ravel()

    def solve(self):
        sol=solve_ivp(
            self.rhs,
            (0,2*3600),
            self.initial(),
            method="BDF",
            rtol=1e-4,
            atol=1e-7,
            max_step=300
        )

        U=sol.y[:,-1].reshape(len(self.x),10)

        O2=U[:,0]
        Glu=U[:,6]

        JR=self.JR_const*Glu/(Glu+self.Kg)*O2/(O2+self.Km)
        JG=self.JG_const*Glu/(Glu+self.Kg)

        ATP=(3/self.R**3)*trapezoid((2*JG+30*JR)*(self.x**2),self.x)

        return SimulationResult(self.x,U,ATP,sol.success)


def diffusion_pdepe_profiles_python(**kwargs):
    return Model(**kwargs).solve().profiles()
