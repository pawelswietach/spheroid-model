
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from steady_state_solver_production import diffusion_steady_state

st.set_page_config(layout="wide")

st.title("Cancer spheroid steady-state diffusion model")

st.sidebar.header("Inputs")

R = st.sidebar.number_input("Spheroid radius (um):", value=100.0)
RR = st.sidebar.number_input("Respiratory rate (mM/min):", value=1.0)
GR = st.sidebar.number_input("Glycolytic rate (mM/min):", value=1.0)
ve = st.sidebar.number_input("Extracellular volume fraction", value=0.2)

startO2 = st.sidebar.number_input("Bath O2 (mM)", value=0.13)
startCO2 = st.sidebar.number_input("Bath CO2 (mM)", value=1.2)
startHCO3 = st.sidebar.number_input("Bath HCO3 (mM)", value=24.0)
startGlucose = st.sidebar.number_input("Bath Glucose (mM)", value=5.0)

NHE = st.sidebar.radio("NHE", ["yes", "no"])
n_points = st.sidebar.number_input("Radial mesh points", value=80)

if st.button("Solve"):

    out = diffusion_steady_state(
        R=R, RR=RR, GR=GR, ve=ve,
        startO2=startO2,
        startCO2=startCO2,
        startHCO3=startHCO3,
        startGlucose=startGlucose,
        NHE=NHE,
        n_points=int(n_points)
    )

    st.write(f"Solver success: {out.get('success')}")

    x = out["x_um"]
    depth = R - x

    df = pd.DataFrame({
        "Radial depth (um)": depth,
        "O2": out["O2_mM"],
        "Glucose": out["Glu_mM"],
        "CO2": out["CO2_mM"],
        "Lactic acid": out["HLac_mM"],
        "HCO3e": out["HCO3e_mM"],
        "HCO3i": out["HCO3i_mM"],
        "pHe": out["pHe"],
        "pHi": out["pHi"],
        "Lace": out["Lace_mM"],
        "Laci": out["Laci_mM"],
    })

    fig, axs = plt.subplots(2, 4, figsize=(18, 10))

    axs[0,0].plot(depth, df["O2"])
    axs[0,0].set_title("O2")

    axs[0,1].plot(depth, df["Glucose"])
    axs[0,1].set_title("Glucose")

    axs[0,2].plot(depth, df["CO2"])
    axs[0,2].set_title("CO2")

    axs[0,3].plot(depth, df["Lactic acid"])
    axs[0,3].set_title("Lactic acid")

    for ax in axs.flat:
        ax.set_xlabel("Radial depth (um)")

    st.pyplot(fig)
