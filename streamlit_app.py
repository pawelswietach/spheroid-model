
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from steady_state_model import diffusion_pdepe_profiles_python

st.set_page_config(layout="wide")

st.image("image.png")

st.title("Cancer spheroid steady-state diffusion model")

st.markdown(
    '[Using Mathematical Modeling of Tumor Metabolism to Predict the Magnitude, Composition, and Hypoxic Interactions of Microenvironment Acidosis](https://onlinelibrary.wiley.com/doi/10.1002/bies.70101)'
)

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
n_points = st.sidebar.number_input("Radial mesh points", value=100)

@st.cache_data
def run_model(R, RR, GR, ve, startO2, startCO2, startHCO3, startGlucose, NHE, n_points):
    return diffusion_pdepe_profiles_python(
        R=R, RR=RR, GR=GR, ve=ve,
        startO2=startO2, startCO2=startCO2,
        startHCO3=startHCO3,
        startGlucose=startGlucose,
        NHE=NHE,
        n_points=int(n_points)
    )

if st.button("Solve"):

    with st.spinner("Solving steady state..."):
        out = run_model(R, RR, GR, ve, startO2, startCO2, startHCO3, startGlucose, NHE, n_points)

    st.write(f"Solver success: {out.get('solver_success')}")
    st.write(f"Converged: {out.get('converged')}")

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

    st.subheader("Profiles")

    fig, axs = plt.subplots(2, 4, figsize=(18, 10))

    axs[0,0].plot(depth, df["O2"], color="black")
    axs[0,1].plot(depth, df["Glucose"], color="black")
    axs[0,2].plot(depth, df["CO2"], color="black")
    axs[0,3].plot(depth, df["Lactic acid"], color="black")

    axs[1,0].plot(depth, df["HCO3e"], color="red", label="Extracellular")
    axs[1,0].plot(depth, df["HCO3i"], color="blue", label="Intracellular")
    axs[1,0].set_title("Bicarbonate")
    axs[1,0].legend()

    axs[1,1].plot(depth, df["pHe"], color="red", label="Extracellular")
    axs[1,1].plot(depth, df["pHi"], color="blue", label="Intracellular")
    axs[1,1].set_title("pH")
    axs[1,1].legend()

    axs[1,2].plot(depth, df["Lace"], color="red", label="Extracellular")
    axs[1,2].plot(depth, df["Laci"], color="blue", label="Intracellular")
    axs[1,2].set_title("Lactate")
    axs[1,2].legend()

    axs[1,3].plot(df["O2"], df["pHe"], color="blue", label="pHe")
    axs[1,3].plot(df["O2"], df["pHi"], color="red", label="pHi")
    axs[1,3].set_title("pH vs O2")
    axs[1,3].set_xlabel("O2 (mM)")
    axs[1,3].legend()

    for i, ax in enumerate(axs.flat):
        if i != 7:
            ax.set_xlabel("Radial depth (um)")

    plt.subplots_adjust(hspace=0.5)

    st.pyplot(fig, use_container_width=True)

    st.subheader("Spatial data")
    st.dataframe(df, use_container_width=True)
