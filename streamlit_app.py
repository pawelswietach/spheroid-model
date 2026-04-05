
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from steady_state_model import diffusion_solver

st.set_page_config(layout="wide")

col1, col2 = st.columns([1,1])
with col1:
    st.image("image.png")

st.title("Cancer spheroid diffusion-reaction model for pH and oxygen dynamics")

st.markdown('[Using Mathematical Modeling of Tumor Metabolism to Predict the Magnitude, Composition, and Hypoxic Interactions of Microenvironment Acidosis](https://onlinelibrary.wiley.com/doi/10.1002/bies.70101)')

st.sidebar.header("Inputs")

R = st.sidebar.number_input("Radius (um)", value=200.0)
RR = st.sidebar.number_input("Respiratory rate (mM/min)", value=1.0)
GR = st.sidebar.number_input("Fermentative rate (mM/min)", value=1.0)
ve = st.sidebar.number_input("Extracellular volume fraction", value=0.2)

startO2 = st.sidebar.number_input("Bath [O2] (mM)", value=0.13)
startCO2 = st.sidebar.number_input("Bath [CO2] (mM)", value=1.2)
startHCO3 = st.sidebar.number_input("Bath [HCO3-] (mM)", value=24.0)
startGlucose = st.sidebar.number_input("Bath [Glucose] (mM)", value=5.0)

CA = st.sidebar.number_input("Carbonic anhydrase activity", value=100.0)
pHi0 = st.sidebar.number_input("Initial intracellular pH", value=7.2)

NHE = st.sidebar.radio("NHE activity", ["yes","no"])
n_points = st.sidebar.number_input("Mesh points", value=20)

if st.button("Solve"):

    out = diffusion_solver(
        R=R,RR=RR,GR=GR,ve=ve,
        startO2=startO2,startCO2=startCO2,
        startHCO3=startHCO3,startGlucose=startGlucose,
        NHE=NHE,
        CA=CA,
        pHi0=pHi0,
        n_points=int(n_points)
    )

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

    fig, axs = plt.subplots(2,4,figsize=(18,10))

    axs[0,0].plot(depth,out["O2_mM"],'k'); axs[0,0].set_title("O2 (mM)")
    axs[0,1].plot(depth,out["Glu_mM"],'k'); axs[0,1].set_title("Glucose (mM)")
    axs[0,2].plot(depth,out["CO2_mM"],'k'); axs[0,2].set_title("CO2 (mM)")
    axs[0,3].plot(depth,1000*out["HLac_mM"],'k'); axs[0,3].set_title("Lactic acid (µM)")

    axs[1,0].plot(depth, out["HCO3e_mM"], 'r', label="Extracellular")
    axs[1,0].plot(depth, out["HCO3i_mM"], 'b', label="Intracellular")
    axs[1,0].set_title("Bicarbonate (mM)")
    axs[1,0].legend()

    axs[1,1].plot(depth, out["pHe"], 'r', label="Extracellular")
    axs[1,1].plot(depth, out["pHi"], 'b', label="Intracellular")
    axs[1,1].set_title("pH")
    axs[1,1].legend()

    axs[1,2].plot(depth, out["Lace_mM"], 'r', label="Extracellular")
    axs[1,2].plot(depth, out["Laci_mM"], 'b', label="Intracellular")
    axs[1,2].set_title("Lactate (mM)")
    axs[1,2].legend()

    axs[1,3].plot(out["O2_mM"], out["pHe"], 'r', label="Extracellular")
    axs[1,3].plot(out["O2_mM"], out["pHi"], 'b', label="Intracellular")
    axs[1,3].set_title("pH vs O2")
    axs[1,3].set_xlabel("O2 (mM)")
    axs[1,3].legend()

    for i, ax in enumerate(axs.flat):
        if i != 7:
            ax.set_xlabel("Radial depth (µm)")


    plt.subplots_adjust(hspace=0.5)

    st.pyplot(fig, use_container_width=True)

    st.subheader("Spatial data")
    st.dataframe(df, use_container_width=True)
