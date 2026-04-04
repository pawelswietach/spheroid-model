
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from steady_state_model import diffusion_solver

st.set_page_config(layout="wide")

st.image("image.png", use_container_width=True)

st.title("Cancer spheroid diffusion model")

R = st.sidebar.number_input("Radius", value=100.0)
RR = st.sidebar.number_input("Resp rate", value=1.0)
GR = st.sidebar.number_input("Glycolysis", value=1.0)
ve = st.sidebar.number_input("ve", value=0.2)

startO2 = st.sidebar.number_input("O2", value=0.13)
startCO2 = st.sidebar.number_input("CO2", value=1.2)
startHCO3 = st.sidebar.number_input("HCO3", value=24.0)
startGlucose = st.sidebar.number_input("Glucose", value=5.0)

NHE = st.sidebar.radio("NHE", ["yes","no"])

if st.button("Solve"):

    out = diffusion_solver(
        R=R,RR=RR,GR=GR,ve=ve,
        startO2=startO2,startCO2=startCO2,
        startHCO3=startHCO3,startGlucose=startGlucose,
        NHE=NHE
    )

    x = out["x_um"]
    depth = R - x

    fig, axs = plt.subplots(2,4,figsize=(18,10))

    axs[0,0].plot(depth,out["O2_mM"]); axs[0,0].set_title("O2")
    axs[0,1].plot(depth,out["Glu_mM"]); axs[0,1].set_title("Glucose")
    axs[0,2].plot(depth,out["CO2_mM"]); axs[0,2].set_title("CO2")
    axs[0,3].plot(depth,out["HLac_mM"]); axs[0,3].set_title("Lactic acid")

    axs[1,0].plot(depth,out["HCO3e_mM"],'r'); axs[1,0].plot(depth,out["HCO3i_mM"],'b')
    axs[1,0].set_title("Bicarbonate")

    axs[1,1].plot(depth,out["pHe"],'r'); axs[1,1].plot(depth,out["pHi"],'b')
    axs[1,1].set_title("pH")

    axs[1,2].plot(depth,out["Lace_mM"],'r'); axs[1,2].plot(depth,out["Laci_mM"],'b')
    axs[1,2].set_title("Lactate")

    axs[1,3].plot(out["O2_mM"],out["pHe"]); axs[1,3].plot(out["O2_mM"],out["pHi"])
    axs[1,3].set_title("pH vs O2")

    st.pyplot(fig)
