from steady_state_model_old import diffusion_pdepe_profiles_python as _time_solver

def diffusion_pdepe_profiles_python(
    R, RR, GR, ve,
    startO2, startCO2, startHCO3, startGlucose,
    NHE,
    n_points=100,
):
    return _time_solver(
        R=R,
        RR=RR,
        GR=GR,
        ve=ve,
        startO2=startO2,
        startCO2=startCO2,
        startHCO3=startHCO3,
        startGlucose=startGlucose,
        NHE=NHE,
        CA=300,
        n_points=n_points,
        t_final_s=3*3600
    )