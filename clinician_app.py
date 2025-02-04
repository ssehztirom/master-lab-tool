import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
import torch
from oauth2client.service_account import ServiceAccountCredentials
from scipy.stats import lognorm

# Define the new ODE system
def ode_system(y, t, params):
    """
    Simplified ODE system for 5 biomarkers (C, H, B, A, I).

    Args:
        y (tensor): Current values of biomarkers [C, H, B, A, I].
        t (float): Time variable (unused here, required for solvers).
        params (list): List of parameters.

    Returns:
        tensor: Derivatives [dC/dt, dH/dt, dB/dt, dA/dt, dI/dt].
    """ 
    # Unpack current values
    C, H, B, A, I = y

    # Baselines (fixed reference values)
    C0, H0, B0, A0, I0 = 1.0, 14.0, 18.0, 4.0, 50.0
    K_C = 200  # CRP capacity

    # Unpack parameters
    kC0, kC1, kC2 = params[0], params[1], params[2]
    kH1, kH2, kH3 = params[3], params[4], params[5]
    kB0, kB1, kB2, kB3 = params[6], params[7], params[8], params[9]
    kA0, kA1, kA2 = params[10], params[11], params[12]
    kI0, kI1, kI2, kI3 = params[13], params[14], params[15], params[16]

    # Define the differential equations
    dCdt = (kC0 + kC1 * (1 - A / A0)) * C * torch.relu(1 - C / K_C) + kC2 * (1 - B / B0) 
    dHdt = -kH1 * (C - C0) - kH2 * (1 - I / I0) + kH3 * (H0 - H)
    dBdt = kB0 * (B0 - B) - kB1 * (C - C0) + kB2 * (H0 - H) - kB3 * (1 - A / A0)
    dAdt = kA0 * (A0 - A) - kA1 * (C - C0) - kA2 * (1 - B / B0)
    dIdt = kI0 * (I0 - I) - kI1 * (C - C0) + kI2 * (1 - B / B0) + kI3 * (1 - A / A0)
    
    return torch.stack([dCdt, dHdt, dBdt, dAdt, dIdt])

# Streamlit App Title and Description
st.title("RDEB Patient Progression Simulation")

st.markdown("""
### Understanding the ODE System
This simulation models the progression of biomarkers in RDEB patients using a simplified ODE system. 

Each biomarker's change over time is influenced by different factors:
- **CRP (C):** Growth depends on baseline levels, albumin levels, and BMI.
- **Hemoglobin (H):** Affected by CRP, iron levels, and a self-regulating mechanism.
- **BMI (B):** Influenced by CRP, hemoglobin, and albumin.
- **Albumin (A):** Changes based on CRP and BMI levels.
- **Iron (I):** Depends on CRP, BMI, and albumin.

This model helps us understand interactions among biomarkers and how they evolve over time.
""")

# Display ODE function structure
st.code('''
def ode_system(y, t, params):
    C, H, B, A, I = y  # Biomarker values
    C0, H0, B0, A0, I0 = 1.0, 14.0, 18.0, 4.0, 50.0  # Baselines
    K_C = 200  # CRP capacity

    # Extract parameters
    kC0, kC1, kC2 = params[0], params[1], params[2]
    kH1, kH2, kH3 = params[3], params[4], params[5]
    kB0, kB1, kB2, kB3 = params[6], params[7], params[8], params[9]
    kA0, kA1, kA2 = params[10], params[11], params[12]
    kI0, kI1, kI2, kI3 = params[13], params[14], params[15], params[16]

    # Differential Equations
    dCdt = (kC0 + kC1 * (1 - A / A0)) * C * torch.relu(1 - C / K_C) + kC2 * (1 - B / B0)
    dHdt = -kH1 * (C - C0) - kH2 * (1 - I / I0) + kH3 * (H0 - H)
    dBdt = kB0 * (B0 - B) - kB1 * (C - C0) + kB2 * (H0 - H) - kB3 * (1 - A / A0)
    dAdt = kA0 * (A0 - A) - kA1 * (C - C0) - kA2 * (1 - B / B0)
    dIdt = kI0 * (I0 - I) - kI1 * (C - C0) + kI2 * (1 - B / B0) + kI3 * (1 - A / A0)

    return torch.stack([dCdt, dHdt, dBdt, dAdt, dIdt])
''', language='python')

st.markdown("""
Each equation describes how a biomarker's level changes over time. The parameters (k-values) define how strongly one biomarker influences another. 
You can adjust these parameters in the sidebar to see how different values affect patient progression.
""")

# The rest of your app, including parameter inputs, simulation execution, and visualization, should follow below.
