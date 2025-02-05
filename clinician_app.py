import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##########################
# 1) Introduction Section
##########################
st.title("EB Patient ODE Model - Biomarker Simulation")

st.markdown("""
## How This Model Works

We simulate **5 biomarkers** in EB patients:
- **C** (CRP) - Inflammatory marker
- **H** (Hemoglobin) - Oxygen-carrying protein
- **B** (BMI) - Body Mass Index
- **A** (Albumin) - Protein linked to nutrition & liver function
- **I** (Iron) - Essential mineral for blood function

### Model Parameters
- **kC0**: Baseline CRP production rate.
- **kC1**: CRP increase when albumin is low.
- **kC2**: CRP increase when BMI is low.
- **delta**: Extra CRP boost for severe patients (Intermediate = 0, Severe > 0).
- **Other k-values**: Control growth/decay rates for Hemoglobin, BMI, Albumin, and Iron.
""")

##########################
# 2) Sidebar Inputs
##########################
st.sidebar.title("Adjust Model Parameters")

st.sidebar.markdown("### CRP Parameters")
kC0 = st.sidebar.slider("kC0 (Baseline CRP Production)", 0.0, 1.0, 0.1, 0.01)
kC1 = st.sidebar.slider("kC1 (Albumin Effect on CRP)", -1.0, 1.0, 0.05, 0.01)
kC2 = st.sidebar.slider("kC2 (BMI Effect on CRP)", -1.0, 1.0, 0.1, 0.01)
delta = st.sidebar.slider("Delta (CRP Boost for Severe Patients)", 0.0, 2.0, 0.5, 0.1)

st.sidebar.markdown("### Hemoglobin Parameters")
kH1 = st.sidebar.slider("kH1 (CRP -> Hemoglobin Decrease)", 0.0, 1.0, 0.2, 0.01)
kH2 = st.sidebar.slider("kH2 (Low Iron -> Hemoglobin Decrease)", 0.0, 1.0, 0.2, 0.01)
kH3 = st.sidebar.slider("kH3 (Hemoglobin Recovery Rate)", 0.0, 1.0, 0.2, 0.01)

st.sidebar.markdown("### BMI Parameters")
kB0 = st.sidebar.slider("kB0 (BMI Restoration Rate)", 0.0, 1.0, 0.1, 0.01)
kB1 = st.sidebar.slider("kB1 (CRP -> BMI Decrease)", 0.0, 1.0, 0.1, 0.01)
kB2 = st.sidebar.slider("kB2 (Hemoglobin -> BMI Increase)", 0.0, 1.0, 0.1, 0.01)
kB3 = st.sidebar.slider("kB3 (Albumin -> BMI Decrease)", 0.0, 1.0, 0.1, 0.01)

st.sidebar.markdown("### Albumin Parameters")
kA0 = st.sidebar.slider("kA0 (Albumin Restoration Rate)", 0.0, 1.0, 0.1, 0.01)
kA1 = st.sidebar.slider("kA1 (CRP -> Albumin Decrease)", 0.0, 1.0, 0.1, 0.01)
kA2 = st.sidebar.slider("kA2 (BMI -> Albumin Decrease)", 0.0, 1.0, 0.1, 0.01)

st.sidebar.markdown("### Iron Parameters")
kI0 = st.sidebar.slider("kI0 (Iron Restoration Rate)", 0.0, 1.0, 0.1, 0.01)
kI1 = st.sidebar.slider("kI1 (CRP -> Iron Decrease)", 0.0, 1.0, 0.1, 0.01)
kI2 = st.sidebar.slider("kI2 (BMI -> Iron Increase)", 0.0, 1.0, 0.1, 0.01)
kI3 = st.sidebar.slider("kI3 (Albumin -> Iron Increase)", 0.0, 1.0, 0.1, 0.01)

##########################
# 3) Define the ODE System
##########################
def ode_system_EB(y, t,
                  kC0, kC1, kC2,
                  kH1, kH2, kH3,
                  kB0, kB1, kB2, kB3,
                  kA0, kA1, kA2,
                  kI0, kI1, kI2, kI3,
                  K_C=200.0, delta=0.0):
    C, H, B, A, I = y
    C0, H0, B0, A0, I0 = 1.0, 14.0, 18.0, 4.0, 50.0

    dCdt = ((kC0 + delta) + kC1 * (1 - A / A0)) * C * max(0, 1 - C / K_C) + kC2 * (1 - B / B0)
    dHdt = -kH1 * (C - C0) - kH2 * (1 - I / I0) + kH3 * (H0 - H)
    dBdt = kB0 * (B0 - B) - kB1 * (C - C0) + kB2 * (H0 - H) - kB3 * (1 - A / A0)
    dAdt = kA0 * (A0 - A) - kA1 * (C - C0) - kA2 * (1 - B / B0)
    dIdt = kI0 * (I0 - I) - kI1 * (C - C0) + kI2 * (1 - B / B0) + kI3 * (1 - A / A0)

    return np.array([dCdt, dHdt, dBdt, dAdt, dIdt])

##########################
# 4) Solve Using Euler Method
##########################
def euler_method(func, y0, t, *params):
    ys = [y0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        dydt = func(ys[-1], t[i - 1], *params)
        ys.append(ys[-1] + dydt * dt)
    return np.array(ys)

##########################
# 5) Run Simulation
##########################
time_points = np.linspace(0, 10, 50)
initial_conditions = np.array([1.0, 14.0, 18.0, 4.0, 50.0])

params = (kC0, kC1, kC2, kH1, kH2, kH3, kB0, kB1, kB2, kB3, kA0, kA1, kA2, kI0, kI1, kI2, kI3, 200.0, 0.0)
inter_sim = euler_method(ode_system_EB, initial_conditions, time_points, *params)

params_severe = list(params)
params_severe[-1] = delta
se_sim = euler_method(ode_system_EB, initial_conditions, time_points, *params_severe)

##########################
# 6) Plot Results
##########################
fig, ax = plt.subplots(figsize=(10, 6))
biomarkers = ["CRP", "Hemoglobin", "BMI", "Albumin", "Iron"]
colors = ["blue", "green", "red", "gold", "purple"]

for i in range(5):
    ax.plot(time_points, inter_sim[:, i], label=f"Intermediate - {biomarkers[i]}", linestyle="dashed", color=colors[i])
    ax.plot(time_points, se_sim[:, i], label=f"Severe - {biomarkers[i]}", color=colors[i])

ax.set_xlabel("Time")
ax.set_ylabel("Biomarker Level")
ax.legend()
st.pyplot(fig)
