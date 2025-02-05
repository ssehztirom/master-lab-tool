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
""")

st.markdown("## **ODE System for EB Biomarkers**")
st.latex(r"""
\begin{aligned}
\frac{dC}{dt} &= ((k_{C0} + \delta) + k_{C1} (1 - A / A_0)) C \cdot \max(0, 1 - C / K_C) + k_{C2} (1 - B / B_0) \\
\frac{dH}{dt} &= - k_{H1} (C - C_0) - k_{H2} (1 - I / I_0) + k_{H3} (H_0 - H) \\
\frac{dB}{dt} &= k_{B0} (B_0 - B) - k_{B1} (C - C_0) + k_{B2} (H_0 - H) - k_{B3} (1 - A / A_0) \\
\frac{dA}{dt} &= k_{A0} (A_0 - A) - k_{A1} (C - C_0) - k_{A2} (1 - B / B_0) \\
\frac{dI}{dt} &= k_{I0} (I_0 - I) - k_{I1} (C - C_0) + k_{I2} (1 - B / B_0) + k_{I3} (1 - A / A_0)
\end{aligned}
""")

# 2. Display Parameter Explanations in LaTeX
st.markdown("## **Parameter Explanations**")

st.latex(r"k_{C0}: \text{ Baseline CRP production rate.}")
st.latex(r"k_{C1}: \text{ How low albumin } (A) \text{ boosts CRP production.}")
st.latex(r"k_{C2}: \text{ Additional CRP increase when BMI } (B) \text{ is below baseline.}")
st.latex(r"\delta: \text{ Extra CRP production shift for severe patients.}")
st.latex(r" \quad \text{If } \delta = 0, \text{ patient is intermediate.}")
st.latex(r" \quad \text{If } \delta > 0, \text{ patient is severe.}")

st.latex(r"k_{H1}: \text{ How CRP elevation reduces hemoglobin } (H).")
st.latex(r"k_{H2}: \text{ Effect of low iron } (I) \text{ on hemoglobin.}")
st.latex(r"k_{H3}: \text{ Rate at which hemoglobin is restored to } H_0.")

st.latex(r"k_{B0}: \text{ Rate at which BMI returns to normal.}")
st.latex(r"k_{B1}: \text{ How CRP elevation reduces BMI.}")
st.latex(r"k_{B2}: \text{ How low hemoglobin affects BMI.}")
st.latex(r"k_{B3}: \text{ How low albumin affects BMI.}")

st.latex(r"k_{A0}: \text{ Albumin recovery rate.}")
st.latex(r"k_{A1}: \text{ How CRP reduces albumin.}")
st.latex(r"k_{A2}: \text{ How low BMI affects albumin.}")

st.latex(r"k_{I0}: \text{ Iron recovery rate.}")
st.latex(r"k_{I1}: \text{ How CRP reduces iron.}")
st.latex(r"k_{I2}: \text{ How low BMI affects iron.}")
st.latex(r"k_{I3}: \text{ How low albumin affects iron.}")

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
