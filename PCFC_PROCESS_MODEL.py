
"""
PCFC (Protonic Ceramic Fuel Cell) multiphysics model.

Solves a coupled 1-D axial model for a co-flow PCFC channel including:
  - electrochemical kinetics (Butler-Volmer, Nernst, ohmic/conc./act. losses)
  - ammonia decomposition kinetics (Temkin–Pyzhev catalytic decompositions via Barat et al.)
  - species mass balances (H2, NH3, O2, H2O, N2) in fuel and air channels
  - energy balances for PEN, interconnect, fuel gas, and air gas
  - Shomate-equation thermodynamic properties (Cp, H, S, G)

The sweep loop iterates over average current density values defined in J_CELL_SWEEP.

Dependencies: numpy, scipy, matplotlib, pandas
Run: python code_PCFC.py
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import pandas as pd


# =========================================================================
# Adjustable Inputs
# =========================================================================


# AVERAGE CURRENT DENSITY SWEEP 
j_cell_min = 4000.0  # [A/m^2] lower parametric sweep limit
j_cell_max = 12000.0  # [A/m^2] upper parametric sweep limit

total_runs = 5  # number of sweeps

#  TEMPERATURE INPUTS

T_A = 775.5
T_F = 774.5
T_PEN_init = 775.0




# =========================================================================
# Parameters
# =========================================================================

N = 60              # [-] number of axial segments
L_channel = 0.1    # [m] channel length
dz = L_channel / N   # [m] segment length

# Fundamental constants
F = 96485.4          # [C/mol]
R = 8.314  # [J/(mol*K)]

# Pressures
P_cell = 1.005*101325       # [Pa] nominal cell pressure (shared default)

# Geometry / discretization
A_active_total = 0.0081  # Huang 2023, Large-area anode-supported protonic ceramic fuel cells combining with multilayer-tape casting and hot-pressing lamination technology
A_contact_total = 0.75*A_active_total   # [m^2] CONTACT AREA BETWEEN PEN AND FUEL CHANNEL


A_active = A_active_total / N   # [m^2] active electrochemical area
A_contact = A_contact_total / N  # [m^2] contact area (set equal to A_active per request)

A_cs = 5.2 *10**(-5)

A_cs_fuel = 0.005*0.75*0.1*10**(-1)   # [m^2] fuel channel cross-sectional area
A_cs_air = A_cs_fuel

d_h_A=0.005  #[m] Hydraulic Diameter Air Channel
d_h_F=0.005  


k_IC = 24 #[W/(m*K)] Thermal Conductivity of Interconnect
A_cs_IC = 6.5*10**(-4)*0.1 
A_contact_IC = 0.25*A_contact


thickness_IC = 6.5*10**(-4)  # Thickness of interconnect [m]
thickness_PEN = (485 +15 + 20)*1e-6  # Thickness of PEN [m]

T_fuel_init = T_F
T_air_init = T_A
T_init=T_PEN_init
T_IC_in = T_PEN_init-1

#------------------------------Electrochemistry---------------------------------

#OPERATING CONDITIONS


# Transport parameters

tau_ca = 0.00002
tau_an = 0.0005

D_an = 3.0e-5*0.2
D_ca = 3.0e-6*0.2


kg_conv= 0.001 # kg/s conversion factor for mass flow rates


MM_H2 = 2.016*kg_conv       # [g/mol] molar mass of H2
MM_O2 = 32.00*kg_conv        # [g/mol] molar mass of O2
MM_H2O = 18.015*kg_conv       # [g/mol] molar mass of H2O
MM_N2 = 28.0134*kg_conv       # [g/mol] molar mass of N2   
MM_NH3 = 17.031*kg_conv        # [g/mol] molar mass of NH3

MM_air = 28.97*kg_conv         # [g/mol] molar mass of air (approximate)


# ===================== PARAMETRIC SWEEP SETTINGS =====================
J_CELL_SWEEP = np.linspace(j_cell_min, j_cell_max, total_runs)  


def update_inlet_flows(j_cell_avg_scalar):
    global j_cell_avg, n_dot_H2_in_f, n_dot_NH3_in_f, n_dot_O2_in_a
    global n_dot_H2O_in_a, n_dot_N2_in_f, n_dot_N2_in_a, n_dot_H2O_in_f

    j_cell_avg = np.array([j_cell_avg_scalar])

    molar_flow_scaling_H2 = j_cell_avg_scalar * A_active_total / (2 * F)
    molar_flow_scaling_NH3 = j_cell_avg_scalar * A_active_total / (3 * F)
    molar_flow_scaling_O2 = j_cell_avg_scalar * A_active_total / (4 * F)*2
    molar_flow_scaling_H2O = molar_flow_scaling_O2 * 0.5

    n_dot_H2_in_f = molar_flow_scaling_H2
    n_dot_NH3_in_f = molar_flow_scaling_NH3
    n_dot_O2_in_a = molar_flow_scaling_O2
    n_dot_H2O_in_a = molar_flow_scaling_H2O
    n_dot_N2_in_f = 0.8 * n_dot_H2_in_f
    n_dot_N2_in_a = 3.76 * n_dot_O2_in_a
    n_dot_H2O_in_f = 0.0


# initialize with the first sweep point
update_inlet_flows(J_CELL_SWEEP[0])


#============================== Cp (T) Calculations - SHOMATE Polynomial Parameters ==============================================

SD = {
    
    "H2":  {"A": 33.066, "B": -11.363417, "C": 11.432816, "D": -2.772874, "E": -0.158558, "F": -9.980797,"G":172.707974, "H": 0.0},
    "H2O": {"A": 30.09200, "B": 6.832514, "C": 6.79345, "D": -2.53448, "E": 0.082139, "F": -250.8810, "G":223.3967,"H": -241.8264},
    "N2":  {"A": 19.50583, "B": 19.88705, "C": -8.598535, "D": 1.369784, "E": 0.527601, "F": -8.671914, "G":212.3900, "H": 0.0},
    "O2": {"A": 30.03235, "B": 8.772972, "C": -3.988133, "D": 0.788313, "E": -0.741599, "F": -11.32468, "G": 236.1663, "H": 0.0},
    "NH3": {"A": 19.99563, "B": 49.77119, "C": -15.37599, "D": 1.921168, "E": 0.189174, "F": -53.30667, "G":203.8591, "H": -45.89806},
}


#==============================STATE VARIABLE STORAGE=================================================

state = {

    "T_PEN": np.full(N, T_PEN_init),
    "T_fuel": np.full(N, T_fuel_init),
    "T_air": np.full(N, T_air_init),


    "H2_fuel": np.full(N+1, n_dot_H2_in_f),
    "N2_fuel": np.full(N+1, n_dot_N2_in_f),
    "H2O_fuel": np.full(N+1, n_dot_H2O_in_f), 
    "NH3_fuel": np.full(N+1, n_dot_NH3_in_f),
    "O2_air": np.full(N+1, n_dot_O2_in_a),
    "H2O_air": np.full(N+1, n_dot_H2O_in_a),
    "N2_air": np.full(N+1, n_dot_N2_in_a),

}


coupled_solv_iters = 1


internal_stats = {
    "V_cell": np.full(N, np.nan),
    "E_Nernst": np.full(N, np.nan),
    "eta_ohm": np.full(N, np.nan),
    "eta_conc": np.full(N, np.nan),
    "eta_act": np.full(N, np.nan),
}
pressure_stats = {
    "P_H2": np.full(N, np.nan),
    "P_O2": np.full(N, np.nan),
    "P_NH3": np.full(N, np.nan),
    "P_H2O": np.full(N, np.nan),
    "P_H2_TPB": np.full(N, np.nan),
    "P_O2_TPB": np.full(N, np.nan),
    "P_H2O_TPB": np.full(N, np.nan)
}

PEN_database = {
    "T_PEN": np.full(N, np.nan),
    "R_PEN": np.full(N, np.nan),
    "q_rxn": np.full(N, np.nan),
    "q_conv_sources": np.full(N, np.nan)
}

HT_behavior_stats_PEN = {
    "HT_conv_air_PEN": np.full(N, np.nan),
    "HT_conv_fuel_PEN": np.full(N, np.nan),
    "HT_cond_PEN": np.full(N-1, np.nan),
    "HT_rxn_PEN": np.full(N, np.nan),
    "HT_NH3_PEN": np.full(N, np.nan),
    "HT_cond_PEN_IC": np.full(N-1, np.nan)
}
    


def GLOBAL_RESIDUALS(x):


    idx = 0

    V_cell = x[idx]
    idx += 1

    j_cell = x[idx:idx+N]
    idx += N

    T_PEN = x[idx:idx+N]
    idx += N

    T_fuel = x[idx:idx+N]
    idx += N
    T_air = x[idx:idx+N]
    idx += N
    T_IC = x[idx:idx+N]
    idx += N
    n_dot_H2 = x[idx : idx + N + 1]
    idx += N + 1  
    n_dot_O2 = x[idx : idx + N + 1]
    idx += N + 1
    n_dot_H2O = x[idx : idx + N + 1]
    idx += N + 1
    n_dot_NH3 = x[idx : idx + N + 1]
    idx += N + 1  



    #=================================================PARTIAL PRESSURES=================================================

    #-------------------------FUEL CHANNEL INPUTS-----------------------------


    n_dot_N2 = np.empty(N+1)
    n_dot_N2[0] = n_dot_N2_in_f
    for i in range(N):
        n_dot_N2[i+1] = n_dot_N2[i] + 0.5 * (n_dot_NH3[i] - n_dot_NH3[i+1])


    mol_frac_H2=np.empty(N)
    mol_frac_N2_f=np.empty(N)
    mol_frac_NH3=np.empty(N)

    for i in range(N):
        mol_frac_H2[i]=n_dot_H2[i]/(n_dot_H2[i]+n_dot_N2[i] + n_dot_NH3[i])
        mol_frac_N2_f[i]=n_dot_N2[i]/(n_dot_H2[i]+n_dot_N2[i]+n_dot_NH3[i])
        mol_frac_NH3[i]=n_dot_NH3[i]/(n_dot_H2[i]+n_dot_N2[i]+n_dot_NH3[i])
        

    P_H2=np.empty(N)
    P_N2=np.empty(N)
    P_NH3=np.empty(N)

    for i in range(N):
        P_H2[i]=mol_frac_H2[i]*P_cell
        P_N2[i]=mol_frac_N2_f[i]*P_cell
        P_NH3[i]=mol_frac_NH3[i]*P_cell
    #-------------------------AIR CHANNEL INPUTS------------------------------------

    n_dot_N2_air = np.ones(N+1)*n_dot_N2_in_a

    mol_frac_O2=np.empty(N)
    mol_frac_H2O=np.empty(N)
    mol_frac_N2_a=np.empty(N)

    for i in range(N):
        mol_frac_O2[i]=n_dot_O2[i]/(n_dot_O2[i]+n_dot_H2O[i]+n_dot_N2_air[i])
        mol_frac_H2O[i]=n_dot_H2O[i]/(n_dot_O2[i]+n_dot_H2O[i]+n_dot_N2_air[i])
        mol_frac_N2_a[i]=n_dot_N2_air[i]/(n_dot_O2[i]+n_dot_H2O[i]+n_dot_N2_air[i])

    P_O2=np.empty(N)
    P_H2O=np.empty(N)
    P_N2_a=np.empty(N)

    for i in range(N):
        P_O2[i]=mol_frac_O2[i]*P_cell
        P_H2O[i]=mol_frac_H2O[i]*P_cell
        P_N2_a[i]=mol_frac_N2_a[i]*P_cell

 
    # ==================================================SHOMATE POLYNOMIAL CALCULATIONS - ENTHALPY, ENTROPY & GIBBS FREE ENERGY=================================================

    T_air_shomate=np.empty(N)
    T_fuel_shomate=np.empty(N)

    cp_O2=np.empty(N)
    cp_H2O=np.empty(N)
    cp_N2=np.empty(N)
    cp_H2=np.empty(N)
    cp_NH3=np.empty(N)

    cp_A = np.empty(N)
    cp_F = np.empty(N)

    delH_O2=np.empty(N)
    delH_H2O=np.empty(N)
    delH_N2=np.empty(N)
    delH_H2=np.empty(N)
    delH_NH3=np.empty(N)
    delH_N2_fuel=np.empty(N)

    delH_ORR=np.empty(N)
    delH_NH3_decomp=np.empty(N)


    delS_O2=np.empty(N)
    delS_H2O=np.empty(N)
    delS_N2=np.empty(N)
    delS_H2=np.empty(N)
    delS_NH3=np.empty(N)
    delS_N2_fuel=np.empty(N)

    delS_ORR=np.empty(N)
    delS_NH3_decomp=np.empty(N)

    delG_ORR=np.empty(N)
    delG_NH3=np.empty(N)

    for i in range(N):
        T_air_shomate[i] = T_air[i]/1000
        T_fuel_shomate[i] = T_fuel[i]/1000

        cp_O2[i]= SD["O2"]["A"] + SD["O2"]["B"]*T_air_shomate[i] + SD["O2"]["C"]*T_air_shomate[i]**2 + SD["O2"]["D"]*T_air_shomate[i]**3 + SD["O2"]["E"]/T_air_shomate[i]**2
        cp_H2O[i]= SD["H2O"]["A"] + SD["H2O"]["B"]*T_air_shomate[i] + SD["H2O"]["C"]*T_air_shomate[i]**2 + SD["H2O"]["D"]*T_air_shomate[i]**3 + SD["H2O"]["E"]/T_air_shomate[i]**2
        cp_N2[i]= SD["N2"]["A"] + SD["N2"]["B"]*T_air_shomate[i] + SD["N2"]["C"]*T_air_shomate[i]**2 + SD["N2"]["D"]*T_air_shomate[i]**3 + SD["N2"]["E"]/T_air_shomate[i]**2
        cp_H2[i]= SD["H2"]["A"] + SD["H2"]["B"]*T_fuel_shomate[i] + SD["H2"]["C"]*T_fuel_shomate[i]**2 + SD["H2"]["D"]*T_fuel_shomate[i]**3 + SD["H2"]["E"]/T_fuel_shomate[i]**2
        cp_NH3[i]= SD["NH3"]["A"] + SD["NH3"]["B"]*T_fuel_shomate[i] + SD["NH3"]["C"]*T_fuel_shomate[i]**2 + SD["NH3"]["D"]*T_fuel_shomate[i]**3 + SD["NH3"]["E"]/T_fuel_shomate[i]**2
        #H° − H°298.15= A*t + B*t2/2 + C*t3/3 + D*t4/4 − E/t + F − H

        #Cp units J/mol*K,
        #ENTALPY UNITS kJ/mol
        #Entropy UNITS J/mol*K

        h_form_H2O= -241.8264  #[kJ/mol]
        h_form_NH3= -45.94  #[kJ/mol] TAKEN FROM NIST
    
        delH_O2[i]= SD["O2"]["A"]*T_air_shomate[i] + SD["O2"]["B"]*T_air_shomate[i]**2/2 + SD["O2"]["C"]*T_air_shomate[i]**3/3 + SD["O2"]["D"]*T_air_shomate[i]**4/4 - SD["O2"]["E"]/T_air_shomate[i] + SD["O2"]["F"] - SD["O2"]["H"]
        delH_H2O[i] = SD["H2O"]["A"]*T_air_shomate[i] + SD["H2O"]["B"]*T_air_shomate[i]**2/2 + SD["H2O"]["C"]*T_air_shomate[i]**3/3 + SD["H2O"]["D"]*T_air_shomate[i]**4/4 - SD["H2O"]["E"]/T_air_shomate[i] + SD["H2O"]["F"] - SD["H2O"]["H"]+h_form_H2O
        delH_H2[i] = SD["H2"]["A"]*T_fuel_shomate[i] + SD["H2"]["B"]*T_fuel_shomate[i]**2/2 + SD["H2"]["C"]*T_fuel_shomate[i]**3/3 + SD["H2"]["D"]*T_fuel_shomate[i]**4/4 - SD["H2"]["E"]/T_fuel_shomate[i] + SD["H2"]["F"] - SD["H2"]["H"]
        delH_N2_fuel[i] = SD["N2"]["A"]*T_fuel_shomate[i] + SD["N2"]["B"]*T_fuel_shomate[i]**2/2 + SD["N2"]["C"]*T_fuel_shomate[i]**3/3 + SD["N2"]["D"]*T_fuel_shomate[i]**4/4 - SD["N2"]["E"]/T_fuel_shomate[i] + SD["N2"]["F"] - SD["N2"]["H"]
        delH_N2[i] = SD["N2"]["A"]*T_air_shomate[i] + SD["N2"]["B"]*T_air_shomate[i]**2/2 + SD["N2"]["C"]*T_air_shomate[i]**3/3 + SD["N2"]["D"]*T_air_shomate[i]**4/4 - SD["N2"]["E"]/T_air_shomate[i] + SD["N2"]["F"] - SD["N2"]["H"]
        delH_NH3[i] = SD["NH3"]["A"]*T_fuel_shomate[i] + SD["NH3"]["B"]*T_fuel_shomate[i]**2/2 + SD["NH3"]["C"]*T_fuel_shomate[i]**3/3 + SD["NH3"]["D"]*T_fuel_shomate[i]**4/4 - SD["NH3"]["E"]/T_fuel_shomate[i] + SD["NH3"]["F"] - SD["NH3"]["H"] + h_form_NH3
        

        delH_ORR[i]= (delH_H2O[i]-delH_H2[i]-delH_O2[i]/2)*1000
        delH_NH3_decomp[i]= (3/2*delH_H2[i]+0.5*delH_N2_fuel[i]-delH_NH3[i])*1000


        delS_O2[i] = SD["O2"]["A"]*np.log(T_air_shomate[i]) + SD["O2"]["B"]*T_air_shomate[i] + SD["O2"]["C"]*T_air_shomate[i]**2/2 + SD["O2"]["D"]*T_air_shomate[i]**3/3 - SD["O2"]["E"]/(2*T_air_shomate[i]**2) + SD["O2"]["G"]
        delS_H2O[i] = SD["H2O"]["A"]*np.log(T_air_shomate[i]) + SD["H2O"]["B"]*T_air_shomate[i] + SD["H2O"]["C"]*T_air_shomate[i]**2/2 + SD["H2O"]["D"]*T_air_shomate[i]**3/3 - SD["H2O"]["E"]/(2*T_air_shomate[i]**2) + SD["H2O"]["G"]
        delS_H2[i] = SD["H2"]["A"]*np.log(T_fuel_shomate[i]) + SD["H2"]["B"]*T_fuel_shomate[i] + SD["H2"]["C"]*T_fuel_shomate[i]**2/2 + SD["H2"]["D"]*T_fuel_shomate[i]**3/3 - SD["H2"]["E"]/(2*T_fuel_shomate[i]**2) + SD["H2"]["G"]
        delS_N2_fuel[i] = SD["N2"]["A"]*np.log(T_fuel_shomate[i]) + SD["N2"]["B"]*T_fuel_shomate[i] + SD["N2"]["C"]*T_fuel_shomate[i]**2/2 + SD["N2"]["D"]*T_fuel_shomate[i]**3/3 - SD["N2"]["E"]/(2*T_fuel_shomate[i]**2) + SD["N2"]["G"]
        delS_N2[i] = SD["N2"]["A"]*np.log(T_air_shomate[i]) + SD["N2"]["B"]*T_air_shomate[i] + SD["N2"]["C"]*T_air_shomate[i]**2/2 + SD["N2"]["D"]*T_air_shomate[i]**3/3 - SD["N2"]["E"]/(2*T_air_shomate[i]**2) + SD["N2"]["G"]
        delS_NH3[i] = SD["NH3"]["A"]*np.log(T_fuel_shomate[i]) + SD["NH3"]["B"]*T_fuel_shomate[i] + SD["NH3"]["C"]*T_fuel_shomate[i]**2/2 + SD["NH3"]["D"]*T_fuel_shomate[i]**3/3 - SD["NH3"]["E"]/(2*T_fuel_shomate[i]**2) + SD["NH3"]["G"]

        delS_ORR[i] = delS_H2O[i] - delS_H2[i] - delS_O2[i]/2
        delS_NH3_decomp[i] = 3/2*delS_H2[i]+0.5*delS_N2_fuel[i]-delS_NH3[i]


        delG_ORR[i] = delH_ORR[i] - (T_PEN[i]*delS_ORR[i])
        delG_NH3[i] = delH_NH3[i] - (T_PEN[i]*delS_NH3_decomp[i])


    #=================================================AMMONIA DECOMPOSITION KINETICS=================================================
    invT = np.empty(N)
    ln_k = np.empty(N)
    ln_khat = np.empty(N)
    khat = np.empty(N)
    k = np.empty(N)
    for i in range(N):

        invT[i] = 1.0 / T_PEN[i] 


        # ---- k coefficients ---- Kinetic rate constant k describes the intrinsic speed of the chemical reaction on the catalyst surface at a given temperature. 
        a_k = -5.996
        b_k = 4.344e4
        c_k = -2.610e7

        # ---- khat coefficients ---- Hydrogen inhibition coefficient k  represents the competitive adsorption between hydrogen and ammonia on the anode catalyst surface
        a_khat = -6.181
        b_khat = 2.849e4
        c_khat = -1.287e7

        ln_k[i] = a_k + b_k*invT[i] + c_k*invT[i]**2
        ln_khat[i] = a_khat + b_khat*invT[i] + c_khat*invT[i]**2

        k[i] = np.exp(ln_k[i])
        khat[i] = np.exp(ln_khat[i])

    
    A_cat = 4*10**6 #[m-1] Active catalyst area per electrode volume (using barat et al.

    react_NH3 = np.empty(N)

    RR_NH3 = np.empty(N)
    RR_H2 = np.empty(N)
    RR_N2 = np.empty(N)
    

    RR_NH3 = np.zeros(N)
    RR_H2  = np.zeros(N)  
    RR_N2  = np.zeros(N)
    Q_NH3_decomp = np.zeros(N)
    tanh_arg = np.empty(N)

    for i in range(N):

        # --- Ammonia decomposition kinetics based on Temkin–Pyzhev catalytic decomposition model (Barat et al. 2020) ---
        # Implemented tanh smoothing function to avoid numerical instability at low ammonia partial pressures

        tanh_arg[i] = 0.5 * (np.tanh((P_NH3[i] -3)))+0.5
        react_NH3[i] = - tanh_arg[i]*(k[i] * np.maximum(P_NH3[i],0)**2)/((P_H2[i]**(3/2)+khat[i]*np.maximum(P_NH3[i],1000))**2)

        
        RR_NH3[i] = react_NH3[i] * A_cat * A_cs * dz   
        
        
        RR_H2[i] = -3/2 * RR_NH3[i]
        RR_N2[i] = -1/2 * RR_NH3[i]

        Q_NH3_decomp[i] = RR_NH3[i] * delH_NH3_decomp[i]  # Heat released/absorbed by ammonia decomposition in this segment


    #==============================ELECTROCHEMISTRY SOLVER=================================================
    
    # ================= ELECTROCHEMISTRY (CURRENT-CONTROLLED) =================


    
    i_cell = np.empty(N)

    

    P_O2_TPB = np.empty(N)
    P_H2_TPB = np.empty(N)
    P_H2O_TPB = np.empty(N)

    j_0ca = np.empty(N)
    j_0an = np.empty(N)

    eta_conc = np.empty(N)
    eta_ohm = np.empty(N)
    eta_act = np.empty(N)
    E_Nernst = np.empty(N)

    R_V_cell = np.empty(N)  # Residual for voltage balance at each segment
    ASR = np.empty(N)  # Area-specific resistance for ohmic losses

    for i in range(N):


        i_cell[i] = j_cell[i] * A_active


        E_Nernst[i] = -delG_ORR[i]/(2*F) + (R * T_PEN[i] / (2 * F)) * np.log(
            (P_H2[i]/P_cell * np.sqrt(P_O2[i]/P_cell)) / (P_H2O[i]/P_cell)
        )

        # --- Concentration losses ---
        P_O2_TPB[i] = P_O2[i] - (R*T_PEN[i]*tau_ca/(4*F*D_ca)) * j_cell[i]
        P_H2O_TPB[i] = P_H2O[i] + (R*T_PEN[i]*tau_ca/(2*F*D_ca)) * j_cell[i]
        P_H2_TPB[i] = P_cell - (P_cell - P_H2[i]) * np.exp(
            (R*T_PEN[i]*tau_an/(2*F*D_an*P_cell)) * j_cell[i]
        )
       

        
        eta_conc[i] = (R*T_PEN[i]/(2*F)) * (
            np.log(P_H2[i] / P_H2_TPB[i])
            + np.log((P_H2O_TPB[i]/P_cell * np.sqrt(P_O2[i]/P_cell)) /
                    (P_H2O[i]/P_cell * np.sqrt(P_O2_TPB[i]/P_cell)))
        ) 
        
        # Ohmic losses based on Albrecht et al. 2012, 
        ASR[i] = 0.0039 * np.exp(3.9551*1000/T_PEN[i])  # Based on Albrecht
        eta_ohm[i] = ASR[i] * i_cell[i] / (A_active*100**2)


        # --- Activation ---   ZHU RICOTE AND KEE THERMODYNAMICS PAPER FOR KINETIC PARAMETERS -43360 AND -83000
        # THE 8.817e9 AND 9.577e8 ARE BASED ON ZHANG Mathematical modeling of a proton-conducting solid oxide fuel cell with current leakage AND SAHLI
        j_0ca[i] = 8.817e9 * np.exp(-43360/(R*T_PEN[i]))
        j_0an[i] = 9.577e8 * np.exp(-83000/(R*T_PEN[i]))

        
        eta_act[i] = (
            (2*R*T_PEN[i]/F) * np.arcsinh(j_cell[i]/(2*j_0an[i]))
        + (2*R*T_PEN[i]/F) * np.arcsinh(j_cell[i]/(2*j_0ca[i]))
        )
        

        internal_stats["V_cell"]= V_cell
        internal_stats["E_Nernst"][i] = E_Nernst[i]
        internal_stats["eta_ohm"][i] = eta_ohm[i]
        internal_stats["eta_conc"][i] = eta_conc[i]
        internal_stats["eta_act"][i] = eta_act[i]
        
        R_V_cell[i] = E_Nernst[i] - eta_ohm[i] - eta_conc[i] - eta_act[i] - V_cell

    R_j_cell = j_cell_avg - np.sum(j_cell)/N  # Simple residual to check if current distribution is consistent with average current


    # ---------------------------------ENERGY BALANCE---------------------------------

    n = 2  # electrons transferred per ORR


   
    #------------------------- Air Side Calculations-------------------------

    rho_A=np.empty(N)
    Re_A=np.empty(N)
    mu_A=np.empty(N)
    k_A=np.empty(N)
    volume_dot_air=np.empty(N)
    vel_air=np.empty(N)
    Pr_A=np.empty(N)
    Nu_A=np.empty(N)
    h_air=np.empty(N)

    MM_air_en=np.empty(N)
    
    
    
    
    for i in range(N):

        # air channel heat transfer coefficient calculations based on Dittus-Boelter correlation for turbulent flow in a circular pipe (valid for Re > 4000)


        volume_dot_air[i] = (n_dot_O2[i]+n_dot_H2O[i]+n_dot_N2_air[i])*R*T_air[i]/(P_cell)  #[m3/s] Volumetric flow rate of fuel at segment i (using ideal gas law)
        vel_air[i] = volume_dot_air[i] / A_cs_air  #[m/s] Velocity of fuel at segment i (using volumetric flow rate and contact area)
       
        MM_air_en[i] = mol_frac_O2[i]*MM_O2 + mol_frac_N2_a[i]*MM_N2 + mol_frac_H2O[i]*MM_H2O  # Molar mass of fuel mixture at segment i (using mole fractions and molar masses)

        rho_A[i] = P_cell * MM_air_en[i] / (R * T_air[i])

        mu_A[i] = (0.0000273829*T_air[i] + 0.0179576572)*10**(-3) # BASED ON ASPEN HYSYS FOR AIR

        k_A[i] = 0.0085678629 + 0.0000699089 * T_air[i]  

        cp_A[i] = mol_frac_O2[i] * cp_O2[i] + mol_frac_N2_a[i] * cp_N2[i] + mol_frac_H2O[i] * cp_H2O[i]

        Re_A[i] = rho_A[i] * vel_air[i] * d_h_A / mu_A[i]

        Pr_A[i] = mu_A[i] * cp_A[i] / (k_A[i]*MM_air_en[i])

        Nu_A[i] = 1.86 * (Re_A[i] * Pr_A[i] * d_h_A / dz) ** (1/3)

        h_air[i]=Nu_A[i]*k_A[i]/d_h_A


   

    #------------------------- Fuel side calculations -------------------------
    MM_fuel=np.empty(N)
    rho_F=np.empty(N)
    Re_F=np.empty(N)
    mu_F=np.empty(N)
    k_F=np.empty(N)
    volume_dot_fuel=np.empty(N)
    vel_fuel=np.empty(N)
    Pr_F=np.empty(N)
    Nu_F=np.empty(N)
    h_fuel=np.empty(N)


    for i in range(N):

        # Fuel channel heat transfer coefficient calculations 

        volume_dot_fuel[i] = (n_dot_H2[i]+n_dot_N2[i]+n_dot_NH3[i])*R*T_fuel[i]/(P_cell)  #[m3/s] Volumetric flow rate of fuel at segment i (using ideal gas law)
        vel_fuel[i] = volume_dot_fuel[i] / A_cs_fuel  #[m/s] Velocity of fuel at segment i (using volumetric flow rate and contact area)
        
        MM_fuel[i] = mol_frac_H2[i]*MM_H2 + mol_frac_N2_f[i]*MM_N2 + mol_frac_NH3[i]*MM_NH3  # Molar mass of fuel mixture at segment i (using mole fractions and molar masses)


        # Calculate thermal conductivity and viscosity 
        k_F[i] = 0.0455250000 + 0.0001685000 * T_fuel[i] 
    
        mu_F[i] = (0.0000235427*T_fuel[i] +0.0133115641)*10**(-3) # BASED ON ASPEN HYSYS FOR 0.1 NH3, 0.3 N2, 0.6 H2 MIXTURE

        # Calculate density
        rho_F[i] = P_cell * MM_fuel[i] / (R * T_fuel[i])

        # Calculate specific heat capacity (weighted average)
        cp_F[i] = mol_frac_H2[i] * cp_H2[i]+ mol_frac_N2_f[i] * cp_N2[i] + mol_frac_NH3[i] * cp_NH3[i]

        # Calculate Reynolds number
        Re_F[i] = rho_F[i] * vel_fuel[i] * d_h_F / mu_F[i]

        # Calculate Prandtl number
        Pr_F[i] = mu_F[i] * cp_F[i] / (k_F[i]*MM_fuel[i])

        # Calculate Nusselt number
        Nu_F[i] = 1.86 * (Re_F[i] * Pr_F[i] * d_h_F / dz) ** (1/3)

        # Calculate heat transfer coefficient
        h_fuel[i] = Nu_F[i] * k_F[i] / d_h_F


    #-------------------------PEN HEAT BALANCE SOLVER-------------------------


    k_PEN=2 #[W/(m*K)] Thermal Conductivity of PEN BASED ON NIU 2025

    A=np.zeros((N,N))  
    b=np.zeros(N)      

    q_rxn=np.zeros(N)
    q_conv_sources=np.zeros(N)


    R_total = thickness_IC / (k_IC * A_cs_IC) + thickness_PEN / (k_PEN * A_cs)  # Total thermal resistance of the interconnect segment

    for i in range(1, N-1): 

       
        q_rxn[i]= (i_cell[i])*((delH_ORR[i])/(n*F)-V_cell)   # Reaction heat per segment

        q_conv_sources[i]= (h_air[i]*A_contact*T_air[i] + h_fuel[i]*A_contact*T_fuel[i]) # Convective heat transfer from air and fuel channels to PEN segment

        # Setting up the finite difference matrix for the PEN temperature distribution

        A[i, i-1] = k_PEN*A_cs / dz**2
        A[i, i]   = -2*k_PEN*A_cs / dz**2 - h_air[i]*A_contact - h_fuel[i]*A_contact - 1/R_total
        A[i, i+1] = k_PEN*A_cs / dz**2

        b[i] = q_rxn[i] - q_conv_sources[i] - Q_NH3_decomp[i] - 1/R_total*T_IC[i]


    

    R_PEN=np.zeros(N)

    T_left=T_PEN_init             

    R_PEN=A @ T_PEN -b
    R_PEN[0]=T_PEN[0]-T_left
    R_PEN[N-1] = (T_PEN[N-1] - T_PEN[N-2]) / dz 

    PEN_database = {
        "T_PEN": T_PEN,
        "R_PEN": R_PEN,
        "q_rxn": q_rxn,
        "Q_NH3_decomp": Q_NH3_decomp,
        "q_conv_sources": q_conv_sources
    }


    #==========================Interconnect heat balance solver ========================================================

    A_IC = np.zeros((N,N)) 
    b_IC = np.zeros(N)      
    R_IC=np.zeros(N)


    q_cond_sources_IC=np.zeros(N)
    q_conv_sources_IC = np.zeros(N)
    
    r_ohmic_IC=np.zeros(N)
    q_ohmic_IC=np.zeros(N)

    for i in range(1, N-1):
        r_ohmic_IC[i] = 1.176 * 1e-4 * (thickness_IC/A_cs_IC)

        q_conv_sources_IC[i]= (h_air[i]*A_contact_IC*T_air[i] + h_fuel[i]*A_contact_IC*T_fuel[i])
        q_cond_sources_IC[i] = 1/R_total * (T_PEN[i]) 
        q_ohmic_IC[i] = i_cell[i]**2 * r_ohmic_IC[i] / (2*dz)  # Ohmic heating in the interconnect segment

        A_IC[i, i-1] = k_IC*A_cs_IC / dz**2
        A_IC[i, i]   = -2*k_IC*A_cs_IC / dz**2 - h_air[i]*A_contact_IC - h_fuel[i]*A_contact_IC - 1/R_total   
        A_IC[i, i+1] = k_IC*A_cs_IC / dz**2

   
        b_IC[i] = - q_conv_sources_IC[i] - q_cond_sources_IC[i] - q_ohmic_IC[i] 

    R_IC = A_IC @ T_IC - b_IC
    R_IC[0] = T_IC[0] - T_IC_in  # Coupling with PEN at left boundary
    R_IC[N-1] = (T_IC[N-1] - T_IC[N-2]) / dz  # Insulated boundary at right end

    #-------------------------FUEL HEAT BALANCE SOLVER-------------------------


    n_dot_total_fuel=np.zeros(N)

    for i in range(N):

        n_dot_total_fuel[i] = n_dot_H2[i]+n_dot_N2[i] + n_dot_NH3[i]


    R_fuel=np.zeros(N)

    R_fuel[0]=T_fuel[0]-T_F

    


    for i in range(0, N-1):  #Adjust range as needed 

        R_fuel[i+1] = (T_fuel[i+1]- T_fuel[i]- (h_fuel[i] * A_contact * (T_PEN[i] - T_fuel[i]) / (n_dot_total_fuel[i] * cp_F[i])) - (h_fuel[i] * A_contact_IC * (T_IC[i] - T_fuel[i]) / (n_dot_total_fuel[i] * cp_F[i])))
        
    


    #------------------------- AIR HEAT BALANCE SOLVER-------------------------

    n_dot_total_air = np.zeros(N)
    for i in range(N):
        n_dot_total_air[i] = (n_dot_O2[i] + n_dot_H2O[i]+n_dot_N2_air[i])

    R_AIR=np.zeros(N)

    R_AIR[0]=T_air[0]-T_A

    q_conv_air=np.zeros(N)

    for i in range(0, N-1):  #Adjust range as needed

        q_conv_air[i]= h_air[i]*A_contact*(T_PEN[i]-T_air[i])

        # Update air temperature along the channel
        R_AIR[i+1] = T_air[i+1] - T_air[i] - (q_conv_air[i])/(n_dot_total_air[i]*cp_A[i]) - (h_air[i] * A_contact_IC * (T_IC[i] - T_air[i]) / (n_dot_total_air[i] * cp_A[i]))
    

    # ===================================================MASS BALANCE============================================================


    #--------------------------------FUEL CHANNEL MASS BALANCE----------------------------

    R_n_dot_H2=np.zeros(N+1)
    R_n_dot_NH3=np.zeros(N+1)


    R_n_dot_H2[0]=n_dot_H2[0]-n_dot_H2_in_f
    R_n_dot_NH3[0]=n_dot_NH3[0]-n_dot_NH3_in_f

    for i in range(N):
        R_n_dot_H2[i+1]=n_dot_H2[i+1]-n_dot_H2[i]+(i_cell[i]/(2*F))-RR_H2[i]
        R_n_dot_NH3[i+1] = n_dot_NH3[i+1]-(n_dot_NH3[i]+RR_NH3[i])


    #--------------------------------AIR CHANNEL MASS BALANCE----------------------------
    R_n_dot_O2=np.zeros(N+1)
    R_n_dot_O2[0]=n_dot_O2[0]-n_dot_O2_in_a


    R_n_dot_H2O=np.zeros(N+1)
    R_n_dot_H2O[0]=n_dot_H2O[0]-n_dot_H2O_in_a

    for i in range(N):
        R_n_dot_O2[i+1]=n_dot_O2[i+1]-(n_dot_O2[i]-i_cell[i]/((4*F)))
        R_n_dot_H2O[i+1]=n_dot_H2O[i+1]-(n_dot_H2O[i]+i_cell[i]/((2*F)))

  

    pressure_stats["P_H2"]=P_H2
    pressure_stats["P_O2"]=P_O2
    pressure_stats["P_NH3"]=P_NH3
    pressure_stats["P_H2O"]=P_H2O
    pressure_stats["P_H2_TPB"]=P_H2_TPB
    pressure_stats["P_O2_TPB"]=P_O2_TPB
    pressure_stats["P_H2O_TPB"]=P_H2O_TPB

    HT_conv_air_PEN=np.zeros(N)
    HT_conv_fuel_PEN=np.zeros(N)
    HT_cond_PEN=np.full(N, np.nan)
    HT_rxn_PEN=np.full(N, np.nan)
    HT_NH3_PEN=np.full(N, np.nan)
    HT_conv_air_IC=np.zeros(N)
    HT_conv_fuel_IC=np.zeros(N)
    HT_cond_IC=np.zeros(N)
    HT_rxn_IC=np.zeros(N)
    HT_ohmic_IC=np.zeros(N)
    HT_conv_PEN_air=np.zeros(N)
    HT_conv_IC_air=np.zeros(N)
    HT_conv_PEN_fuel=np.zeros(N)
    HT_conv_IC_fuel=np.zeros(N)
    HT_cond_PEN_IC=np.zeros(N)
    HT_cond_IC_PEN=np.zeros(N)


    # ====================================== HEAT BALANCE CONSISTENCY CHECK ======================================
    for i in range(N):

        HT_conv_air_PEN[i]= h_air[i]*A_contact*(T_air[i]-T_PEN[i])
        HT_conv_fuel_PEN[i]= h_fuel[i]*A_contact*(T_fuel[i]-T_PEN[i])
        
        HT_rxn_PEN[i]= q_rxn[i]
        HT_NH3_PEN[i]= Q_NH3_decomp[i]

        HT_conv_air_IC[i]= h_air[i]*A_contact_IC*(T_air[i]-T_IC[i])
        HT_conv_fuel_IC[i]= h_fuel[i]*A_contact_IC*(T_fuel[i]-T_IC[i])
        HT_rxn_IC[i]= q_ohmic_IC[i]
        HT_ohmic_IC[i]= q_ohmic_IC[i]

        HT_conv_PEN_air[i]= h_air[i]*A_contact*(T_air[i]-T_PEN[i])
        HT_conv_IC_air[i]= h_air[i]*A_contact_IC*(T_air[i]-T_IC[i])
        
        HT_conv_PEN_fuel[i]= h_fuel[i]*A_contact*(T_fuel[i]-T_PEN[i])
        HT_conv_IC_fuel[i]= h_fuel[i]*A_contact_IC*(T_fuel[i]-T_IC[i])

    for i in range(1, N-1):
        HT_cond_PEN[i]= k_PEN*A_cs*(T_PEN[i-1]-2*T_PEN[i]+T_PEN[i+1])/dz**2
        HT_cond_PEN_IC[i] = 1/R_total * (T_IC[i]-T_PEN[i])  # Conduction from PEN to IC, positive if PEN is hotter than IC
        HT_cond_IC[i]= k_IC*A_cs_IC*(T_IC[i-1]-2*T_IC[i]+T_IC[i+1])/dz**2
        HT_cond_IC_PEN[i] = 1/R_total * (T_PEN[i]-T_IC[i])  # Conduction from IC to PEN, positive if IC is hotter than PEN


    HT_behavior_stats_PEN["HT_conv_air_PEN"]=HT_conv_air_PEN
    HT_behavior_stats_PEN["HT_conv_fuel_PEN"]=HT_conv_fuel_PEN
    HT_behavior_stats_PEN["HT_cond_PEN"]=HT_cond_PEN
    HT_behavior_stats_PEN["HT_rxn_PEN"]=HT_rxn_PEN
    HT_behavior_stats_PEN   ["HT_NH3_PEN"]=HT_NH3_PEN
    HT_behavior_stats_PEN["HT_cond_PEN_IC"]=HT_cond_PEN_IC


    
    return np.concatenate([R_V_cell, R_j_cell, R_PEN, R_fuel, R_AIR, R_IC, R_n_dot_O2, R_n_dot_H2, R_n_dot_H2O, R_n_dot_NH3])  #, R_n_dot_H2, R_n_dot_O2, R_n_dot_H2O


n_dot_H2_in_guess=0.00001/MM_H2
n_dot_O2_in_guess=0.00005/MM_O2
n_dot_H2O_in_guess=0.00002/MM_H2O
n_dot_NH3_in_guess=0.00001/MM_NH3


V_cell_x0 = np.array([0.9])  # V
j_cell_x0 = np.full(N, 2000)  # A, uniform
T_PEN_x0 = np.full(N, T_init)
T_fuel_x0 = np.full(N, T_F)
T_air_x0 = np.full(N, T_A)
T_IC_x0 = np.full(N, T_init)     
n_dot_H2_x0=np.full(N+1, n_dot_H2_in_guess)
n_dot_O2_x0=np.full(N+1, n_dot_O2_in_guess)
n_dot_H2O_x0=np.full(N+1, n_dot_H2O_in_guess)
n_dot_NH3_x0=np.full(N+1, n_dot_NH3_in_guess)


x0 = np.concatenate([V_cell_x0, j_cell_x0, T_PEN_x0, T_fuel_x0, T_air_x0, T_IC_x0, n_dot_H2_x0, n_dot_O2_x0, n_dot_H2O_x0, n_dot_NH3_x0])  #THE FIRST ELEMENT USED TO BE V_CELL_GUESS_X0, CHECK IF ARRAY IS CORRECT


def extract_solution_fields(solution_vector):
    return {
        "V_cell": solution_vector[0],
        "j_cell": solution_vector[1:1+N], 
        "T_PEN": solution_vector[1+N:1+2*N],
        "T_fuel": solution_vector[1+2*N:1+3*N],
        "T_air": solution_vector[1+3*N:1+4*N],
        "T_IC": solution_vector[1+4*N:1+5*N],
        "n_dot_H2": solution_vector[1+5*N:1+6*N+1],
        "n_dot_O2": solution_vector[1+6*N+1:1+7*N+2],
        "n_dot_H2O": solution_vector[1+7*N+2:1+8*N+3],
        "n_dot_NH3": solution_vector[1+8*N+3:1+9*N+4],
    }


def update_guess_for_case(x_guess):
    x_guess = x_guess.copy()
    x_guess[0] = 0.9
    x_guess[1:1+N] = np.clip(x_guess[1:1+N], 100.0, None)

    idx0 = 1 + 5*N
    idx1 = idx0 + (N+1)
    idx2 = idx1 + (N+1)
    idx3 = idx2 + (N+1)

    x_guess[idx0:idx0+N+1] = np.maximum(x_guess[idx0:idx0+N+1], 0.5*n_dot_H2_in_f)
    x_guess[idx1:idx1+N+1] = np.maximum(x_guess[idx1:idx1+N+1], 0.5*n_dot_O2_in_a)
    x_guess[idx2:idx2+N+1] = np.maximum(x_guess[idx2:idx2+N+1], 0.0)
    x_guess[idx3:idx3+N+1] = np.maximum(x_guess[idx3:idx3+N+1], 0.5*n_dot_NH3_in_f)

    x_guess[idx0] = n_dot_H2_in_f
    x_guess[idx1] = n_dot_O2_in_a
    x_guess[idx2] = n_dot_H2O_in_a
    x_guess[idx3] = n_dot_NH3_in_f
    return x_guess

#============================================ Distance along the channel============================================

distance = np.empty(N)

for i in range(N):
    distance[i] = i * dz


if __name__ == "__main__":

    sweep_results = []
    x_guess = x0.copy()

    for trial, j_target in enumerate(J_CELL_SWEEP, start=1):
        update_inlet_flows(float(j_target))
        x_guess = update_guess_for_case(x_guess)

        print(f"\n{'='*70}")
        print(f"Trial {trial}: j_cell_avg = {j_target:.0f} A/m²")
        print(f"{'='*70}")

        sol, infodict, ier, msg = fsolve(GLOBAL_RESIDUALS, x_guess, full_output=True, maxfev=20000)
        GLOBAL_RESIDUALS(sol)   # ensure internal_stats & pressure_stats reflect converged solution
        res_norm = np.linalg.norm(GLOBAL_RESIDUALS(sol))
        fields = extract_solution_fields(sol)

        print(f"  ier = {ier} | nfev = {infodict.get('nfev','NA')} | ||R|| = {res_norm:.3e} | V = {fields['V_cell']:.4f} V")

        sweep_results.append({
            "trial":     trial,
            "j_avg":     float(j_target),
            "ier":       ier,
            "nfev":      infodict.get("nfev", np.nan),
            "res_norm":  res_norm,
            # scalars
            "V_cell":    fields["V_cell"],
            # spatial profiles (length N)
            "j_cell":    fields["j_cell"].copy(),
            "T_PEN":     fields["T_PEN"].copy(),
            "T_fuel":    fields["T_fuel"].copy(),
            "T_air":     fields["T_air"].copy(),
            "T_IC":      fields["T_IC"].copy(),
            "E_Nernst":  internal_stats["E_Nernst"].copy(),
            "eta_ohm":   internal_stats["eta_ohm"].copy(),
            "eta_conc":  internal_stats["eta_conc"].copy(),
            "eta_act":   internal_stats["eta_act"].copy(),
            # partial pressures
            "P_H2":      pressure_stats["P_H2"].copy(),
            "P_NH3":     pressure_stats["P_NH3"].copy(),
            # molar flows (length N+1)
            "n_dot_H2":  fields["n_dot_H2"].copy(),
            "n_dot_NH3": fields["n_dot_NH3"].copy(),
            # NH3 conversion
            "NH3_conversion": 1.0 - fields["n_dot_NH3"][-1] / fields["n_dot_NH3"][0],
        })

        x_guess = sol.copy()

    # ------------------------------------------------------------------------------
    # SUMMARY TABLE
    # ------------------------------------------------------------------------------
    sweep_summary = pd.DataFrame([{
        "trial":         r["trial"],
        "j_avg_A_m2":    r["j_avg"],
        "V_cell_V":      r["V_cell"],
        "solver_flag":   r["ier"],
        "nfev":          r["nfev"],
        "residual_norm": r["res_norm"],
        "E_Nernst_avg":  np.mean(r["E_Nernst"]),
        "eta_ohm_avg":   np.mean(r["eta_ohm"]),
        "eta_conc_avg":  np.mean(r["eta_conc"]),
        "eta_act_avg":   np.mean(r["eta_act"]),
    } for r in sweep_results])
    print("\nj_cell Sweep Summary:")
    print(sweep_summary.to_string(index=False))

    # ==============================================================================
    # PLOTS — established Option C format
    # ==============================================================================
    import os as _os

    segment_index = np.arange(N)
    node_index    = np.arange(N + 1)
    colors_j      = plt.cm.plasma(np.linspace(0.1, 0.85, len(sweep_results)))
    j_labels      = [f'{r["j_avg"]:.0f} A/m²' for r in sweep_results]

    # ------------------------------------------------------------------------------
    # Figure C1 — Profile chain: H2+NH3 flows → P_H2 → E_Nernst
    # ------------------------------------------------------------------------------
    fig_c1, axes_c1 = plt.subplots(1, 3, figsize=(15, 5))
    fig_c1.suptitle(
        r"Current Density Effect: Molar Flows $\rightarrow$ Partial Pressure $\rightarrow$ Nernst Potential",
        fontsize=12, fontweight="bold"
    )

    ax_mol = axes_c1[0]
    for r, c in zip(sweep_results, colors_j):
        ax_mol.plot(node_index, r["n_dot_H2"]  * 1e3, color=c, lw=1.8, label=f'{r["j_avg"]:.0f} A/m²')
        ax_mol.plot(node_index, r["n_dot_NH3"] * 1e3, color=c, lw=1.8, ls="--")
    ax_mol.set_xlabel("Axial Distance [m]", fontsize=10)
    ax_mol.set_ylabel(r"$\dot{n}$ [mmol/s]", fontsize=10)
    ax_mol.set_title(r"$\dot{n}_{H_2}$ (—) and $\dot{n}_{NH_3}$ (- -)", fontsize=10)
    ax_mol.grid(True, alpha=0.35, ls="--")
    ax_mol.tick_params(labelsize=9)
    # no legend here — moved to Nernst panel

    ax_ph2 = axes_c1[1]
    for r, c in zip(sweep_results, colors_j):
        ax_ph2.plot(distance, r["P_H2"] / 1e3, color=c, lw=1.8)
    ax_ph2.set_xlabel("Axial Distance [m]", fontsize=10)
    ax_ph2.set_ylabel(r"$P_{H_2}$ [kPa]", fontsize=10)
    ax_ph2.set_title(r"$H_2$ Partial Pressure", fontsize=10)
    ax_ph2.grid(True, alpha=0.35, ls="--")
    ax_ph2.tick_params(labelsize=9)

    ax_en = axes_c1[2]
    for r, c in zip(sweep_results, colors_j):
        ax_en.plot(distance, r["E_Nernst"], color=c, lw=1.8)
    ax_en.set_xlabel("Axial Distance [m]", fontsize=10)
    ax_en.set_ylabel(r"$E_{Nernst}$ [V]", fontsize=10)
    ax_en.set_title("Nernst Potential", fontsize=10)
    ax_en.grid(True, alpha=0.35, ls="--")
    ax_en.tick_params(labelsize=9)
    # Legend on Nernst panel — most free space in lower left
    handles = [plt.Line2D([0], [0], color=c, lw=1.8) for c in colors_j]
    ax_en.legend(handles, j_labels, fontsize=8, loc="lower left", framealpha=0.8)

    fig_c1.tight_layout()

    #plt.close(fig_c1)


    # ------------------------------------------------------------------------------
    # Figure C2 — Summary: overpotential breakdown | V_cell & E_Nernst | NH3 conversion
    # ------------------------------------------------------------------------------
    x            = np.arange(len(sweep_results))
    V_cell_vals  = [r["V_cell"]           for r in sweep_results]
    eta_ohm_avg  = [np.mean(r["eta_ohm"])  for r in sweep_results]
    eta_conc_avg = [np.mean(r["eta_conc"]) for r in sweep_results]
    eta_act_avg  = [np.mean(r["eta_act"])  for r in sweep_results]
    E_Nernst_avg = [np.mean(r["E_Nernst"]) for r in sweep_results]
    NH3_conv     = [r["NH3_conversion"] * 100 for r in sweep_results]

    fig_c2, axes_c2 = plt.subplots(1, 2, figsize=(12, 5))
    fig_c2.suptitle(
        r"Electrochemical Performance Summary vs Average Current Density",
        fontsize=12, fontweight="bold"
    )

    bar_w = 0.5

    # Left — stacked overpotential breakdown
    ax_bar = axes_c2[0]
    ax_bar.bar(x, eta_ohm_avg,  width=bar_w, label=r"$\eta_{ohm}$",  color="#4C72B0")
    ax_bar.bar(x, eta_conc_avg, width=bar_w, bottom=eta_ohm_avg,
               label=r"$\eta_{conc}$", color="#DD8452")
    ax_bar.bar(x, eta_act_avg,  width=bar_w,
               bottom=[o+c for o, c in zip(eta_ohm_avg, eta_conc_avg)],
               label=r"$\eta_{act}$",  color="#55A868")
    ax_bar.set_xlabel(r"$j_{avg}$ [A/m²]", fontsize=10)
    ax_bar.set_ylabel("Average Overpotential [V]", fontsize=10)
    ax_bar.set_title("Overpotential Breakdown", fontsize=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'{r["j_avg"]:.0f}' for r in sweep_results], fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.35, ls="--")
    ax_bar.legend(fontsize=9, framealpha=0.8)

    # Right — V_cell and E_Nernst vs j_avg (polarization curve view)
    ax_v = axes_c2[1]
    ax_v.plot(x, E_Nernst_avg, 'o--', color="steelblue",  lw=1.8, label=r"$\langle E_{Nernst} \rangle$")
    ax_v.plot(x, V_cell_vals,  's-',  color="darkorange",  lw=1.8, label=r"$V_{cell}$")
    ax_v.set_xlabel(r"$j_{avg}$ [A/m²]", fontsize=10)
    ax_v.set_ylabel("Voltage [V]", fontsize=10)
    ax_v.set_title(r"$V_{cell}$ and $\langle E_{Nernst} \rangle$", fontsize=10)
    ax_v.set_xticks(x)
    ax_v.set_xticklabels([f'{r["j_avg"]:.0f}' for r in sweep_results], fontsize=9)
    ax_v.grid(True, alpha=0.35, ls="--")
    ax_v.legend(fontsize=9, framealpha=0.8)

    fig_c2.tight_layout()

    # ==============================================================================
    # NEW FIGURES — Temperature profiles, NH3 decomposition, Power density
    # ==============================================================================

    # Shared scalar pre-computations
    T_PEN_profiles = [r["T_PEN"] for r in sweep_results]
    T_IC_profiles  = [r.get("T_IC", np.full(N, np.nan)) for r in sweep_results]

    T_PEN_peak   = [np.max(p)  for p in T_PEN_profiles]
    T_PEN_inlet  = [p[0]       for p in T_PEN_profiles]
    T_PEN_delta  = [pk - i0    for pk, i0 in zip(T_PEN_peak,  T_PEN_inlet)]

    T_IC_peak    = [np.max(p)  for p in T_IC_profiles]
    T_IC_inlet   = [p[0]       for p in T_IC_profiles]
    T_IC_delta   = [pk - i0    for pk, i0 in zip(T_IC_peak,   T_IC_inlet)]

    NH3_conv_pct = [r["NH3_conversion"] * 100 for r in sweep_results]
    PD_profiles  = [r["j_cell"] * r["V_cell"] * 0.1 for r in sweep_results]   # W/m^2 → mW/cm^2
    PD_avg       = [np.mean(pd) for pd in PD_profiles]
    PD_peak      = [np.mean(pd)  for pd in PD_profiles]

    bar_w  = 0.5

    # Shared y-axis limits for temperature figures (for direct comparability)
    all_T_prof = np.concatenate(T_PEN_profiles + T_IC_profiles)
    all_T_prof = all_T_prof[~np.isnan(all_T_prof)]
    T_ylim = (np.min(all_T_prof) - 5, np.max(all_T_prof) + 5)

    all_dT = T_PEN_delta + T_IC_delta
    dT_ylim = (0, max(all_dT) * 1.25)

    # Shared summary x-axis
    x_sum = np.arange(len(sweep_results))
    j_xtick_labels = [f'{r["j_avg"]:.0f}' for r in sweep_results]

    def _summary_axes(ax, ylabel, title):
        ax.set_xlabel(r"$j_{avg}$ [A/m²]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x_sum)
        ax.set_xticklabels(j_xtick_labels, fontsize=9)
        ax.grid(True, axis="y", alpha=0.35, ls="--")
        ax.tick_params(labelsize=9)

    def _profile_axes(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.35, ls="--")
        ax.tick_params(labelsize=9)

    # ------------------------------------------------------------------------------
    # Figure T_PEN — PEN temperature profile + inlet/peak/ΔT summary
    # ------------------------------------------------------------------------------
    fig_tpen, (ax_tp_prof, ax_tp_sum) = plt.subplots(1, 2, figsize=(13, 5))
    fig_tpen.suptitle(r"PEN Temperature — $j_{cell}$ Sweep",
                      fontsize=13, fontweight="bold")

    for r, c, p in zip(sweep_results, colors_j, T_PEN_profiles):
        ax_tp_prof.plot(distance, p, color=c, lw=1.8,
                        label=f'{r["j_avg"]:.0f} A/m²')
    ax_tp_prof.set_ylim(T_ylim)
    _profile_axes(ax_tp_prof, "Axial Distance [m]", r"$T_{PEN}$ [K]", "PEN Temperature Profile")
    ax_tp_prof.legend(fontsize=8, framealpha=0.8)

    ax_tp_sum.plot(x_sum, T_PEN_inlet, 'o--', color="steelblue", lw=1.8, label=r"$T_{PEN,inlet}$")
    ax_tp_sum.plot(x_sum, T_PEN_peak,  's-',  color="darkorange", lw=1.8, label=r"$T_{PEN,peak}$")
    ax_tp_sum.set_ylim(T_ylim)
    ax_dT_pen = ax_tp_sum.twinx()
    ax_dT_pen.bar(x_sum, T_PEN_delta, width=bar_w, alpha=0.28, color="grey",
                  label=r"$\Delta T_{PEN}$")
    ax_dT_pen.set_ylim(dT_ylim)
    ax_dT_pen.set_ylabel(r"$\Delta T_{PEN}$ [K]", fontsize=10, color="grey")
    ax_dT_pen.tick_params(axis="y", labelcolor="grey", labelsize=9)
    _summary_axes(ax_tp_sum, r"$T_{PEN}$ [K]",
                  r"$T_{PEN}$ Inlet, Peak & $\Delta T$ vs $j_{avg}$")
    lines1, labs1 = ax_tp_sum.get_legend_handles_labels()
    lines2, labs2 = ax_dT_pen.get_legend_handles_labels()
    ax_tp_sum.legend(lines1 + lines2, labs1 + labs2, fontsize=9, framealpha=0.8)

    fig_tpen.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------
    # Figure T_IC — Interconnect temperature profile + inlet/peak/ΔT summary
    #               Same axis limits as T_PEN for direct comparison
    # ------------------------------------------------------------------------------
    fig_tic, (ax_ti_prof, ax_ti_sum) = plt.subplots(1, 2, figsize=(13, 5))
    fig_tic.suptitle(r"Interconnect Temperature — $j_{cell}$ Sweep",
                     fontsize=13, fontweight="bold")

    for r, c, p in zip(sweep_results, colors_j, T_IC_profiles):
        ax_ti_prof.plot(distance, p, color=c, lw=1.8,
                        label=f'{r["j_avg"]:.0f} A/m²')
    ax_ti_prof.set_ylim(T_ylim)
    _profile_axes(ax_ti_prof, "Axial Distance [m]", r"$T_{IC}$ [K]", "Interconnect Temperature Profile")
    ax_ti_prof.legend(fontsize=8, framealpha=0.8)

    ax_ti_sum.plot(x_sum, T_IC_inlet, 'o--', color="steelblue", lw=1.8, label=r"$T_{IC,inlet}$")
    ax_ti_sum.plot(x_sum, T_IC_peak,  's-',  color="darkorange", lw=1.8, label=r"$T_{IC,peak}$")
    ax_ti_sum.set_ylim(T_ylim)
    ax_dT_ic = ax_ti_sum.twinx()
    ax_dT_ic.bar(x_sum, T_IC_delta, width=bar_w, alpha=0.28, color="grey",
                 label=r"$\Delta T_{IC}$")
    ax_dT_ic.set_ylim(dT_ylim)
    ax_dT_ic.set_ylabel(r"$\Delta T_{IC}$ [K]", fontsize=10, color="grey")
    ax_dT_ic.tick_params(axis="y", labelcolor="grey", labelsize=9)
    _summary_axes(ax_ti_sum, r"$T_{IC}$ [K]",
                  r"$T_{IC}$ Inlet, Peak & $\Delta T$ vs $j_{avg}$")
    lines1, labs1 = ax_ti_sum.get_legend_handles_labels()
    lines2, labs2 = ax_dT_ic.get_legend_handles_labels()
    ax_ti_sum.legend(lines1 + lines2, labs1 + labs2, fontsize=9, framealpha=0.8)

    fig_tic.tight_layout()
    plt.show()


    # ------------------------------------------------------------------------------
    # Figure NH3 — Decomposition profiles + utilisation bar
    # ------------------------------------------------------------------------------
    fig_nh3, axes_nh3 = plt.subplots(1, 3, figsize=(17, 5))
    fig_nh3.suptitle(r"NH$_3$ Decomposition & Utilisation — $j_{cell}$ Sweep",
                     fontsize=13, fontweight="bold")

    # Left: H2 (solid) & NH3 (dashed) on shared axis
    ax_both = axes_nh3[0]
    for r, c in zip(sweep_results, colors_j):
        ax_both.plot(node_index, r["n_dot_H2"]  * 1e3, color=c, lw=1.8,
                     label=f'{r["j_avg"]:.0f} A/m²')
        ax_both.plot(node_index, r["n_dot_NH3"] * 1e3, color=c, lw=1.8, ls="--")
    _profile_axes(ax_both, "Node Index", r"$\dot{n}$ [mmol/s]",
                  r"$\dot{n}_{H_2}$ (—) and $\dot{n}_{NH_3}$ (- -)")
    handles_nh3 = [plt.Line2D([0], [0], color=c, lw=1.8) for c in colors_j]
    ax_both.legend(handles_nh3, j_labels, fontsize=8, framealpha=0.8)

    # Middle: NH3 profile only
    ax_nh3_only = axes_nh3[1]
    for r, c in zip(sweep_results, colors_j):
        ax_nh3_only.plot(node_index, r["n_dot_NH3"] * 1e3, color=c, lw=1.8)
    _profile_axes(ax_nh3_only, "Node Index", r"$\dot{n}_{NH_3}$ [mmol/s]",
                  r"NH$_3$ Decomposition Profile")

    # Right: utilisation bar with annotations
    ax_util = axes_nh3[2]
    ax_util.bar(x_sum, NH3_conv_pct, width=bar_w, color=colors_j,
                edgecolor="white", linewidth=0.5)
    _summary_axes(ax_util, r"NH$_3$ Utilisation [%]",
                  r"NH$_3$ Utilisation at Outlet")
    ax_util.set_ylim(0, 105)
    for xi, val in zip(x_sum, NH3_conv_pct):
        ax_util.text(xi, val + 1.5, f'{val:.1f}%', ha='center',
                     fontsize=8, fontweight='bold')

    fig_nh3.tight_layout()
    plt.show()


    # ------------------------------------------------------------------------------
    # Figure Power Density — spatial profiles + average/peak summary (twin axis)
    # ------------------------------------------------------------------------------
    fig_pd, (ax_pd_prof, ax_pd_sum) = plt.subplots(1, 2, figsize=(13, 5))
    fig_pd.suptitle(r"PCFC Power Density — $j_{cell}$ Sweep",
                    fontsize=13, fontweight="bold")

    for r, c, pd in zip(sweep_results, colors_j, PD_profiles):
        ax_pd_prof.plot(distance, pd, color=c, lw=1.8,
                        label=f'{r["j_avg"]:.0f} A/m²')
    _profile_axes(ax_pd_prof, "Axial Distance [m]", r"Power Density [mW/cm$^2$]",
                  "Local Power Density Profile")
    ax_pd_prof.legend(fontsize=8, framealpha=0.8)

    #ax_pd_sum.bar(x_sum, PD_avg, width=bar_w, color="#4C72B0", label=r"$\langle P \rangle$")
    #ax_pk = ax_pd_sum.twinx()
    #ax_pk.plot(x_sum, PD_peak, 's--', color="darkorange", lw=1.8, label=r"$P_{peak}$")
    '''ax_pk.set_ylabel(r"Peak Power Density [mW/cm$^2$]", fontsize=10, color="darkorange")
    ax_pk.tick_params(axis="y", labelcolor="darkorange", labelsize=9)'''



    ax_pd_sum.bar(x_sum, PD_avg, width=bar_w,
              color="#4C72B0",
              label=r"$\langle P \rangle$")

    ax_pd_sum.plot(x_sum, PD_peak,
               's--',
               color="darkorange",
               lw=1.8,
               label=r"$P_{peak}$")
    
    _summary_axes(ax_pd_sum, r"Avg Power Density [mW/cm$^2$]",
                  r"Average & Peak Power Density vs $j_{avg}$")
    
    ax_pd_sum.legend(fontsize=9, framealpha=0.8)

    '''lines1, labs1 = ax_pd_sum.get_legend_handles_labels()
    lines2, labs2 = ax_pk.get_legend_handles_labels()
    ax_pd_sum.legend(lines1 + lines2, labs1 + labs2, fontsize=9, framealpha=0.8)'''

    fig_pd.tight_layout()

    plt.show()
    # plt.close(fig_pd)
    
