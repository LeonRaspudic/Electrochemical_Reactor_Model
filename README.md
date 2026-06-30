# Electrochemical_Reactor_Model


Dependencies: numpy, scipy, matplotlib, pandas

Run: PCFC_PROCESS_MODEL.py


*******OVERVIEW******

Solves a coupled 1-D axial process model for a co-flow PCFC channel including:
  - species mass balances (H2, NH3, O2, H2O, N2) in fuel and air channels
  - energy balances for PEN, interconnect, fuel gas, and air gas
  - Shomate-equation thermodynamic properties (Cp, H, S, G)
  - electrochemical kinetics (Butler-Volmer, Nernst, ohmic/conc./act. losses)
  - ammonia decomposition kinetics (Temkin–Pyzhev catalytic decompositions via Barat et al.)


Coupled nonlinear system solved with scipy.optimize.fsolve (Newton-Krylov)

The sweep loop iterates over average current density values defined in J_CELL_SWEEP.






*****ASSUMPTIONS******

  Adiabatic system (representative of interior stack cells)
  Equipotential electrodes (no in-plane voltage gradients)
  Steady state operation
  PEN uniform across thickness; Interconnects lumped as single element
  Multispecies diffusion simplified through Fickian formulation
  Laminar, fully-developed flow (no cross-section gradients)
  No pressure losses across the cell
  Radiation heat transfer between PEN and interconnect considered negligible


  PEN ENERGY BALANCE: 
  
    Axial conduction (2nd-order FD)
    Convection ↔ fuel & air channels
    Conduction ↔ interconnect (thermal resistance)
    Heat source: ORR heat generation
    Heat sink: NH₃ decomposition (endothermic)
  
  
  INTERCONNECT ENERGY BALANCE:
  
    Axial conduction (2nd-order FD)
    Convection ↔ fuel & air channels
    Conduction ↔ PEN structure
    Ohmic (Joule) heating — I^2 R_IC
  
  
  FUEL CHANNEL ENERGY BALANCE: 
  
    Convection-only (algebraic)
    Heat exchange with PEN surface
    Heat exchange with interconnect
    Thermally separated from air channel by PEN
    Local Nusselt (Graetz–Lévêque correlation)
  
  
  
  AIR CHANNEL ENERGY BALANCE: 
  
    Convection-only (algebraic)
    Heat exchange with PEN surface
    Heat exchange with interconnect
    Local Nusselt (Graetz–Lévêque correlation)
  
  FUEL AND AIR CHANNEL MASS BALANCE:
  
    Electrochemical consumption (H₂, O₂) and generation (H₂O) − Faraday′s law
    Thermo−catalytic NH₃ decomposition
    N₂ treated as inert diluent on fuel side
    Air-side N₂ fixed (constant), defined as 3.76 n ̇_(O2,in) (atmospheric composition)
  
  AMMONIA DECOMPOSITION KINETICS
  
    Ammonia decomposition kinetics is captured through a Langmuir-Hinshelwood based expression developed by Barat et. al.
    Expression is calibrated to fit experimental data by Okura et. al. 
    Calibrated against temperature range 650 – 950  K and atmospheric pressure.

  Numerical stabilization - continuous max function / hyperbolic tangent function (Prevents instability at reactant depletion )


******RESULTS********

PCFC_PROCESS_MODEL: Current density parametric sweep

    - Molar Flow Profiles -> H2 Partial Pressure -> Maximum Theoretical Voltage
    - PEN Temperature Profiles -> Temperature Change & Peak Temperature vs Current density
    - Interconnect Temperature Profiles -> Temperature Change & Peak Temperature vs Current density
    - Operating Loss Mechanisms Distribution - Ohmic, Concentration & Activation Losses
    - Power Density Profile -> Delivered power vs. Current Density





