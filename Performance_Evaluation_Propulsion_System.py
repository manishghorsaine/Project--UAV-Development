import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt

TARGET_ID = 20#<-------- Change the configuration here
RHO = 1.157
V_stall = 10 
vz = 2.0  
g = 9.81
Total_mass = 1.5

# Component Database: [Mass(g), Cost(Euro)]
prop_db = {"8x6": [15, 4], "8x8": [15, 4], "9x6": [19, 4], "9x7.5": [19, 5], "9x9": [19, 5]}
motor_db = {"MotorA": [83, 76], "MotorB": [83, 76], "MotorC": [113, 83], "MotorD": [113, 83], "MotorE": [143, 89]}
batt_db = {"BatteryI": [158, 30], "BatteryII": [167, 23], "BatteryIII": [181, 35]}

# Paths for experimental data of propeller, motor specifications and battery data.
PROP_DIR = r"E:\Aerospace\05 Semester\Lab Course\Design_Build_Fly\Sub_Team\Propeller_Database" 
MOTOR_FILE = r"E:\Aerospace\05 Semester\Lab Course\Design_Build_Fly\Sub_Team\Motor_Database\Motor_Datas.csv"
BATTERY_FILE = r"E:\Aerospace\05 Semester\Lab Course\Design_Build_Fly\Sub_Team\Battery_Database\Battery_Database.csv"

# 
# Configuration selection
# 
PROPS = ["8x6", "8x8", "9x6", "9x7.5", "9x9"]
MOTORS = ["MotorA", "MotorB", "MotorC", "MotorD", "MotorE"]
BATTERIES = ["BatteryI", "BatteryII", "BatteryIII"]

combinations = []
id_counter = 1
for p in PROPS:
    for m in MOTORS:
        for b in BATTERIES:
            combinations.append({"ID": id_counter, "Prop": p, "Motor": m, "Batt": b})
            id_counter += 1

selected = next((item for item in combinations if item["ID"] == TARGET_ID), None)

def get_ct_cp(J_query, J_arr, Ct_arr, Cp_arr):
    Jq = np.clip(J_query, J_arr.min(), J_arr.max()) # In order to not go beyond available datas.
    return np.interp(Jq, J_arr, Ct_arr), np.interp(Jq, J_arr, Cp_arr)

def find_equilibrium(V_inf, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, throttle): #Finding equilibrium rpm Qmotor=Qpropeller
    rpm_range = np.linspace(1000, 18000, 20000)
    eta_esc = 0.75 + 0.2 * throttle
    n_vals = rpm_range / 60.0
    J_vals = V_inf / (n_vals * D)
    Ct_v, Cp_v = get_ct_cp(J_vals, J_arr, Ct_arr, Cp_arr)
    Qp_vals = RHO * Cp_v * (n_vals**2) * (D**5) / (2 * math.pi)
    
    Ub = V_nom
    for _ in range(20): #Voltage sags, fixed-point iteration as stated in the report 
        Im = np.maximum((throttle * Ub - (2 * np.pi * n_vals) / Kv_SI) / Ri_m, I0)
        Ib = (Im * throttle) / eta_esc
        Ub = V_nom - Ib * R_batt
        
    Qm_vals = (Im - I0) / Kv_SI 
    diffs = np.abs(Qm_vals - Qp_vals)
    idx = np.argmin(diffs) #index of minimum difference of torque i.e. Qm-Qp=!0
    
    if Qp_vals[idx] <= 1e-6 or diffs[idx] / Qp_vals[idx] > 0.02: #Check for error and tolerance
        return {"success": False}
    
    n_eq = n_vals[idx]
    thrust = Ct_v[idx] * RHO * (n_eq**2) * (D**4)
    p_batt = Ub[idx] * Ib[idx]
    
    return {"success": True, "rpm": n_eq * 60, "thrust": thrust, "I_m": Im[idx], "I_b": Ib[idx], 
            "P_batt": p_batt, "P_shaft": Qp_vals[idx] * (2 * np.pi * n_eq), "throttle": throttle, 
            "eff": (thrust * V_inf) / p_batt if p_batt > 0 and V_inf > 0 else 0,
            "Ub": Ub[idx], "Um": throttle * Ub[idx]} 

def solve_throttle_for_drag(V_target, D_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt): #For given flight conditions: cruise 1, 20 m/s, find the throttle settings for given Treq and Velocity and compute all required values. 
    low, high = 0.1, 1 #Used bisection methods learnt in Computational foundations
    sol_max = find_equilibrium(V_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, 1.0)
    if not sol_max["success"] or sol_max["thrust"] < D_target: return None
    for _ in range(25):
        mid = (low + high) / 2
        sol = find_equilibrium(V_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, mid)
        if not sol["success"] or sol["thrust"] < D_target: low = mid
        else: high = mid
    return sol

if selected:
    m_p, _ = prop_db[selected['Prop']]; m_m, _ = motor_db[selected['Motor']]; m_b, _ = batt_db[selected['Batt']]
    weight = Total_mass * g

    df_p = pd.read_csv(os.path.join(PROP_DIR, f"{selected['Prop']}.csv"))
    df_m = pd.read_csv(MOTOR_FILE).set_index('Name').loc[selected['Motor']]
    df_b = pd.read_csv(BATTERY_FILE).set_index('Name').loc[selected['Batt']]
    
    drag_velocity =[9.58, 10.0, 10.4, 10.9, 11.3, 11.7, 12.2, 12.6, 13.0, 13.5, #given drag vs velocity values from Aerodynamics team
 13.9, 14.3, 14.8, 15.2, 15.6, 16.0, 16.5, 16.9, 17.3, 17.8,
 18.2, 18.6, 19.1, 19.5, 19.9, 20.4, 20.8, 21.2, 21.6, 22.1,
 22.5, 22.9, 23.4, 23.8, 24.2, 24.7, 25.1, 25.5, 26.0, 26.4,
 26.8, 27.2, 27.7, 28.1, 28.5, 29.0, 29.4, 29.8, 30.3,
 30.7, 31.1, 31.6, 32.0, 32.4, 32.8, 33.3, 33.7, 34.1,
 34.6, 35.0]

    drag_values = [0.97, 0.955, 0.947, 0.945, 0.947, 0.955, 0.966, 0.983, 1.00, 1.02,
 1.05, 1.08, 1.11, 1.14, 1.18, 1.21, 1.25, 1.30, 1.34, 1.38,
 1.43, 1.48, 1.53, 1.58, 1.63, 1.69, 1.74, 1.80, 1.86, 1.92,
 1.98, 2.04, 2.10, 2.17, 2.23, 2.30, 2.37, 2.44, 2.51, 2.58,
 2.66, 2.73, 2.81, 2.88, 2.96, 3.04, 3.12, 3.20, 3.28,
 3.36, 3.45, 3.53, 3.61, 3.70, 3.79, 3.88, 3.96,
 4.06, 4.15, 4.24]

    D = float(selected['Prop'].split('x')[0]) * 0.0254 #Automate for other D for eg. 9 no need to change manually.
    Kv_SI, Ri_m, I0 = float(df_m['KV']) * 2 * np.pi / 60, float(df_m['Ri']), float(df_m['I0'])
    V_nom, Q_ah, C_rate = float(df_b['Nominal_Voltage']), float(df_b['Capacity_Ah']), float(df_b['C_Rate'])
    R_batt = float(df_b['Cells_S']) * ((3.367e-3 * C_rate + 0.4097) / (C_rate * Q_ah))
   # print("***R_battery= ",R_batt)
   # print("***R_motor= ",Ri_m)

    full_throttle = [] #Calculation for full throttle
    v_max = 0
    for V in range(0, 41):
        s = find_equilibrium(V, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, Ri_m, I0, V_nom, D, R_batt, 1.0)
        if s["success"]:
            dr = np.interp(V, drag_velocity, drag_values)
            full_throttle.append(s | {"V": V})
            if s['thrust'] >= dr: v_max = V
    
    if v_max < 25:
        print(f" ID {TARGET_ID} FAILED: v_max ({v_max} m/s) < 25 m/s") #Check for Vmax>=25
        sys.exit()

    df_ft = pd.DataFrame(full_throttle)
    Pconsumption_max = df_ft['P_batt'].max()
    T_max = df_ft.loc[df_ft['V'] == 0, 'thrust'].values[0]
    v_plot = [item['V'] for item in full_throttle]
    thrust_max = [item['thrust'] for item in full_throttle]

    drag_plot = [np.interp(v, drag_velocity, drag_values) for v in v_plot]

    plt.figure(figsize=(10, 6)) #Just an example plot to check
    plt.plot(v_plot, thrust_max, 'b-', lw=2.5, label='Max Available Thrust (100% Throttle)')
    plt.plot(v_plot, drag_plot, 'r--', lw=2.5, label='Total Aircraft Drag')
    plt.axvline(x=v_max, color='gray', linestyle=':', alpha=0.7)
    plt.scatter(v_max, np.interp(v_max, v_plot, thrust_max), color='black', zorder=5)
    plt.annotate(f'V_max = {v_max} m/s', 
              xy=(v_max, np.interp(v_max, v_plot, thrust_max)), 
              xytext=(v_max-8, np.interp(v_max, v_plot, thrust_max)+1),
              arrowprops=dict(facecolor='black', shrink=0.05),
              fontsize=10, fontweight='bold')
    plt.fill_between(v_plot, drag_plot, thrust_max, where=(np.array(thrust_max) > np.array(drag_plot)), 
                  color='green', alpha=0.1, label='Excess Thrust (Climb Potential)')
    plt.title(f'Thrust vs. Drag: Configuration {TARGET_ID}', fontsize=14)
    plt.xlabel('Airspeed (m/s)', fontsize=12)
    plt.ylabel('Force (N)', fontsize=12)
    plt.xlim(0, max(v_plot))
    plt.ylim(0, max(thrust_max) + 2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    Level_Flight = [] #Level flight calculation
    for V in drag_velocity:
        dr = np.interp(V, drag_velocity, drag_values)# to match velocity and drag row
        sol = solve_throttle_for_drag(V, dr, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, Ri_m, I0, V_nom, D, R_batt) #Calling function to compute 
        if sol: Level_Flight.append(sol | {"V": V, "Drag": dr})
    
    df_lvl = pd.DataFrame(Level_Flight)
    v_min = df_lvl.loc[df_lvl['P_batt'].idxmin(), 'V']
    v_climb = max(v_min, V_stall + 3) #Given from RFP that any velocity should be atleast 3+Vstall
    #Equation for climb
    Treq_climb = np.interp(v_climb, drag_velocity, drag_values) + weight * (vz / v_climb)
    climb_sol = solve_throttle_for_drag(v_climb, Treq_climb, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, Ri_m, I0, V_nom, D, R_batt)

    v_cruise1_2 = v_climb #Cruise calculation
    treq_cruise1_2 = np.interp(v_cruise1_2, drag_velocity, drag_values)
    cruise1_2_sol = solve_throttle_for_drag(v_cruise1_2, treq_cruise1_2, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, Ri_m, I0, V_nom, D, R_batt)

    v_takeoff = 1.2 * V_stall #Given from Performance team
    takeoff_sol = find_equilibrium(v_takeoff, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, Ri_m, I0, V_nom, D, R_batt, 1.0)
    
    v20 = df_lvl.iloc[(df_lvl['V']-20).abs().argsort()[:1]].iloc[0]

    # Output print
    print(f"Configuration ID {TARGET_ID}: {selected['Prop']} | {selected['Motor']} | {selected['Batt']}")
    
    print(f"\n--- FULL THROTTLE ---")
    static_sol = df_ft.loc[df_ft['V'] == 0].iloc[0]
    print(f"v_max: {v_max} m/s | T_max (Static): {T_max:.2f} N | Pconsumption_max: {Pconsumption_max:.2f} W")
    print(f"rpmequilibrium: {static_sol['rpm']:.0f} | Imotor: {static_sol['I_m']:.2f} A | Ibattery: {static_sol['I_b']:.2f} A | Umotor: {static_sol['Um']:.2f} V | Ubattery: {static_sol['Ub']:.2f} V")

    print(f"\n--- CRUISE (v= 20 m/s) ---")
    print(f"Treq: {v20['Drag']:.2f} N | Pcons: {v20['P_batt']:.2f} W | Eff: {v20['eff']:.4f} | Throttle: {v20['throttle']:.3f}")
    print(f"rpmequilibrium: {v20['rpm']:.0f} | Imotor: {v20['I_m']:.2f} A | Ibattery: {v20['I_b']:.2f} A | Umotor: {v20['Um']:.2f} V | Ubattery: {v20['Ub']:.2f} V")

    print(f"\n--- CRUISE 1 & 2 (v= {v_cruise1_2} m/s) ---")
    if cruise1_2_sol:
        print(f"Treq: {treq_cruise1_2:.2f} N | Pcons: {cruise1_2_sol['P_batt']:.2f} W | Eff: {cruise1_2_sol['eff']:.4f} | Throttle: {cruise1_2_sol['throttle']:.3f}")
        print(f"rpmequilibrium: {cruise1_2_sol['rpm']:.0f} | Imotor: {cruise1_2_sol['I_m']:.2f} A | Ibattery: {cruise1_2_sol['I_b']:.2f} A | Umotor: {cruise1_2_sol['Um']:.2f} V | Ubattery: {cruise1_2_sol['Ub']:.2f} V")

    print(f"\n--- CLIMB ANALYSIS (v= {v_climb} m/s) ---")
    print(f"Treq_climb: {Treq_climb:.2f} N | Pcons: {climb_sol['P_batt']:.2f} W | Eff: {climb_sol['eff']:.4f} | Throttle: {climb_sol['throttle']:.3f}")
    print(f"rpmequilibrium: {climb_sol['rpm']:.0f} | Imotor: {climb_sol['I_m']:.2f} A | Ibattery: {climb_sol['I_b']:.2f} A | Umotor: {climb_sol['Um']:.2f} V | Ubattery: {climb_sol['Ub']:.2f} V")

    print(f"\n--- TAKE OFF ANALYSIS (Full Throttle @ v= {v_takeoff:.1f} m/s) ---")
    if takeoff_sol["success"]:
        print(f"T_takeoff: {takeoff_sol['thrust']:.2f} N | Pcons_takeoff: {takeoff_sol['P_batt']:.2f} W | Eff_takeoff: {takeoff_sol['eff']:.4f}")
        print(f"rpmequilibrium: {takeoff_sol['rpm']:.0f} | Imotor: {takeoff_sol['I_m']:.2f} A | Ibattery: {takeoff_sol['I_b']:.2f} A | Umotor: {takeoff_sol['Um']:.2f} V | Ubattery: {takeoff_sol['Ub']:.2f} V")