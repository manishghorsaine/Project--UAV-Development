import numpy as np
import pandas as pd
import math
import os

# 
# Constraint Check for all configurations: 
# 
RHO = 1.157
V_STALL = 9.5
vz = 2.0  
g = 9.81

PROP_DIR = r"E:\Aerospace\05 Semester\Lab Course\Design_Build_Fly\Sub_Team\Propeller_Database"
MOTOR_FILE = r"E:\Aerospace\05 Semester\Lab Course\Design_Build_Fly\Sub_Team\Motor_Database\Motor_Datas.csv"
BATTERY_FILE = r"E:\Aerospace\05 Semester\Lab Course\Design_Build_Fly\Sub_Team\Battery_Database\Battery_Database.csv"
DRAG_FILE = r"E:\Aerospace\05 Semester\Lab Course\Design_Build_Fly\Sub_Team\Drag_Vs_Velocity_Database\Drag_Vs_Velocity01.csv"

PROPS = ["8x6", "8x8", "9x6", "9x7.5", "9x9"]
MOTORS = ["MotorA", "MotorB", "MotorC", "MotorD", "MotorE"]
BATTERIES = ["BatteryI", "BatteryII", "BatteryIII"]

PROP_DB = {"8x6": [15, 4], "8x8": [15, 4], "9x6": [19, 4], "9x7.5": [19, 5], "9x9": [19, 5]}
MOTOR_DB = {"MotorA": [83, 76], "MotorB": [83, 76], "MotorC": [113, 83], "MotorD": [113, 83], "MotorE": [143, 89]}
BATT_DB = {"BatteryI": [158, 30], "BatteryII": [167, 23], "BatteryIII": [181, 35]}

def get_ct_cp(J_query, J_arr, Ct_arr, Cp_arr):
    Jq = np.clip(J_query, J_arr.min(), J_arr.max())
    return np.interp(Jq, J_arr, Ct_arr), np.interp(Jq, J_arr, Cp_arr)

def find_equilibrium(V_inf, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, throttle):
    rpm_range = np.linspace(1000, 18000, 20000)
    eta_esc = 0.75 + 0.2 * throttle
    n_vals = rpm_range / 60.0
    J_vals = V_inf / (n_vals * D)
    Ct_v, Cp_v = get_ct_cp(J_vals, J_arr, Ct_arr, Cp_arr)
    Qp_vals = RHO * Cp_v * (n_vals**2) * (D**5) / (2 * math.pi)
    Ub = V_nom
    for _ in range(20):
        Im = np.maximum((throttle * Ub - (2 * np.pi * n_vals) / Kv_SI) / Ri_m, I0)
        Ib = (Im * throttle) / eta_esc
        Ub = V_nom - Ib * R_batt
    Qm_vals = (Im - I0) / Kv_SI 
    diffs = np.abs(Qm_vals - Qp_vals)
    idx = np.argmin(diffs)
    if Qp_vals[idx] <= 1e-6 or diffs[idx]/Qp_vals[idx] > 0.02:
        return {"success": False}
    return {"success": True, "rpm": n_vals[idx]*60, "thrust": Ct_v[idx]*RHO*(n_vals[idx]**2)*(D**4), 
            "I_m": Im[idx], "I_b": Ib[idx], "P_shaft": Qp_vals[idx]*(2*np.pi*n_vals[idx]), "P_batt": Ub[idx]*Ib[idx]}

def solve_throttle_for_drag(V_target, D_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt):
    sol_max = find_equilibrium(V_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, 1.0)
    if not sol_max["success"] or sol_max["thrust"] < D_target: return None
    low, high = 0.1, 1.0
    for _ in range(15):
        mid = (low + high) / 2
        sol = find_equilibrium(V_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, mid)
        if not sol["success"] or sol["thrust"] < D_target: low = mid
        else: high = mid
    return sol

# =====================================================##########################
# MAIN LOOP
# =====================================================####################################
df_drag = pd.read_csv(DRAG_FILE)
drag_v, drag_f = df_drag['Velocity'].values, df_drag['Total Drag'].values
all_results = []
id_cnt = 1

for p_name in PROPS:
    df_p = pd.read_csv(os.path.join(PROP_DIR, f"{p_name}.csv"))
    D = float(p_name.split('x')[0]) * 0.0254
    for m_name in MOTORS:
        df_m = pd.read_csv(MOTOR_FILE).set_index('Name').loc[m_name]
        Kv_SI = float(df_m['KV']) * 2 * np.pi / 60
        for b_name in BATTERIES:
            df_b = pd.read_csv(BATTERY_FILE).set_index('Name').loc[b_name]
            V_nom, Q_ah, C_rate = float(df_b['Nominal_Voltage']), float(df_b['Capacity_Ah']), float(df_b['C_Rate'])
            Ib_limit = Q_ah * C_rate
            R_batt = float(df_b['Cells_S']) * ((3.367e-3 * C_rate + 0.4097) / (C_rate * Q_ah))
            
            # Individual masses
            m_prop = PROP_DB[p_name][0]
            m_motor = MOTOR_DB[m_name][0]
            m_batt = BATT_DB[b_name][0]
            m_tot = 1760 # Initial assumptions
            weight_n = (m_tot / 1000.0) * g

            reasons = []
            max_ps, Pcons_max, max_im, max_ib, v_max = 0, 0, 0, 0,0
            lvl_results = []

            # Full Throttle Sweep
            for V in range(0, 41):
                s = find_equilibrium(V, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, df_m['Ri'], df_m['I0'], V_nom, D, R_batt, 1.0)
                if s['success']:
                    max_ps = max(max_ps, s['P_shaft']); max_im = max(max_im, s['I_m']); max_ib = max(max_ib, s['I_b']); Pcons_max = max(Pcons_max, s['P_batt'])
                    dr = np.interp(V, drag_v, drag_f)
                    if s['thrust'] >= dr: v_max = V
                
                dr_lvl = np.interp(V, drag_v, drag_f)
                sol_lvl = solve_throttle_for_drag(V, dr_lvl, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, df_m['Ri'], df_m['I0'], V_nom, D, R_batt)
                if sol_lvl: lvl_results.append((V, sol_lvl['P_batt']))

            if max_ps > df_m['Power_max']: reasons.append(f"Pshaft({max_ps:.1f}W > {df_m['Power_max']}W)")
            if max_im > df_m['Max_Current']: reasons.append(f"Imot({max_im:.1f}A > {df_m['Max_Current']}A)")
            if max_ib > Ib_limit: reasons.append(f"Ibatt({max_ib:.1f}A > {Ib_limit:.1f}A)")
            if v_max < 25: reasons.append(f"Vmax({v_max}m/s < 25m/s)")

            if not lvl_results:
                reasons.append("No Level Flight Possible")
            else:
                v_min = min(lvl_results, key=lambda x: x[1])[0]
                v_climb = max(v_min, V_STALL + 3)
                treq_climb = np.interp(v_climb, drag_v, drag_f) + weight_n * (vz / v_climb)
                s_climb = find_equilibrium(v_climb, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, df_m['Ri'], df_m['I0'], V_nom, D, R_batt, 1.0)
                if not s_climb['success'] or s_climb['thrust'] < treq_climb:
                    t_val = s_climb['thrust'] if s_climb['success'] else 0
                    reasons.append(f"ClimbFail(Tmax@{v_climb}m/s: {t_val:.2f}N < {treq_climb:.2f}N)")

            static_thrust = find_equilibrium(0, df_p['J'].values, df_p['Ct'].values, df_p['Cp'].values, Kv_SI, df_m['Ri'], df_m['I0'], V_nom, D, R_batt, 1.0).get('thrust', 0)

            all_results.append({
                "ID": id_cnt, "Prop": p_name, "Motor": m_name, "Batt": b_name,
                "Mass_Prop_g": m_prop, "Mass_Motor_g": m_motor, "Mass_Batt_g": m_batt,
                "Total_Mass_g": m_tot, 
                "Total_Cost_Eur": PROP_DB[p_name][1]+MOTOR_DB[m_name][1]+BATT_DB[b_name][1],
                "Max_Thrust_N": round(static_thrust, 2), 
                "Max_P_W": round(max_ps, 1), 
                "Pcons_max":round(Pcons_max,1),
                "Status": "WORK" if not reasons else "NOT WORK", 
                "Reason": ", ".join(reasons) if reasons else "OK"
            })
            id_cnt += 1

# ===================================++++++++++++++++++++++++++++++
# SAVING OUTPUT FILES
# =====================================================
df_all = pd.DataFrame(all_results)

# 1. Save all combinations
df_all.to_csv("All_Configurations_Audit.csv", index=False)

# 2. Save only working configurations (Clean list)
df_working = df_all[df_all["Status"] == "WORK"].copy()
# Remove the 'Status' and 'Reason' columns for the clean working CSV 
df_working = df_working.drop(columns=["Status", "Reason"])
df_working.to_csv("Working_Configurations.csv", index=False)

print(f"Audit Complete.")
print(f"Total Working Configs: {len(df_working)}")
print(f"Saved: 'All_Configurations_Audit.csv' and 'Working_Configurations.csv'")