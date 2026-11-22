import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = THIS_DIR

# USER INPUTS
TARGET_ID = 20                 # choose configuration ID (1..75)
RHO = 1.157
V_stall = 9.5
vz = 2.0
g = 9.81
Total_mass = 1.76

# Component Database: [Mass(g), Cost(Euro)] 
prop_db = {"8x6": [15, 4], "8x8": [15, 4], "9x6": [19, 4], "9x7.5": [19, 5], "9x9": [19, 5]}
motor_db = {"MotorA": [83, 76], "MotorB": [83, 76], "MotorC": [113, 83], "MotorD": [113, 83], "MotorE": [143, 89]}
batt_db = {"BatteryI": [158, 30], "BatteryII": [167, 23], "BatteryIII": [181, 35]}


# PATHS 
DATA_DIR   = os.path.join(BASE_DIR, "Database_CSV")
PROP_DIR   = DATA_DIR
MOTOR_FILE = os.path.join(DATA_DIR, "Motor_Datas.csv")
BATTERY_FILE = os.path.join(DATA_DIR, "Battery_Database.csv")
DRAG_FILE  = os.path.join(DATA_DIR, "drag_vs_velocity.csv")

OUT_DIR = os.path.join(BASE_DIR, "plots_out")
os.makedirs(OUT_DIR, exist_ok=True)


def require_file(path: str, hint: str = ""):
    if not os.path.exists(path):
        print(f"\n[ERROR] File/Folder not found:\n  {path}")
        if hint:
            print(hint)
        if os.path.isdir(BASE_DIR):
            csvs = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".csv")]
            if csvs:
                print("\n[INFO] CSV files currently in BASE_DIR:")
                for f in csvs:
                    print("  -", f)
        sys.exit(1)

require_file(PROP_DIR, "Expected: ...\\Propeller_Database\\8x6.csv etc.")
require_file(MOTOR_FILE, "Expected motor file: Motor_Datas.csv")
require_file(BATTERY_FILE, "Expected battery file: Battery_Database.csv (rename or update BATTERY_FILE)")


# SYSTEM MAPPING
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
if not selected:
    print(f"[ERROR] TARGET_ID {TARGET_ID} is invalid (should be 1..{len(combinations)}).")
    sys.exit(1)


def get_ct_cp(J_query, J_arr, Ct_arr, Cp_arr):
    Jq = np.clip(J_query, np.min(J_arr), np.max(J_arr))
    Ct = np.interp(Jq, J_arr, Ct_arr)
    Cp = np.interp(Jq, J_arr, Cp_arr)
    return Ct, Cp

# find torque equilibrium for given V and throttle

def find_equilibrium(V_inf, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, throttle):
    rpm_range = np.linspace(1000, 18000, 30000)
    n_vals = rpm_range / 60.0
    omega_vals = 2 * np.pi * n_vals

    # ESC model 
    eta_esc = 0.75 + 0.2 * throttle

    # Prop
    J_vals = V_inf / (n_vals * D)
    Ct_v, Cp_v = get_ct_cp(J_vals, J_arr, Ct_arr, Cp_arr)

    Qp_vals = RHO * Cp_v * (n_vals**2) * (D**5) / (2 * math.pi)
    P_shaft_vals = Qp_vals * omega_vals

    # Battery sag iteration 
    Ub = np.full_like(n_vals, V_nom, dtype=float)
    for _ in range(100):
        U_m = throttle * Ub
        Im = np.maximum((U_m - omega_vals / Kv_SI) / Ri_m, I0)
        Ib = (Im * throttle) / eta_esc
        Ub = V_nom - Ib * R_batt

    # Motor torque model (Drela)
    Qm_vals = (Im - I0) / Kv_SI

    diffs = np.abs(Qm_vals - Qp_vals)
    idx = int(np.argmin(diffs))

    # reject poor matches
    if Qp_vals[idx] <= 1e-9:
        return {"success": False}
    if diffs[idx] / max(Qp_vals[idx], 1e-9) > 0.02:
        return {"success": False}

    n_eq = n_vals[idx]
    omega_eq = omega_vals[idx]

    thrust = Ct_v[idx] * RHO * (n_eq**2) * (D**4)

    Ub_eq = float(Ub[idx])
    Im_eq = float(Im[idx])
    Ib_eq = float(Ib[idx])
    U_m_eq = float(throttle * Ub_eq)

    P_batt = Ub_eq * Ib_eq
    P_motor_in = U_m_eq * Im_eq
    P_shaft = float(P_shaft_vals[idx])

    eta_batt = Ub_eq / V_nom if V_nom > 0 else 0.0
    eta_motor = P_shaft / P_motor_in if P_motor_in > 1e-9 else 0.0
    eta_prop = (thrust * V_inf) / P_shaft if (P_shaft > 1e-9 and V_inf > 0) else 0.0
    eta_overall = (thrust * V_inf) / P_batt if (P_batt > 1e-9 and V_inf > 0) else 0.0

    return {
        "success": True,
        "rpm": float(n_eq * 60),
        "thrust": float(thrust),
        "I_m": Im_eq,
        "I_b": Ib_eq,
        "U_b": Ub_eq,
        "U_m": U_m_eq,
        "P_batt": float(P_batt),
        "P_motor_in": float(P_motor_in),
        "P_shaft": float(P_shaft),
        "throttle": float(throttle),
        "eta_batt": float(eta_batt),
        "eta_esc": float(eta_esc),
        "eta_motor": float(eta_motor),
        "eta_prop": float(eta_prop),
        "eta_overall": float(eta_overall),
    }


# Inverse solve throttle for required thrust at a given V 
def solve_throttle_for_thrust(V_target, T_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt):
    low, high = 0.1, 1.0

    sol_max = find_equilibrium(V_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, 1.0)
    if (not sol_max["success"]) or (sol_max["thrust"] < T_target):
        return None

    sol = None
    for _ in range(28):
        mid = 0.5 * (low + high)
        sol_mid = find_equilibrium(V_target, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, mid)
        if (not sol_mid["success"]) or (sol_mid["thrust"] < T_target):
            low = mid
        else:
            high = mid
            sol = sol_mid
    return sol


# Torque matching & Full-throttle T/D vs V

def _esc_eff(theta: float) -> float:
    return 0.75 + 0.2 * theta

def _calc_I_batt_from_I_motor(I_motor: float, theta: float) -> float:
    if theta <= 1e-9:
        return 0.0
    return theta * I_motor / _esc_eff(theta)

def _iterate_battery_sag(V_nom: float, R_batt: float, Kv_SI: float, Ri_m: float, I0: float,
                         omega: float, theta: float, iters: int = 15):
    U_batt = V_nom
    for _ in range(iters):
        U_motor = theta * U_batt
        I_motor = (U_motor - omega / Kv_SI) / Ri_m
        I_batt = _calc_I_batt_from_I_motor(I_motor, theta)
        U_batt = V_nom - I_batt * R_batt

    U_motor = theta * U_batt
    I_motor = (U_motor - omega / Kv_SI) / Ri_m
    I_batt = _calc_I_batt_from_I_motor(I_motor, theta)
    return U_batt, U_motor, I_motor, I_batt

def _motor_torque(I_motor: float, I0: float, Kv_SI: float):
    if I_motor <= I0:
        return 0.0
    return (I_motor - I0) / Kv_SI

def _prop_at_rpm(V_inf: float, rpm: float, J_arr, Ct_arr, Cp_arr, D: float, rho: float):
    n = rpm / 60.0
    if n <= 1e-12:
        return 0.0, 0.0, np.nan, 0.0
    J = V_inf / (n * D)
    Ct, Cp = get_ct_cp(J, J_arr, Ct_arr, Cp_arr)
    T = Ct * rho * (n**2) * (D**4)
    Q = rho * Cp * (n**2) * (D**5) / (2 * np.pi)
    omega = 2 * np.pi * n
    P_thrust = T * V_inf
    P_shaft = Q * omega
    eta_prop = (P_thrust / P_shaft) if P_shaft > 1e-12 else np.nan
    return T, Q, eta_prop, J

def plot_torque_matching_at_V(
    V_target: float,
    T_required: float,
    theta_req: float,
    J_arr, Ct_arr, Cp_arr,
    Kv_SI: float, Ri_m: float, I0: float,
    V_nom: float, D: float, R_batt: float,
    rho: float,
    out_path: str,
    rpm_min: float = 3000,
    rpm_max: float = 18000,
    n_rpm: int = 400,
):
    rpm_grid = np.linspace(rpm_min, rpm_max, n_rpm)
    Qprop = np.zeros_like(rpm_grid)
    Qmot = np.zeros_like(rpm_grid)

    eta_prop = np.full_like(rpm_grid, np.nan, dtype=float)
    eta_motor = np.full_like(rpm_grid, np.nan, dtype=float)
    eta_drive = np.full_like(rpm_grid, np.nan, dtype=float)

    for i, rpm in enumerate(rpm_grid):
        _, Qp, etap, _ = _prop_at_rpm(V_target, rpm, J_arr, Ct_arr, Cp_arr, D, rho)
        Qprop[i] = Qp
        eta_prop[i] = np.clip(etap, 0, 1) if np.isfinite(etap) else np.nan

        omega = 2 * np.pi * (rpm / 60.0)
        U_batt, U_motor, I_motor, I_batt = _iterate_battery_sag(
            V_nom, R_batt, Kv_SI, Ri_m, I0, omega, theta_req
        )

        Qm = _motor_torque(I_motor, I0, Kv_SI)
        Qmot[i] = Qm

        P_mech = Qm * omega
        P_motor_elec = U_motor * I_motor
        P_batt = U_batt * I_batt

        if P_motor_elec > 1e-12 and P_mech > 0:
            eta_motor[i] = np.clip(P_mech / P_motor_elec, 0, 1)
        if P_batt > 1e-12 and P_mech > 0:
            eta_drive[i] = np.clip(P_mech / P_batt, 0, 1)

    idx_eq = int(np.argmin(np.abs(Qmot - Qprop)))
    rpm_eq = float(rpm_grid[idx_eq])

    fig, (axQ, axE) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axQ.plot(rpm_grid, Qprop, linewidth=2.5, label="Propeller torque")
    axQ.plot(rpm_grid, Qmot, linewidth=2.5, label="Motor torque")
    axQ.axvline(rpm_eq, linestyle="--", linewidth=1.5, label=f"Equilibrium ≈ {rpm_eq:.0f} rpm")
    axQ.set_ylabel("Torque [N·m]")
    axQ.set_title(f"Torque matching @ V={V_target:.1f} m/s (θ≈{theta_req:.3f}, Treq≈{T_required:.2f} N)")
    axQ.grid(True, alpha=0.3)
    axQ.legend()
    axQ.set_ylim(bottom=0)

    eta_esc_line = np.full_like(rpm_grid, _esc_eff(theta_req), dtype=float)
    axE.plot(rpm_grid, eta_prop, linewidth=2.5, label="Propeller")
    axE.plot(rpm_grid, eta_motor, linewidth=2.5, label="Motor")
    axE.plot(rpm_grid, eta_esc_line, linewidth=2.5, label="ESC (η=0.75+0.2θ)")
    axE.plot(rpm_grid, eta_drive, linewidth=2.5, linestyle="--", label="Battery→Shaft")
    axE.axvline(rpm_eq, linestyle="--", linewidth=1.5)
    axE.set_xlabel("RPM")
    axE.set_ylabel("Efficiency [-]")
    axE.set_ylim(0, 1.0)
    axE.grid(True, alpha=0.3)
    axE.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_full_throttle_TD_vs_V(
    J_arr, Ct_arr, Cp_arr,
    Kv_SI: float, Ri_m: float, I0: float,
    V_nom: float, D: float, R_batt: float,
    rho: float,
    drag_velocity, drag_values,
    Vstall: float,
    Vdrag_max: float,
    out_path: str,
    V_max_search: float = 43.0,
):
    V_scan = np.linspace(0.0, V_max_search, 200)

    V_ok, T_ok = [], []
    for V in V_scan:
        s = find_equilibrium(V, J_arr, Ct_arr, Cp_arr, Kv_SI, Ri_m, I0, V_nom, D, R_batt, 1.0)
        if s.get("success", False):
            V_ok.append(V)
            T_ok.append(s["thrust"])

    V_ok = np.asarray(V_ok, dtype=float)
    T_ok = np.asarray(T_ok, dtype=float)

    mask_drag = (drag_velocity >= Vstall) & (drag_velocity <= Vdrag_max)
    V_for_D = np.asarray(drag_velocity[mask_drag], dtype=float)
    D_for_D = np.asarray(drag_values[mask_drag], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(V_ok, T_ok, linewidth=2.5, label="Thrust @ full throttle (θ=1)")
    if V_for_D.size:
        plt.plot(V_for_D, D_for_D, linewidth=2.5, linestyle="-.", label="Drag")

    plt.axvline(Vstall, linestyle="--", linewidth=1.5, label=f"V_stall = {Vstall:.1f} m/s")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Airspeed V [m/s]")
    plt.ylabel("Force [N]")
    plt.title("Full-throttle Thrust / Drag vs V")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# Load databases
prop_path = os.path.join(PROP_DIR, f"{selected['Prop']}.csv")
require_file(prop_path, f"Expected prop file: {selected['Prop']}.csv in Propeller_Database")

df_p = pd.read_csv(prop_path)

df_motor_table = pd.read_csv(MOTOR_FILE)
df_batt_table = pd.read_csv(BATTERY_FILE)

if "Name" not in df_motor_table.columns:
    print("[ERROR] Motor_Datas.csv must contain column 'Name'. Current columns:", df_motor_table.columns.tolist())
    sys.exit(1)
if "Name" not in df_batt_table.columns:
    print("[ERROR] Battery_Database.csv must contain column 'Name'. Current columns:", df_batt_table.columns.tolist())
    sys.exit(1)

df_m = df_motor_table.set_index("Name").loc[selected["Motor"]]
df_b = df_batt_table.set_index("Name").loc[selected["Batt"]]


# Load Drag curve from CSV (from Database_CSV)
require_file(DRAG_FILE, "Expected drag file: drag_vs_velocity.csv")


df_drag = pd.read_csv(DRAG_FILE)

# expected columns: velocity [m/s], total drag [N]
drag_velocity = df_drag.iloc[:, 0].to_numpy(dtype=float)
drag_values   = df_drag.iloc[:, 1].to_numpy(dtype=float)


mask = np.isfinite(drag_velocity) & np.isfinite(drag_values)
drag_velocity = drag_velocity[mask]
drag_values   = drag_values[mask]

idx = np.argsort(drag_velocity)
drag_velocity = drag_velocity[idx]
drag_values   = drag_values[idx]



# Convert parameters
D = float(selected["Prop"].split("x")[0]) * 0.0254  # inch to m (diameter)
Kv_SI = float(df_m["KV"]) * 2 * np.pi / 60          # rpm/V -> rad/s/V
Ri_m = float(df_m["Ri"])
I0 = float(df_m["I0"])

V_nom = float(df_b["Nominal_Voltage"])
Q_ah = float(df_b["Capacity_Ah"])
C_rate = float(df_b["C_Rate"])
Cells_S = float(df_b["Cells_S"])


# battery internal resistance model 
R_batt = Cells_S * ((3.367e-3 * C_rate + 0.4097) / (C_rate * Q_ah))


# Compute full-throttle envelope
full_throttle_rows = []

for V in range(0, 41):
    sol = find_equilibrium(
        V,
        df_p["J"].values,
        df_p["Ct"].values,
        df_p["Cp"].values,
        Kv_SI,
        Ri_m,
        I0,
        V_nom,
        D,
        R_batt,
        1.0   # full throttle
    )
    dr = float(np.interp(V, drag_velocity, drag_values))

    if sol["success"]:
        full_throttle_rows.append({**sol, "V": V, "Drag": dr})
    else:
        full_throttle_rows.append({"V": V, "Drag": dr, "thrust": np.nan})

df_ft = pd.DataFrame(full_throttle_rows)


# FIGURE 1 : Torque matching at V=20 
V_TORQUE = 20.0
TREQ_VT = float(np.interp(V_TORQUE, drag_velocity, drag_values))

sol_vt = solve_throttle_for_thrust(
    V_TORQUE, TREQ_VT,
    df_p["J"].values, df_p["Ct"].values, df_p["Cp"].values,
    Kv_SI, Ri_m, I0, V_nom, D, R_batt
)
if sol_vt:
    theta_req = float(sol_vt["throttle"])
    torque_fig_path = os.path.join(OUT_DIR, f"TorqueMatch_V{int(V_TORQUE)}_ID{TARGET_ID}.png")
    plot_torque_matching_at_V(
        V_target=V_TORQUE,
        T_required=TREQ_VT,
        theta_req=theta_req,
        J_arr=df_p["J"].values, Ct_arr=df_p["Ct"].values, Cp_arr=df_p["Cp"].values,
        Kv_SI=Kv_SI, Ri_m=Ri_m, I0=I0,
        V_nom=V_nom, D=D, R_batt=R_batt,
        rho=RHO,
        out_path=torque_fig_path,
        rpm_min=5800, rpm_max=8000, n_rpm=600
    )
else:
    print(f"[WARN] No throttle solution at V={V_TORQUE} m/s. Skip torque-matching plot.")


# FIGURE 2 : Full-throttle T/D vs V  
VSTALL = 9.58
VDRAG_MAX = 35.0
td_fig_path = os.path.join(OUT_DIR, f"TD_vs_V_FullThrottle_ID{TARGET_ID}.png")

plot_full_throttle_TD_vs_V(
    J_arr=df_p["J"].values, Ct_arr=df_p["Ct"].values, Cp_arr=df_p["Cp"].values,
    Kv_SI=Kv_SI, Ri_m=Ri_m, I0=I0,
    V_nom=V_nom, D=D, R_batt=R_batt,
    rho=RHO,
    drag_velocity=drag_velocity, drag_values=drag_values,
    Vstall=VSTALL, Vdrag_max=VDRAG_MAX,
    out_path=td_fig_path,
    V_max_search=43.0
)



# Plot 3: Overall efficiency vs V (3 configurations)

COMPARE_IDS = [20, 26, 36]
V_MARKS = [12, 13, 20]

# Display mapping for legend 
DISPLAY_ID_MAP = {
    20: "ID1",
    26: "ID2",
    36: "ID3",
}

plt.figure()

component_points = []   # (cfg_id, color, Vmk, dict_of_component_eta)
rows_export = []        # for CSV output

def get_selected_by_id(cfg_id):
    return next((item for item in combinations if item["ID"] == cfg_id), None)

for cfg_id in COMPARE_IDS:
    sel = get_selected_by_id(cfg_id)
    if sel is None:
        continue

    disp_id = DISPLAY_ID_MAP.get(cfg_id, f"ID{cfg_id}")

    # load prop / motor / battery for this cfg
    df_p_i = pd.read_csv(os.path.join(PROP_DIR, f"{sel['Prop']}.csv"))

    m_i = df_motor_table.set_index("Name").loc[sel["Motor"]]
    b_i = df_batt_table.set_index("Name").loc[sel["Batt"]]

    D_i = float(sel["Prop"].split("x")[0]) * 0.0254
    Kv_SI_i = float(m_i["KV"]) * 2 * np.pi / 60
    Ri_m_i = float(m_i["Ri"])
    I0_i = float(m_i["I0"])

    V_nom_i = float(b_i["Nominal_Voltage"])
    Q_ah_i = float(b_i["Capacity_Ah"])
    C_rate_i = float(b_i["C_Rate"])
    Cells_S_i = float(b_i["Cells_S"])
    R_batt_i = Cells_S_i * ((3.367e-3 * C_rate_i + 0.4097) / (C_rate_i * Q_ah_i))

    # 1) Overall efficiency curve 
    rows = []
    for V in drag_velocity:
        dr = float(np.interp(V, drag_velocity, drag_values))
        sol = solve_throttle_for_thrust(
            V, dr,
            df_p_i["J"].values, df_p_i["Ct"].values, df_p_i["Cp"].values,
            Kv_SI_i, Ri_m_i, I0_i, V_nom_i, D_i, R_batt_i
        )
        if sol:
            rows.append({"V": float(V), "eta_overall": float(sol["eta_overall"])})

    df_line = pd.DataFrame(rows)
    if df_line.empty:
        continue

    label_line = disp_id
    line_obj, = plt.plot(df_line["V"], df_line["eta_overall"], label=label_line)
    cfg_color = line_obj.get_color()

    # 2) Mark component efficiencies
    for Vmk in V_MARKS:
        dr_mk = float(np.interp(Vmk, drag_velocity, drag_values))
        sol_mk = solve_throttle_for_thrust(
            Vmk, dr_mk,
            df_p_i["J"].values, df_p_i["Ct"].values, df_p_i["Cp"].values,
            Kv_SI_i, Ri_m_i, I0_i, V_nom_i, D_i, R_batt_i
        )
        if sol_mk:
            component_points.append((
                cfg_id,
                cfg_color,
                Vmk,
                {
                    "eta_batt": float(sol_mk["eta_batt"]),
                    "eta_esc": float(sol_mk["eta_esc"]),
                    "eta_motor": float(sol_mk["eta_motor"]),
                    "eta_prop": float(sol_mk["eta_prop"]),
                }
            ))

            # export data
            rows_export.append({
                "cfg_id": cfg_id,
                "display_id": disp_id,
                "V_mps": float(Vmk),
                "rpm": float(sol_mk["rpm"]),
                "I_battery_A": float(sol_mk["I_b"]),
                "P_battery_W": float(sol_mk["P_batt"]),
            })


# 3) Plot component markers
marker_map = {
    "eta_batt": "o",
    "eta_esc": "s",
    "eta_motor": "^",
    "eta_prop": "D",
}
component_name_map = {
    "eta_batt": "Battery",
    "eta_esc": "ESC",
    "eta_motor": "Motor",
    "eta_prop": "Prop",
}

shown_components = set()
for cfg_id, cfg_color, Vmk, r in component_points:
    for key, mk in marker_map.items():
        label = component_name_map[key] if key not in shown_components else None
        plt.plot(
            [Vmk], [r[key]],
            marker=mk,
            linestyle="None",
            markersize=8,
            color=cfg_color,
            markerfacecolor=cfg_color,
            markeredgecolor=cfg_color,
            label=label
        )
    shown_components.update(marker_map.keys())

# 4) Vertical reference lines
V_LABELS = {12: "Take-off V=12m/s", 13: "Cruise V=13m/s", 20: "Mission V=20m/s"}
for Vmk in V_MARKS:
    plt.axvline(Vmk, linestyle="--", linewidth=1.0)
    plt.text(Vmk + 0.2, plt.ylim()[0] + 0.02, V_LABELS.get(Vmk, f"V={Vmk}"),
             rotation=90, va="bottom", fontsize=9)

plt.xlabel("Airspeed V [m/s]")
plt.ylabel("Overall efficiency [-]")
plt.title("Overall Efficiency vs V \nComponent efficiencies at V = 12 / 13 / 20")
plt.grid(True)
plt.legend()

eff_comp_path = os.path.join(
    OUT_DIR,
    f"OverallEff_ID1_ID2_ID3_V12_13_20.png"
)
plt.savefig(eff_comp_path, dpi=200, bbox_inches="tight")
plt.show()

print(" Plot 3 saved to:")
print(" ", eff_comp_path)


# 5) Export rpm, Ibattery, Pbattery at V=12/13/20
df_export = pd.DataFrame(rows_export).sort_values(["cfg_id", "V_mps"])
export_path = os.path.join(OUT_DIR, "V12_13_20_rpm_Ibatt_Pbatt_ID20_26_36.csv")
df_export.to_csv(export_path, index=False)

print(" Export saved to:")
print(" ", export_path)
