import pandas as pd
import numpy as np
from pyswarms.discrete import BinaryPSO
import pandapower as pp
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD

# Step 1: Load bus and branch data
bus_df = pd.read_excel("new feeder.xlsx", sheet_name="Bus")
branch_df = pd.read_excel("new feeder.xlsx", sheet_name="Branch")

# Step 2: Identify undervoltage buses
undervoltage_threshold = 95
undervoltage_buses = bus_df[bus_df['Voltage'] < undervoltage_threshold]
candidate_buses = undervoltage_buses['Bus ID'].tolist()

# Step 3: Capacitor options and power base
capacitor_sizes_kvar = [0, 50, 100, 150, 200, 300]
base_voltage_kv = 11
base_power_mva = 1
total_mw = bus_df['MW Loading'].sum()
total_mvar = bus_df['Mvar Loading'].sum()

print("Candidate Buses for Capacitor Placement:")
print(candidate_buses)
print(f"Total Load: {total_mw:.2f} MW / {total_mvar:.2f} MVAR")

# Step 4: Define power flow network model using pandapower
def build_network(bus_df, branch_df, capacitor_vector=None, cap_map=None):
    net = pp.create_empty_network()
    bus_lookup = {}

    for _, row in bus_df.iterrows():
        bus_id = row['Bus ID']
        vn_kv = row['Nominal kV']
        bus_lookup[bus_id] = pp.create_bus(net, vn_kv=vn_kv, name=bus_id)

    source_buses = bus_df[bus_df['Nominal kV'] >= 11]
    if not source_buses.empty:
        src_bus = bus_lookup[source_buses.iloc[0]['Bus ID']]
        pp.create_ext_grid(net, bus=src_bus, vm_pu=1.0, name="Slack")

    for _, row in bus_df.iterrows():
        pp.create_load(net, bus=bus_lookup[row['Bus ID']],
                       p_mw=row['MW Loading'],
                       q_mvar=row['Mvar Loading'],
                       name=f"Load@{row['Bus ID']}")

    for _, row in branch_df.iterrows():
        from_bus = row['Bus 1']
        to_bus = row['Bus 2']
        if from_bus in bus_lookup and to_bus in bus_lookup:
            pp.create_line_from_parameters(
                net,
                from_bus=bus_lookup[from_bus],
                to_bus=bus_lookup[to_bus],
                length_km=1.0,
                r_ohm_per_km=0.3,
                x_ohm_per_km=0.1,
                c_nf_per_km=0,
                max_i_ka=0.2,
                name=row['ID']
            )

    if capacitor_vector is not None:
        for bit, bus_id in zip(capacitor_vector, candidate_buses):
            if bit == 1 and bus_id in bus_lookup:
                kvar = 100 if cap_map is None else cap_map.get(bus_id, 100)
                pp.create_shunt(net,
                                bus=bus_lookup[bus_id],
                                q_mvar=-kvar/1000.0,
                                p_mw=0.0,
                                name=f"Cap@{bus_id}")

    try:
        pp.runpp(net)
    except:
        return 1e6, None, None, None, None, None

    total_loss = net.res_line.pl_mw.sum()
    voltage_deviation = np.sum(abs(net.res_bus.vm_pu - 1.0))
    voltage_violations = np.sum((net.res_bus.vm_pu < 0.95) | (net.res_bus.vm_pu > 1.05))
    violation_penalty = voltage_violations * 10

    return total_loss + voltage_deviation + violation_penalty, net.res_bus, net.res_line, total_loss, voltage_deviation, voltage_violations

# Step 5: PSO fitness function using power flow
def fitness_function(x):
    fitness = []
    for particle in x.astype(int):
        score, _, _, _, _, _ = build_network(bus_df, branch_df, capacitor_vector=particle)
        fitness.append(score)
    return np.array(fitness)

# Step 6: Run Binary PSO optimizer
num_candidates = len(candidate_buses)
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9, 'k': 3, 'p': 2}
b_pso = BinaryPSO(n_particles=20, dimensions=num_candidates, options=options)
best_cost, best_pos = b_pso.optimize(fitness_function, iters=30, verbose=True)

# Step 7: Output best locations
best_pos = best_pos.astype(int)
optimal_locations = [candidate_buses[i] for i in range(num_candidates) if best_pos[i] == 1]

print("\nOptimal Capacitor Locations:")
print(optimal_locations)
print(f"Number of Capacitors Installed: {len(optimal_locations)}")

# Step 8: MILP sizing (fixed with binary indicators)
def optimize_capacitor_sizes(best_pos, cap_options=[0, 50, 100, 150, 200, 300]):
    cap_buses = [bus for i, bus in enumerate(candidate_buses) if best_pos[i] == 1]
    if not cap_buses:
        return {}

    prob = LpProblem("CapacitorSizing", LpMinimize)
    cap_vars = {
        bus: [LpVariable(f"Cap_{bus}_{i}", cat=LpBinary) for i in range(len(cap_options))]
        for bus in cap_buses
    }

    for bus in cap_buses:
        prob += lpSum(cap_vars[bus]) == 1  # Only one size per bus

    total_kvar = lpSum(cap_vars[bus][i] * cap_options[i] for bus in cap_buses for i in range(len(cap_options)))
    prob += total_kvar  # Objective: minimize total kvar installed

    prob.solve(PULP_CBC_CMD(msg=False))

    return {
        bus: sum(cap_options[i] * cap_vars[bus][i].value() for i in range(len(cap_options)))
        for bus in cap_buses
    }

cap_solution = optimize_capacitor_sizes(best_pos)
print("\nOptimal Capacitor Ratings (kVAR):")
for bus, kvar in cap_solution.items():
    print(f"{bus}: {kvar:.0f} kVAR")

# Step 9: Final power flow evaluation with optimal locations and sizes
final_cost, bus_voltages, line_losses, final_loss, final_dev, final_viol = build_network(bus_df, branch_df, best_pos, cap_solution)
print(f"\nFinal Network Cost: {final_cost:.4f}")
print(f"Total Power Loss (MW): {final_loss:.4f}")
print(f"Total Voltage Deviation (p.u.): {final_dev:.4f}")
print(f"Voltage Violations (buses outside 0.95–1.05 p.u.): {final_viol}")
print("\nVoltage Profile:")
print(bus_voltages[['bus', 'vm_pu']])
print("\nLine Losses (MW):")
print(line_losses[['from_bus', 'to_bus', 'pl_mw']])
