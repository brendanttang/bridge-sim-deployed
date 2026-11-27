import io
import json
import numpy as np
import matplotlib.pyplot as plt
from algorithm import (x, L, train_loads, envelopes, bridge_parameters, sectional_properties, applied_stress, capacities, factors_of_safety, failure_envelopes, output_plots)

def run(csv_text, load_case):

    #################### cross-section ####################
    if csv_text == "":
        bft, tft, bfb, tfb, hw, tw, bg, tg, sw, a = bridge_parameters(x)
    else:
        param = np.loadtxt(io.StringIO(csv_text), delimiter=",", skiprows=1)
        cols = []
        for i in range(1, 11):
            cols.append(np.interp(x, param[:, 0], param[:, i]))
        bft, tft, bfb, tfb, hw, tw, bg, tg, sw, a = cols

    #################### train ####################
    offsets, loads = train_loads(load_case)
    P = np.sum(loads)
    SFD, BMD, SFDi, BMDi = envelopes(x, offsets, loads, L)

    lead_V = int(np.abs(SFDi).max(axis=1).argmax())
    lead_M = int(BMDi.max(axis=1).argmax())

    offsets2, loads2 = train_loads(load_case, loco_leading=False)
    SFD2, BMD2, SFDi2, BMDi2 = envelopes(x, offsets2, loads2, L)
    SFD = np.maximum(SFD, SFD2)
    BMD = np.maximum(BMD, BMD2)
    SFDi_all = np.vstack([SFDi, SFDi2])
    BMDi_all = np.vstack([BMDi, BMDi2])

    #################### stresses, capacities, FOS at every x ####################
    ybar, ybot, ytop, I, Qcent, Qglue = sectional_properties(bft, tft, bfb, tfb, hw, tw, bg, tg)
    S_top, S_bot, T_cent, T_glue, S_web = applied_stress(SFD, BMD, ybot, ytop, I, Qcent, Qglue, tw, bg, tft)
    S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck = capacities(bft, tft, tw, hw, sw, a, ytop)
    S_tens = np.full(len(x), S_tens)
    S_comp = np.full(len(x), S_comp)
    T_max = np.full(len(x), T_max)
    T_gmax = np.full(len(x), T_gmax)

    with np.errstate(divide="ignore", invalid="ignore"):
        FOS = factors_of_safety(S_top, S_bot, T_cent, T_glue, S_web, S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck)

    #################### the table ####################
    names = ["Tension", "Compression", "Wall shear", "Glue shear", "Flange buckling", "Flange tip buckling", "Web flexural buckling", "Web shear buckling"]
    applied = [S_bot, S_top, T_cent, T_glue, S_top, S_top, S_web, T_cent]
    crit = [S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck]

    table = []
    lowest = []
    for case in range(8):
        fos = np.where(np.isnan(FOS[case]), np.inf, FOS[case])
        i = int(np.argmin(fos))
        lowest.append(float(fos[i]))
        table.append([case + 1, names[case], round(float(applied[case][i]), 3), round(float(crit[case][i]), 3), round(float(fos[i]), 3)])

    worst = 0
    for case in range(8):
        if lowest[case] < lowest[worst]:
            worst = case

    #################### the 6 graphs ###################
    Mf_tens, Mf_comp, Vf_shear, Vf_glue, Mf_buck1, Mf_buck2, Mf_buck3, Vf_buckV = failure_envelopes(S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck, I, ybot, ytop, Qcent, Qglue, tw, bg, tft)
    output_plots(x, SFDi_all, BMDi_all, Mf_tens, Mf_comp, Vf_shear, Vf_glue, Mf_buck1, Mf_buck2, Mf_buck3, Vf_buckV, filename="plots.png")
    plt.close("all") 

    #################### send it back ####################
    answer = {
        "table": table,
        "worst": worst,
        "fail_load": round(lowest[worst] * float(P), 3),
        "offsets": [int(o) for o in offsets],
        "loads": [float(w) for w in loads],
        "lead_V": lead_V,
        "lead_M": lead_M,
        "L": L,
    }
    return json.dumps(answer)
