import numpy as np
from algorithm import (x, L, P, train_loads, envelopes, bridge_parameters, sectional_properties, applied_stress, capacities, factors_of_safety, E_MAT, NU_MAT, failure_envelopes, output_plots)

#################### worst train positions (load case 1) ####################
offsets, loads = train_loads("LC1")
SFD, BMD, SFDi, BMDi = envelopes(x, offsets, loads, L)

lead_V = np.abs(SFDi).max(axis=1).argmax()
i_V = np.abs(SFDi[lead_V]).argmax()
V = abs(SFDi[lead_V][i_V])

lead_M = BMDi.max(axis=1).argmax()
i_M = BMDi[lead_M].argmax()
M = BMDi[lead_M][i_M]

print(f"Vmax = {V:.3f} N (at x = {x[i_V]} mm, front wheel at {lead_V} mm)")
print(f"Mmax = {M:.3f} N*mm (at x = {x[i_M]} mm, front wheel at {lead_M} mm)")
print()

#################### cross-section (Design 0 uses x = 644 mm) ####################
bft, tft, bfb, tfb, hw, tw, bg, tg, sw, a = (v[i_M] for v in bridge_parameters(x))
ybar, ybot, ytop, I, Qcent, Qglue = sectional_properties(bft, tft, bfb, tfb, hw, tw, bg, tg)
print(f"ybar = {ybar:.3f} mm (from bottom),  ybot = {ybot:.3f} mm,  ytop = {ytop:.3f} mm")
print(f"I = {I:.3f} mm^4,  Qcent = {Qcent:.3f} mm^3,  Qglue = {Qglue:.3f} mm^3")
print()

#################### applied stresses and capacities ####################
S_top, S_bot, T_cent, T_glue, S_web = applied_stress(V, M, ybot, ytop, I, Qcent, Qglue, tw, bg, tft)
S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck = capacities(bft, tft, tw, hw, sw, a, ytop)
K = np.pi ** 2 * E_MAT / (12 * (1 - NU_MAT ** 2))

print(f"1) sigma_bot = {S_bot:.3f} MPa (tension)")
print(f"2) sigma_top = {S_top:.3f} MPa (compression)")
print(f"3) tau_max = {T_cent:.3f} MPa")
print(f"4) tau_glue = {T_glue:.3f} MPa")
print(f"   K = {K:.3f} MPa")
print(f"5) sigma_crit = {S_buck1:.3f} MPa")
print(f"6) sigma_crit = {S_buck2:.3f} MPa")
print(f"7) sigma_crit = {S_buck3:.3f} MPa")
print(f"   sigma_web = {S_web:.3f} MPa")
print(f"8) tau_crit = {T_buck:.3f} MPa")
print()

#################### summary table ####################
# i am 100% aware there is a better way to go about this, don't worry :)
FOS = factors_of_safety(S_top, S_bot, T_cent, T_glue, S_web, S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck)
names = ["Tension", "Compression", "Wall shear", "Glue shear", "Flange buckling", "Flange tip buckling", "Web flexural buckling", "Web shear buckling"]
applied = [S_bot, S_top, T_cent, T_glue, S_top, S_top, S_web, T_cent]
crit = [S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck]

table = []
for case in range(8):
    table.append([case + 1, names[case], applied[case], crit[case], FOS[case]])

header = ["Case", "Failure", "Applied (MPa)", "Crit (MPa)", "FOS"]
line = ""
for word in header:
    word = str(word)
    while len(word) < 24:
        word += " "
    line += word
print(line)

for row in table:
    line = ""
    for value in row:
        if type(value) == str or type(value) == int:
            value = str(value)
        else:
            value = f"{value:.3f}"
        while len(value) < 24:
            value += " "
        line += value
    print(line)

worst = table[0]
for row in table:
    if row[4] < worst[4]:
        worst = row

print()
print(f"Minimum FOS = {worst[4]:.3f} ({worst[1].lower()}), so the failure load is {worst[4] * P:.3f} N")

#################### SFE and BME plots ####################
bft, tft, bfb, tfb, hw, tw, bg, tg, sw, a = bridge_parameters(x)
ybar, ybot, ytop, I, Qcent, Qglue = sectional_properties(bft, tft, bfb, tfb, hw, tw, bg, tg)
S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck = capacities(bft, tft, tw, hw, sw, a, ytop)
Mf_tens, Mf_comp, Vf_shear, Vf_glue, Mf_buck1, Mf_buck2, Mf_buck3, Vf_buckV = failure_envelopes(S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck, I, ybot, ytop, Qcent, Qglue, tw, bg, tft)

output_plots(x, SFDi, BMDi, Mf_tens, Mf_comp, Vf_shear, Vf_glue, Mf_buck1, Mf_buck2, Mf_buck3, Vf_buckV, filename="design0_LC1.png")