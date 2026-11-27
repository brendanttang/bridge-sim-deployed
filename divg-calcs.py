import sys
import numpy as np
from algorithm import (x, L, train_loads, envelopes, sectional_properties, applied_stress, capacities, factors_of_safety)

# usage: python divg-calcs.py attempt1.csv
# the csv has the same columns as param in bridge_parameters():
# xc,bft,tft,bfb,tfb,hw,tw,bg,tg,sw,a
# the cross-section can change along the bridge, so every value is checked at every x
# and the table shows each failure mode at the x where its FOS is smallest

LOAD_CASE = "LC1"  # "LC1" is the same as calcs.py, "LC2_base" is what the design journey uses

#################### cross-section from the csv ####################
param = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
bft = np.interp(x, param[:, 0], param[:, 1])
tft = np.interp(x, param[:, 0], param[:, 2])
bfb = np.interp(x, param[:, 0], param[:, 3])
tfb = np.interp(x, param[:, 0], param[:, 4])
hw = np.interp(x, param[:, 0], param[:, 5])
tw = np.interp(x, param[:, 0], param[:, 6])
bg = np.interp(x, param[:, 0], param[:, 7])
tg = np.interp(x, param[:, 0], param[:, 8])
sw = np.interp(x, param[:, 0], param[:, 9])
a = np.interp(x, param[:, 0], param[:, 10])

#################### worst V and M at every x ####################
# the train runs both ways (only matters for load case 2), keep the worst of the two
offsets, loads = train_loads(LOAD_CASE)
P = np.sum(loads)
SFD, BMD, SFDi, BMDi = envelopes(x, offsets, loads, L)

offsets, loads = train_loads(LOAD_CASE, loco_leading=False)
SFD2, BMD2, SFDi2, BMDi2 = envelopes(x, offsets, loads, L)
SFD = np.maximum(SFD, SFD2)
BMD = np.maximum(BMD, BMD2)

#################### stresses, capacities and FOS at every x ####################
ybar, ybot, ytop, I, Qcent, Qglue = sectional_properties(bft, tft, bfb, tfb, hw, tw, bg, tg)
S_top, S_bot, T_cent, T_glue, S_web = applied_stress(SFD, BMD, ybot, ytop, I, Qcent, Qglue, tw, bg, tft)
S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck = capacities(bft, tft, tw, hw, sw, a, ytop)

# material strengths are single numbers, make them arrays like everything else
S_tens = np.full(len(x), S_tens)
S_comp = np.full(len(x), S_comp)
T_max = np.full(len(x), T_max)
T_gmax = np.full(len(x), T_gmax)

# V and M are 0 at some x (ends of the bridge), which makes FOS infinite there. that's fine, it just never gets picked
with np.errstate(divide="ignore", invalid="ignore"):
    FOS = factors_of_safety(S_top, S_bot, T_cent, T_glue, S_web, S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck)

#################### summary table ####################
names = ["Tension", "Compression", "Wall shear", "Glue shear", "Flange buckling", "Flange tip buckling", "Web flexural buckling", "Web shear buckling"]
applied = [S_bot, S_top, T_cent, T_glue, S_top, S_top, S_web, T_cent]
crit = [S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck]

table = []
for case in range(8):
    fos = np.where(np.isnan(FOS[case]), np.inf, FOS[case])
    i = np.argmin(fos)
    table.append([case + 1, names[case], applied[case][i], crit[case][i], fos[i]])

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
