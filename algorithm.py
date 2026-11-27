import numpy as np
import matplotlib.pyplot as plt

#################### matlab sec. 0 ####################
# 0. Initialize Parameters

L = 1200 
n = 1200 
P = 400
x = np.linspace(0, L, n + 1)

WHEEL_OFFSETS = np.array([0, 176, 340, 516, 680, 856])

E_MAT = 4000.0
NU_MAT = 0.2
SIGMA_T = 30.0
SIGMA_C = 6.0
TAU_M = 4.0
TAU_GLUE = 2.0

SHEET_AREA = 813 * 1016
SHEET_MASS_G = 750.0

#################### matlab sec. 1 ####################
# 1. SFD, BMD under train loading

# returns (wheel_offsets, wheel_loads) for the given load case
# load_case: "LC1", "LC2_base", or "LC2_pass"
def train_loads(load_case, freight_weight=None, loco_leading=True):
    if load_case == "LC1":
        loads = np.full(6, 400 / 6)
    elif load_case == "LC2_base":
        loco, freight = 182.0, 135.0
        loads = np.array([loco, loco, freight, freight, freight, freight]) / 2
    else: # (load_case == "LC2_pass")
        light = freight_weight
        heavy = 1.10 * light
        loco = 1.39 * heavy
        loads = np.array([loco, loco, light, light, heavy, heavy]) / 2

    if not loco_leading:
        loads = loads[::-1]
    return WHEEL_OFFSETS.copy(), loads

# returns position of each wheel whne the leading wheel is at x_lead
def wheel_positions(x_lead, wheel_offsets):
    return x_lead - wheel_offsets

# returns the left and right reaction forces
def reactions(positions, loads, L):
    on = (positions >= 0) & (positions <= L)
    right_reaction = np.sum(loads[on] * positions[on]) / L # moments at A
    left_reaction = np.sum(loads[on]) - right_reaction # forces in y
    return left_reaction, right_reaction

# returns array of shear force V at each x
def sfd(x, positions, loads, L):
    left_reaction, right_reaction = reactions(positions, loads, L)

    V = np.full(len(x), left_reaction)
    for i in range(len(positions)):
        if 0 <= positions[i] <= L:
            V[x >= positions[i]] -= loads[i]
    return V

# returns array of bending moment M vs x
def bmd(x, positions, loads, L):
    left_reaction, right_reaction = reactions(positions, loads, L)

    M = left_reaction * x
    for i in range(len(positions)):
        if 0 <= positions[i] <= L:
            M = M - loads[i] * np.maximum(x - positions[i], 0)
    return M

# returns max V and M versus x
def envelopes(x, wheel_offsets, wheel_loads, L, step=1):
    SFD = np.zeros(len(x))
    BMD = np.zeros(len(x))
    SFDi = []
    BMDi = []

    # now we "walk through" the possible locations
    last = L + max(wheel_offsets)
    for x_lead in range(0, last + 1, step):
        # start location
        positions = wheel_positions(x_lead, wheel_offsets)

        V_right = sfd(x, positions, wheel_loads, L)
        V_left = sfd(x - 1e-9, positions, wheel_loads, L)
        M = bmd(x, positions, wheel_loads, L)

        SFD = np.maximum(SFD, np.abs(V_right))
        SFD = np.maximum(SFD, np.abs(V_left))
        BMD = np.maximum(BMD, M)

        SFDi.append(V_right)
        BMDi.append(M)

    return SFD, BMD, np.array(SFDi), np.array(BMDi)

#################### matlab sec. 2 ####################
# 2. Define Bridge Parameters

def bridge_parameters(x):
    #                = xc,   bft, tft,  bfb,tfb,  hw,    tw,   bg,tg,   sw,    a
    param = np.array([[0,    100, 1.27, 80, 1.27, 73.73, 1.27, 5, 1.27, 78.73, 400],
                      [400,  100, 1.27, 80, 1.27, 73.73, 1.27, 5, 1.27, 78.73, 400],
                      [800,  100, 1.27, 80, 1.27, 73.73, 1.27, 5, 1.27, 78.73, 400],
                      [1200, 100, 1.27, 80, 1.27, 73.73, 1.27, 5, 1.27, 78.73, 400]])
    # x_c   Location, x, of cross-section change
    # bft   Top Flange Width
    # tft   Top Flange Thickness
    # bfb   Bottom Flange Width
    # tfb   Bottom Flange Thickness
    # hw    Web Height (between the bottom and top flanges)
    # tw    Web Thickness
    # bg    Glue Tab Width
    # tg    Glue Tab Thickness
    # sw    Spacing between Webs (centre to centre)
    # a     Diaphragm Spacing

    # Extracting user input assuming linear relationship
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
    return bft, tft, bfb, tfb, hw, tw, bg, tg, sw, a

    # MODIFY THESE PARAMETERS TO TEST OUT DIFFERENT BRIDGES.

#################### matlab sec. 3 ####################
# 3. Calculate Sectional Properties

def sectional_properties(bft, tft, bfb, tfb, hw, tw, bg, tg):
    ybarbot = tfb / 2
    ybarmid = tfb + hw / 2 # remember there are TWO mid rectangles
    ybartop = tfb + hw + tft / 2
    ybartab = tfb + hw - tg / 2 # glue tab

    Abot = bfb * tfb
    Amid = tw * hw # again two mid rectangles
    Atop = bft * tft
    Atab = bg * tg # glue tab

    # ybar. location of centroidal axis from the bottom
    ybar = (Abot * ybarbot + 2 * Amid * ybarmid + Atop * ybartop + 2 * Atab * ybartab) / (Abot + 2 * Amid + Atop + 2 * Atab)

    ybot = ybar
    ytop = tfb + hw + tft - ybar

    Ibot = bfb * tfb ** 3 / 12
    Imid = tw * hw ** 3 / 12
    Itop = bft * tft ** 3 / 12
    Itab = bg * tg ** 3 / 12

    # I
    I = Ibot + Abot * (ybar - ybarbot) ** 2 + 2 * (Imid + Amid * (ybar - ybarmid) ** 2) + Itop + Atop * (ybar - ybartop) ** 2 + 2 * (Itab + Atab * (ybar - ybartab) ** 2)

    # Q at centroidal axes
    Qcent = Atop * (ybartop - ybar) + 2 * Atab * (ybartab - ybar) + tw * (tfb + hw - ybar) ** 2

    # Q at glue location
    Qglue = Atop * (ybartop - ybar)

    return ybar, ybot, ytop, I, Qcent, Qglue

#################### matlab sec. 4 ####################
# 4. Calculate Applied Stress

def applied_stress(SFD, BMD, ybot, ytop, I, Qcent, Qglue, tw, bg, tft):
    S_top = BMD * ytop / I
    S_bot = BMD * ybot / I
    T_cent = SFD * Qcent / (I * 2 * tw)
    T_glue = SFD * Qglue / (I * 2 * (bg + tw))
    S_web = BMD * (ytop - tft) / I
    return S_top, S_bot, T_cent, T_glue, S_web

#################### matlab sec. 5 ####################
# 5. Material and Thin Plate Buckling Capacities

def capacities(bft, tft, tw, hw, sw, a, ytop):
    E = E_MAT
    mu = NU_MAT
    S_tens = SIGMA_T
    S_comp = SIGMA_C
    T_max = TAU_M
    T_gmax = TAU_GLUE
    buck_const = np.pi ** 2 * E / (12 * (1 - mu ** 2))
    S_buck1 = 4 * buck_const * (tft / sw) ** 2
    S_buck2 = 0.425 * buck_const * (2 * tft / (bft - sw)) ** 2
    S_buck3 = 6 * buck_const * (tw / (ytop - tft)) ** 2
    T_buck = 5 * buck_const * ((tw / hw) ** 2 + (tw / a) ** 2)
    return S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck

#################### matlab sec. 6 ####################
# 6. FOS

def factors_of_safety(S_top, S_bot, T_cent, T_glue, S_web, S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck):
    FOS_tens = S_tens / S_bot
    FOS_comp = S_comp / S_top
    FOS_shear = T_max / T_cent
    FOS_glue = T_gmax / T_glue
    FOS_buck1 = S_buck1 / S_top
    FOS_buck2 = S_buck2 / S_top
    FOS_buck3 = S_buck3 / S_web
    FOS_buckV = T_buck / T_cent
    return FOS_tens, FOS_comp, FOS_shear, FOS_glue, FOS_buck1, FOS_buck2, FOS_buck3, FOS_buckV

    # note: S_top is the worst compressive stress at the top face, S_bot is the worst tensile stress at the bottom face

#################### matlab sec. 7 ####################
# 7. Min FOS and the failure load Pfail

def failure_load(FOS_tens, FOS_comp, FOS_shear, FOS_glue,
                 FOS_buck1, FOS_buck2, FOS_buck3, FOS_buckV):
    FOS = {"tension": FOS_tens,
           "compression": FOS_comp,
           "shear": FOS_shear,
           "glue": FOS_glue,
           "buckling in the middle flange": FOS_buck1,
           "buckling in the side flanges": FOS_buck2,
           "buckling in the webs": FOS_buck3,
           "shear buckling": FOS_buckV}
    minFOS = float('inf')
    weakest = None
    for type in FOS:
        m = np.min(FOS[type])
        if m < minFOS:
            minFOS = m
            weakest = type
    Pf = minFOS * P
    x_fail = x[np.argmin(FOS[weakest])]
    
    return minFOS, Pf, weakest, x_fail

    # note: these are not from the matlab skeleton,
    # weakest is a string containing the TYPE of the failure load (ie. "FOS_comp")
    # x is position at which this failure load occurs

#################### matlab sec. 8 ####################
# 8. Vfail and Mfail

def failure_envelopes(S_tens, S_comp, T_max, T_gmax, S_buck1, S_buck2, S_buck3, T_buck, I, ybot, ytop, Qcent, Qglue, tw, bg, tft):
    Mf_tens = S_tens * I / ybot
    Mf_comp = S_comp * I / ytop
    Vf_shear = T_max * I * 2 * tw / Qcent
    Vf_glue = T_gmax * I * 2 * (bg + tw) / Qglue
    Mf_buck1 = S_buck1 * I / ytop
    Mf_buck2 = S_buck2 * I / ytop
    Mf_buck3 = S_buck3 * I / (ytop - tft)
    Vf_buckV = T_buck * I * 2 * tw / Qcent
    return Mf_tens, Mf_comp, Vf_shear, Vf_glue, Mf_buck1, Mf_buck2, Mf_buck3, Vf_buckV

#################### matlab sec. 9 ####################
# 9. Output plots of Vfail and Mfail

# draws a shear subplot
def shear_subplot(pos, x, SFDi, Vf, label, title=None, every=100):
    plt.subplot(2, 3, pos)
    plt.grid(True, alpha=0.3); plt.minorticks_on()
    plt.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.plot(x, Vf, "r", label=label)
    plt.plot(x, -Vf, "r")
    plt.plot(x, SFDi[::every].T, "k", linewidth=0.6)
    plt.plot([0, L], [0, 0], "k", linewidth=2)
    plt.legend(loc="best")
    plt.xlabel("Distance along bridge (mm)")
    plt.ylabel("Shear Force (N)")
    if title:
        plt.title(title, fontweight="bold")

# draws a moment subplot
def moment_subplot(pos, x, BMDi, Mfs, labels, title=None, every=100):
    plt.subplot(2, 3, pos)
    plt.grid(True, alpha=0.3); plt.minorticks_on()
    plt.grid(True, which="minor", linestyle=":", alpha=0.2)
    for Mf, label, colour in zip(Mfs, labels, ["r", "b"]):
        plt.plot(x, Mf, colour, label=label)
    plt.plot(x, BMDi[::every].T, "k", linewidth=0.6)
    plt.plot([0, L], [0, 0], "k", linewidth=2)
    plt.gca().invert_yaxis()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.legend(loc="best")
    plt.xlabel("Distance along bridge (mm)")
    plt.ylabel("Bending Moment (Nmm)")
    if title:
        plt.title(title, fontweight="bold")

# draws the 6 subplots shown below, as required in the assignment instructions
def output_plots(x, SFDi, BMDi, Mf_tens, Mf_comp, Vf_shear, Vf_glue, Mf_buck1, Mf_buck2, Mf_buck3, Vf_buckV, filename=None):
    plt.figure(figsize=(16, 9))

    shear_subplot(1, x, SFDi, Vf_shear, "Matboard Shear Failure")
    shear_subplot(2, x, SFDi, Vf_glue, "Glue Shear Failure", "Shear Force Diagram vs Shear Force Capacities")
    shear_subplot(3, x, SFDi, Vf_buckV, "Matboard Shear Buckling Failure")

    moment_subplot(4, x, BMDi, [Mf_tens, Mf_comp], ["Matboard Tension Failure", "Matboard Compression Failure"])
    moment_subplot(5, x, BMDi, [Mf_buck1, Mf_buck2], ["Matboard Buckling Failure, Top Flange - Mid", "Matboard Buckling Failure, Top Flange - Sides"], "Bending Moment Diagram vs Bending Moment Capacities")
    moment_subplot(6, x, BMDi, [Mf_buck3], ["Matboard Buckling Failure, Webs"])

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()

if __name__ == "__main__":
    pass