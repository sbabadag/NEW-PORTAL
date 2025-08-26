# Interactive portal-frame N-V-M tool with Eurocode-like snow shaping and simple wind pressure.
# - One-click: compute selected load combinations and plot N-V-M over geometry.
# - Sections: user enters A (cm^2) and I (cm^4) for columns/rafters; label dropdown
#   shows IPE/HEA names on plots (properties are from numeric inputs).
# - Loads:
#    * Dead load G_roof [kN/m^2] (vertical), optional selfweight of rafters from A
#    * Snow via EN 1991-1-3-style μ(α): s = μ * Ce * Ct * s_k  [kN/m^2] (vertical)
#    * Wind as roof-normal pressure p_n [kN/m^2] (+ upward suction, - downward)
# - Conversion to member line load (local):
#    * Vertical area → local transverse: w_local = (q * bay_spacing * cos α) [kN/m]
#    * Roof-normal area → local transverse: w_local = (p_n * bay_spacing) [kN/m]
# - Combinations (illustrative):
#    * ULS (G + S):   1.35*G + 1.50*S
#    * ULS (G + W):   1.35*G + 1.50*W
#    * SLS-char (G+S): 1.00*G + 1.00*S
#    * SLS-char (G+W): 1.00*G + 1.00*W
#
# Notes: This is a simplified educational tool; national annex factors may differ.
#        All charts use matplotlib (no seaborn), one figure per chart, default colors.

import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos
try:
    import ipywidgets as w
    from IPython.display import display, clear_output
    # Check if we're actually in a Jupyter environment
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            IPYW = False  # Not in IPython/Jupyter
        else:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                IPYW = True  # Jupyter notebook
            elif shell == 'TerminalInteractiveShell':
                IPYW = True  # IPython terminal
            else:
                IPYW = False  # Other shells
    except:
        IPYW = False
except Exception:
    IPYW = False

# ---------- FEM core (re-use from earlier) ----------
class Node:
    __slots__ = ("x","y","fix","load")
    def __init__(self, x, y, fix=(False,False,False), load=(0.0,0.0,0.0)):
        self.x = float(x); self.y = float(y)
        self.fix = tuple(bool(b) for b in fix)
        self.load = tuple(float(f) for f in load)

class Element:
    __slots__ = ("i","j","E","A","I","w","label")
    def __init__(self, i, j, E, A, I, w=0.0, label=""):
        self.i = int(i); self.j = int(j)
        self.E = float(E); self.A = float(A); self.I = float(I)
        self.w = float(w)  # local downward is positive (acts along -y_local)
        self.label = str(label)

def dof_ids(nid):
    b = 3*nid
    return [b, b+1, b+2]

def length_cos_sin(n1, n2):
    dx = n2.x - n1.x; dy = n2.y - n1.y
    L = float(np.hypot(dx, dy))
    c = dx/L; s = dy/L
    return L, c, s

def k_local(E,A,I,L):
    EA_L = E*A/L
    EI = E*I
    L2 = L*L; L3 = L2*L
    return np.array([
        [ EA_L,      0,             0,      -EA_L,      0,             0     ],
        [ 0,         12*EI/L3,      6*EI/L2, 0,         -12*EI/L3,     6*EI/L2],
        [ 0,         6*EI/L2,       4*EI/L,  0,         -6*EI/L2,      2*EI/L ],
        [-EA_L,      0,             0,       EA_L,      0,             0     ],
        [ 0,        -12*EI/L3,     -6*EI/L2, 0,          12*EI/L3,    -6*EI/L2],
        [ 0,         6*EI/L2,       2*EI/L,  0,         -6*EI/L2,      4*EI/L ],
    ], dtype=float)

def T_mat(c,s):
    R = np.array([[ c,  s, 0],
                  [-s,  c, 0],
                  [ 0,  0, 1]], dtype=float)
    T = np.zeros((6,6))
    T[:3,:3] = R; T[3:,3:] = R
    return T

def fixed_end_uniform(L, w_down):
    # local +y is "up"; w_down>0 means load downward (-y_local)
    q = -w_down
    return np.array([0.0, q*L/2.0, q*L**2/12.0, 0.0, q*L/2.0, -q*L**2/12.0], dtype=float)

class Frame2D:
    def __init__(self, nodes, elements):
        self.nodes = nodes; self.elements = elements
        self.ndof = 3*len(nodes)

    def assemble(self):
        K = np.zeros((self.ndof, self.ndof), dtype=float)
        F = np.zeros(self.ndof, dtype=float)
        for nid, n in enumerate(self.nodes):
            F[dof_ids(nid)] += np.array(n.load, dtype=float)
        self._cache = []
        for e in self.elements:
            ni, nj = self.nodes[e.i], self.nodes[e.j]
            L, c, s = length_cos_sin(ni, nj)
            k_loc = k_local(e.E, e.A, e.I, L)
            T = T_mat(c, s)
            k_g = T.T @ k_loc @ T
            edofs = dof_ids(e.i) + dof_ids(e.j)
            K[np.ix_(edofs, edofs)] += k_g
            if abs(e.w) > 0:
                F[edofs] -= T.T @ fixed_end_uniform(L, e.w)
            self._cache.append((e,L,c,s,k_loc,T,edofs))
        self.K_full = K; self.F_full = F
        return K,F

    def solve(self):
        K = self.K_full.copy(); F = self.F_full.copy()
        fixed = []
        for nid, n in enumerate(self.nodes):
            for k, is_fixed in enumerate(n.fix):
                if is_fixed: fixed.append(3*nid+k)
        for d in fixed:
            K[d,:]=0; K[:,d]=0; K[d,d]=1.0; F[d]=0.0
        D = np.linalg.solve(K, F)
        R = self.K_full @ D - self.F_full
        self.D = D; self.R = R
        return D, R

    def sample_internal(self, npts=60):
        out = []
        for (e,L,c,s,k_loc,T,edofs) in self._cache:
            d_gl = self.D[edofs]; d_lo = T @ d_gl
            u1,v1,th1,u2,v2,th2 = d_lo
            xi = np.linspace(0,1,npts); x = xi*L
            # axial
            du_dx = (u2-u1)/L
            N = e.E*e.A*du_dx * np.ones_like(x)
            # bending (Hermite)
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = L*(xi - 2*xi**2 + xi**3)
            N3 = 3*xi**2 - 2*xi**3
            N4 = L*(-xi**2 + xi**3)
            # v''
            d2N1_dx2 = (-6 + 12*xi)/(L**2)
            d2N2_dx2 = (-4 + 6*xi)/L
            d2N3_dx2 = ( 6 - 12*xi)/(L**2)
            d2N4_dx2 = (-2 + 6*xi)/L
            M = e.E*e.I*(d2N1_dx2*v1 + d2N2_dx2*th1 + d2N3_dx2*v2 + d2N4_dx2*th2)
            # V = dM/dx
            d3N1_dx3 = 12/(L**3)
            d3N2_dx3 = 6/(L**2)
            d3N3_dx3 = -12/(L**3)
            d3N4_dx3 = 6/(L**2)
            V = e.E*e.I*(d3N1_dx3*v1 + d3N2_dx3*th1 + d3N3_dx3*v2 + d3N4_dx3*th2)

            ni, nj = self.nodes[e.i], self.nodes[e.j]
            X = ni.x + c*x; Y = ni.y + s*x
            nx, ny = -s, c
            out.append({"e":e,"L":L,"x":x,"N":N,"V":V,"M":M,"X":X,"Y":Y,"nx":nx,"ny":ny})
        return out

# ---------- Helpers for loads & plotting ----------
def roof_angles(h1,h2,ridge,span):
    Lhalf = span/2.0
    alpha_L = atan2(ridge - h1, Lhalf)
    alpha_R = atan2(ridge - h2, Lhalf)
    return alpha_L, alpha_R  # radians

def snow_mu_duopitch(alpha_deg):
    # EN 1991-1-3: μ1 = 0.8 for 0–30°; linear to 0 at 60°+
    a = alpha_deg
    if a <= 30.0: return 0.8
    if a >= 60.0: return 0.0
    return 0.8*(60.0 - a)/30.0

def vertical_area_to_local_kNm(q_kNpm2, spacing_m, alpha_rad):
    # vertical area load on roof -> local transverse line load (kN/m)
    return q_kNpm2 * spacing_m * cos(alpha_rad)

def normal_area_to_local_kNm(pn_kNpm2, spacing_m):
    return pn_kNpm2 * spacing_m

def steel_selfweight_kNpm(A_m2):
    # density* g ≈ 78.5 kN/m^3; line load along member (vertical)
    return 78.5 * A_m2

def build_and_run(params, combo_name):
    # Geometry
    E = params["E"]
    span = params["span"]
    h1 = params["h1"]; h2 = params["h2"]; ridge = params["ridge"]
    spacing = params["spacing"]
    # Sections (m^2, m^4)
    A_col = params["A_col"]; I_col = params["I_col"]
    A_raf = params["A_raf"]; I_raf = params["I_raf"]
    label_col = params["label_col"]; label_raf = params["label_raf"]

    # Angles
    aL, aR = roof_angles(h1,h2,ridge,span)
    aLdeg = abs(aL*180/np.pi); aRdeg = abs(aR*180/np.pi)

    # Loads base (kN/m)
    G_kNm2 = params["G_kNm2"]
    include_sw = params["include_selfweight"]
    s_k = params["s_k"]; Ce = params["Ce"]; Ct = params["Ct"]
    pn_kNm2 = params["pn_kNm2"] * (1.0 if params["wind_upward"] else -1.0)  # upward positive

    # Snow on roof surfaces
    muL = snow_mu_duopitch(aLdeg)
    muR = snow_mu_duopitch(aRdeg)
    sL = muL * Ce * Ct * s_k  # kN/m^2
    sR = muR * Ce * Ct * s_k  # kN/m^2

    # Convert to line kN/m
    G_L = vertical_area_to_local_kNm(G_kNm2, spacing, aL)
    G_R = vertical_area_to_local_kNm(G_kNm2, spacing, aR)
    S_L = vertical_area_to_local_kNm(sL, spacing, aL)
    S_R = vertical_area_to_local_kNm(sR, spacing, aR)
    Wn_L = normal_area_to_local_kNm(pn_kNm2, spacing)   # can be ±
    Wn_R = normal_area_to_local_kNm(pn_kNm2, spacing)

    # Self-weight (rafters), vertical kN/m then to local transverse
    SW_line = steel_selfweight_kNpm(A_raf) if include_sw else 0.0
    SW_L = SW_line * cos(aL)
    SW_R = SW_line * cos(aR)

    # Combinations
    if combo_name == "ULS (G+S)":
        wL = 1.35*(G_L + SW_L) + 1.50*S_L
        wR = 1.35*(G_R + SW_R) + 1.50*S_R
        subtitle = f"ULS(G+S)  μL={muL:.2f}, μR={muR:.2f}  αL={aLdeg:.1f}°, αR={aRdeg:.1f}°"
    elif combo_name == "ULS (G+W)":
        wL = 1.35*(G_L + SW_L) + 1.50*(Wn_L)
        wR = 1.35*(G_R + SW_R) + 1.50*(Wn_R)
        subtitle = f"ULS(G+W)  αL={aLdeg:.1f}°, αR={aRdeg:.1f}°"
    elif combo_name == "SLS (G+S)":
        wL = (G_L + SW_L) + (S_L)
        wR = (G_R + SW_R) + (S_R)
        subtitle = f"SLS-char(G+S)  μL={muL:.2f}, μR={muR:.2f}  αL={aLdeg:.1f}°, αR={aRdeg:.1f}°"
    elif combo_name == "SLS (G+W)":
        wL = (G_L + SW_L) + (Wn_L)
        wR = (G_R + SW_R) + (Wn_R)
        subtitle = f"SLS-char(G+W)  αL={aLdeg:.1f}°, αR={aRdeg:.1f}°"
    else:
        raise ValueError("Unknown combination")

    # Build frame (units SI: convert kN/m -> N/m)
    kNpm_to_Npm = 1e3
    nodes = [
        Node(0.0, 0.0, fix=(True,True,True)),
        Node(0.0, h1),
        Node(span/2.0, ridge),
        Node(span, h2),
        Node(span, 0.0, fix=(True,True,True)),
    ]
    elems = [
        Element(0,1,E,A_col,I_col, w=0.0,       label=label_col + " (kolon)"),
        Element(1,2,E,A_raf,I_raf, w=wL*kNpm_to_Npm, label=label_raf + " (sol kiriş)"),
        Element(2,3,E,A_raf,I_raf, w=wR*kNpm_to_Npm, label=label_raf + " (sağ kiriş)"),
        Element(3,4,E,A_col,I_col, w=0.0,       label=label_col + " (kolon)"),
    ]

    model = Frame2D(nodes, elems)
    model.assemble()
    model.solve()
    samples = model.sample_internal(npts=60)
    return nodes, elems, samples, subtitle

def plot_frame_diagram(nodes, elems, samples, diag_key, title, subtitle, unit_text):
    fig = plt.figure(figsize=(9,5.7))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(f"{title}\n{subtitle}")
    # draw members
    for e in elems:
        ni, nj = nodes[e.i], nodes[e.j]
        ax.plot([ni.x, nj.x], [ni.y, nj.y], linewidth=1.4)
        # add label at mid
        xm = (ni.x+nj.x)/2; ym = (ni.y+nj.y)/2
        ax.text(xm, ym, e.label, fontsize=9, ha='center', va='bottom')

    # scale
    if diag_key in ("N","V"):
        conv = 1e-3  # N->kN
    else:
        conv = 1e-3  # N·m -> kN·m
    vals = np.concatenate([np.atleast_1d(np.abs(s[diag_key])*conv) for s in samples])
    vmax = np.max(vals) if vals.size else 1.0
    if vmax == 0: vmax = 1.0
    Lmax = max([s["L"] for s in samples]) if samples else 1.0
    scale = 0.10 * Lmax / vmax

    for s in samples:
        X = s["X"]; Y = s["Y"]
        val = np.atleast_1d(s[diag_key])*conv
        nx, ny = s["nx"], s["ny"]
        Xo = X + scale * val * nx
        Yo = Y + scale * val * ny
        ax.plot(Xo, Yo, linewidth=2.0)
        ax.plot([X[0], Xo[0]], [Y[0], Yo[0]], linewidth=1.0)
        ax.plot([X[-1], Xo[-1]], [Y[-1], Yo[-1]], linewidth=1.0)

    # supports
    ax.plot(nodes[0].x, nodes[0].y, marker='s')
    ax.plot(nodes[-1].x, nodes[-1].y, marker='s')
    ax.text(0.01, 0.02, f"Ölçek: {scale:.3e} m per {unit_text}", transform=ax.transAxes)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True)
    plt.show()
    return fig

# ---------- UI ----------
default_params = {
    "E": 210e9,
    "span": 20.0, "h1": 7.0, "h2": 7.0, "ridge": 8.5,
    "spacing": 6.0,                 # çerçeve aralığı
    "A_col": 0.020, "I_col": 8e-6,  # m^2, m^4
    "A_raf": 0.020, "I_raf": 8e-6,  # m^2, m^4
    "label_col": "HEA ?", "label_raf": "IPE ?",
    "G_kNm2": 0.5,                  # kN/m^2 (örnek: kaplama+asma+mekanik)
    "include_selfweight": True,
    "s_k": 0.8, "Ce": 1.0, "Ct": 1.0,  # snow (kN/m^2)
    "pn_kNm2": 0.3, "wind_upward": True,  # wind pressure normal to roof (kN/m^2)
}

if IPYW:
    # Geometry
    span_w = w.FloatText(value=default_params["span"], description="Açıklık [m]")
    h1_w   = w.FloatText(value=default_params["h1"], description="Sol kolon [m]")
    h2_w   = w.FloatText(value=default_params["h2"], description="Sağ kolon [m]")
    ridge_w= w.FloatText(value=default_params["ridge"], description="Mahya [m]")
    spacing_w = w.FloatText(value=default_params["spacing"], description="Çerçeve aralığı [m]")

    # Sections
    label_col_w = w.Dropdown(options=["HEA 200","HEA 240","HEA 300","HEA 340","Özel"], value="HEA 240", description="Kolon kesit")
    label_raf_w = w.Dropdown(options=["IPE 240","IPE 270","IPE 300","IPE 360","Özel"], value="IPE 300", description="Kiriş kesit")
    Acol_w = w.FloatText(value=default_params["A_col"]*1e4, description="A_kolon [cm²]")
    Icol_w = w.FloatText(value=default_params["I_col"]*1e8, description="I_kolon [cm⁴]")
    Araf_w = w.FloatText(value=default_params["A_raf"]*1e4, description="A_kiriş [cm²]")
    Iraf_w = w.FloatText(value=default_params["I_raf"]*1e8, description="I_kiriş [cm⁴]")

    # Loads
    G_w = w.FloatText(value=default_params["G_kNm2"], description="G (çatı) [kN/m²]")
    sw_chk = w.Checkbox(value=default_params["include_selfweight"], description="Kiriş öz-ağırlık dâhil")
    sk_w = w.FloatText(value=default_params["s_k"], description="s_k (zemin kar) [kN/m²]")
    Ce_w = w.FloatText(value=default_params["Ce"], description="C_e")
    Ct_w = w.FloatText(value=default_params["Ct"], description="C_t")
    pn_w = w.FloatText(value=default_params["pn_kNm2"], description="p_n (çatı-normali) [kN/m²]")
    wind_dir_w = w.ToggleButtons(options=[("Yukarı (+)", True), ("Aşağı (-)", False)], value=True, description="Rüzgar yönü")

    # Combos
    c1 = w.Checkbox(value=True, description="ULS (G+S)")
    c2 = w.Checkbox(value=True, description="ULS (G+W)")
    c3 = w.Checkbox(value=False, description="SLS (G+S)")
    c4 = w.Checkbox(value=False, description="SLS (G+W)")
    go = w.Button(description="Hesapla ve Çiz", button_style="primary")

    info = w.HTML("Hazır. Parametreleri gir ve <b>Hesapla ve Çiz</b> butonuna bas.")

    def on_click(_):
        clear_output(wait=True)
        display(ui)  # redraw controls
        params = {
            "E": 210e9,
            "span": span_w.value, "h1": h1_w.value, "h2": h2_w.value, "ridge": ridge_w.value,
            "spacing": spacing_w.value,
            "A_col": Acol_w.value/1e4, "I_col": Icol_w.value/1e8,
            "A_raf": Araf_w.value/1e4, "I_raf": Iraf_w.value/1e8,
            "label_col": label_col_w.value, "label_raf": label_raf_w.value,
            "G_kNm2": G_w.value, "include_selfweight": sw_chk.value,
            "s_k": sk_w.value, "Ce": Ce_w.value, "Ct": Ct_w.value,
            "pn_kNm2": pn_w.value, "wind_upward": wind_dir_w.value,
        }
        selected = []
        if c1.value: selected.append("ULS (G+S)")
        if c2.value: selected.append("ULS (G+W)")
        if c3.value: selected.append("SLS (G+S)")
        if c4.value: selected.append("SLS (G+W)")
        if not selected:
            print("En az bir kombinasyon seç.")
            return
        # Show roof angles & μ
        aL, aR = roof_angles(params["h1"], params["h2"], params["ridge"], params["span"])
        muL = snow_mu_duopitch(abs(aL*180/np.pi)); muR = snow_mu_duopitch(abs(aR*180/np.pi))
        print(f"Çatı eğimleri: sol={abs(aL*180/np.pi):.2f}°, sağ={abs(aR*180/np.pi):.2f}° | Kar şekil katsayıları μL={muL:.2f}, μR={muR:.2f}\n")

        for name in selected:
            nodes, elems, samples, subtitle = build_and_run(params, name)
            plot_frame_diagram(nodes, elems, samples, "N", f"{name} - Eksenel N(x) [kN]", subtitle, "1 kN")
            plot_frame_diagram(nodes, elems, samples, "V", f"{name} - Kesme V(x) [kN]",  subtitle, "1 kN")
            plot_frame_diagram(nodes, elems, samples, "M", f"{name} - Moment M(x) [kN·m]",subtitle, "1 kN·m")

    go.on_click(on_click)

    ui = w.VBox([
        info,
        w.HTML("<b>Geometri</b>"),
        w.HBox([span_w, h1_w, h2_w, ridge_w, spacing_w]),
        w.HTML("<b>Kesitler</b> (A ve I değerlerini cm² / cm⁴ gir)"),
        w.HBox([label_col_w, Acol_w, Icol_w]),
        w.HBox([label_raf_w, Araf_w, Iraf_w]),
        w.HTML("<b>Yükler</b>"),
        w.HBox([G_w, sw_chk]),
        w.HBox([sk_w, Ce_w, Ct_w]),
        w.HBox([pn_w, wind_dir_w]),
        w.HTML("<b>Kombinasyonlar</b>"),
        w.HBox([c1,c2,c3,c4]),
        go
    ])
    display(ui)
else:
    print("=" * 60)
    print("Portal Çerçeve Analizi - Konsol Versiyonu")
    print("=" * 60)
    print("\nBu araç terminal/konsol ortamında çalışıyor.")
    print("En iyi deneyim için Jupyter Notebook kullanın!")
    print("\nMevcut seçenekler:")
    print("1. Varsayılan parametrelerle analiz yap")
    print("2. Parametreleri değiştir ve analiz yap") 
    print("3. Çıkış")
    
    while True:
        try:
            choice = input("\nSeçiminizi yapın (1-3): ").strip()
            
            if choice == "1":
                print("\nVarsayılan parametrelerle analiz yapılıyor...")
                params = default_params.copy()
                
                # Show parameters
                print(f"\nParametreler:")
                print(f"- Açıklık: {params['span']} m")
                print(f"- Kolon yükseklikleri: {params['h1']}/{params['h2']} m")
                print(f"- Mahya yüksekliği: {params['ridge']} m")
                print(f"- Çerçeve aralığı: {params['spacing']} m")
                print(f"- Ölü yük: {params['G_kNm2']} kN/m²")
                print(f"- Kar yükü: {params['s_k']} kN/m²")
                print(f"- Rüzgar basıncı: {params['pn_kNm2']} kN/m²")
                
                # Run analysis for ULS (G+S)
                nodes, elems, samples, subtitle = build_and_run(params, "ULS (G+S)")
                plot_frame_diagram(nodes, elems, samples, "N", "ULS (G+S) - Eksenel N(x) [kN]", subtitle, "1 kN")
                plot_frame_diagram(nodes, elems, samples, "V", "ULS (G+S) - Kesme V(x) [kN]",  subtitle, "1 kN")
                plot_frame_diagram(nodes, elems, samples, "M", "ULS (G+S) - Moment M(x) [kN·m]",subtitle, "1 kN·m")
                
                input("\nGrafikleri inceledikten sonra devam etmek için Enter'a basın...")
                
            elif choice == "2":
                print("\nParametreleri girin (Enter = varsayılan değer):")
                params = default_params.copy()
                
                # Get geometry parameters
                span_input = input(f"Açıklık [m] ({params['span']}): ").strip()
                if span_input: params['span'] = float(span_input)
                
                h1_input = input(f"Sol kolon yüksekliği [m] ({params['h1']}): ").strip()
                if h1_input: params['h1'] = float(h1_input)
                
                h2_input = input(f"Sağ kolon yüksekliği [m] ({params['h2']}): ").strip()
                if h2_input: params['h2'] = float(h2_input)
                
                ridge_input = input(f"Mahya yüksekliği [m] ({params['ridge']}): ").strip()
                if ridge_input: params['ridge'] = float(ridge_input)
                
                spacing_input = input(f"Çerçeve aralığı [m] ({params['spacing']}): ").strip()
                if spacing_input: params['spacing'] = float(spacing_input)
                
                # Get load parameters
                G_input = input(f"Ölü yük G [kN/m²] ({params['G_kNm2']}): ").strip()
                if G_input: params['G_kNm2'] = float(G_input)
                
                sk_input = input(f"Kar yükü s_k [kN/m²] ({params['s_k']}): ").strip()
                if sk_input: params['s_k'] = float(sk_input)
                
                pn_input = input(f"Rüzgar basıncı p_n [kN/m²] ({params['pn_kNm2']}): ").strip()
                if pn_input: params['pn_kNm2'] = float(pn_input)
                
                # Select combinations
                print("\nHangi kombinasyonları hesaplayalım?")
                uls_gs = input("ULS (G+S)? (y/n) [y]: ").strip().lower()
                uls_gs = uls_gs != 'n'
                
                uls_gw = input("ULS (G+W)? (y/n) [y]: ").strip().lower()
                uls_gw = uls_gw != 'n'
                
                sls_gs = input("SLS (G+S)? (y/n) [n]: ").strip().lower()
                sls_gs = sls_gs == 'y'
                
                sls_gw = input("SLS (G+W)? (y/n) [n]: ").strip().lower()
                sls_gw = sls_gw == 'y'
                
                # Run selected combinations
                combinations = []
                if uls_gs: combinations.append("ULS (G+S)")
                if uls_gw: combinations.append("ULS (G+W)")
                if sls_gs: combinations.append("SLS (G+S)")
                if sls_gw: combinations.append("SLS (G+W)")
                
                if not combinations:
                    print("Hiç kombinasyon seçilmedi!")
                    continue
                
                print(f"\nAnaliz yapılıyor...")
                aL, aR = roof_angles(params["h1"], params["h2"], params["ridge"], params["span"])
                muL = snow_mu_duopitch(abs(aL*180/np.pi)); muR = snow_mu_duopitch(abs(aR*180/np.pi))
                print(f"Çatı eğimleri: sol={abs(aL*180/np.pi):.2f}°, sağ={abs(aR*180/np.pi):.2f}°")
                print(f"Kar şekil katsayıları: μL={muL:.2f}, μR={muR:.2f}")
                
                for combo_name in combinations:
                    print(f"\n{combo_name} hesaplanıyor...")
                    nodes, elems, samples, subtitle = build_and_run(params, combo_name)
                    plot_frame_diagram(nodes, elems, samples, "N", f"{combo_name} - Eksenel N(x) [kN]", subtitle, "1 kN")
                    plot_frame_diagram(nodes, elems, samples, "V", f"{combo_name} - Kesme V(x) [kN]", subtitle, "1 kN")
                    plot_frame_diagram(nodes, elems, samples, "M", f"{combo_name} - Moment M(x) [kN·m]", subtitle, "1 kN·m")
                
                input("\nTüm grafikler tamamlandı. Devam etmek için Enter'a basın...")
                
            elif choice == "3":
                print("Çıkılıyor...")
                break
            else:
                print("Geçersiz seçim! Lütfen 1, 2 veya 3 girin.")
                
        except ValueError as e:
            print(f"Hata: Geçersiz sayı girişi - {e}")
        except KeyboardInterrupt:
            print("\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"Hata oluştu: {e}")
            break
