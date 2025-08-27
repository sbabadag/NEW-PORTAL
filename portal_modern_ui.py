# Modern Portal Frame Analysis Tool with CustomTkinter
# Interactive portal-frame N-V-M tool with modern UI and integrated graphics display

import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend for better compatibility

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import customtkinter as ctk
from math import atan2, cos
import tkinter as tk
from tkinter import ttk
import threading
from tkinter import messagebox

# Import steel section checking functions
from steel_check import check_section_resistance, format_detailed_check_results, get_section_properties, optimize_section_selection, format_optimization_results, get_hea_sections, get_ipe_sections, get_all_sections, optimize_portal_frame_total_weight, check_deflections

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Import FEM core functions
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core FEM classes and functions from portal.py
class Node:
    __slots__ = ("x","y","fix","load","D")
    def __init__(self, x, y, fix=(False,False,False), load=(0.0,0.0,0.0)):
        self.x = float(x); self.y = float(y)
        self.fix = tuple(bool(b) for b in fix)
        self.load = tuple(float(f) for f in load)
        self.D = None  # Will be set after analysis

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

def calculate_haunch_properties(base_I, base_height, height_increase):
    """
    Calculate moment of inertia increase due to haunch
    Simple approach: assume rectangular section height increase
    """
    # For IPE sections, approximate enhanced I based on height increase
    # This is a simplified approach - in practice would need more detailed calculation
    
    # Simplified approach: increase by ratio of (new_height/old_height)^3
    # This is based on I being proportional to h³ for rectangular sections
    height_ratio = (base_height + height_increase) / base_height
    I_enhanced = base_I * (height_ratio ** 3)
    
    return I_enhanced

def build_and_run(params, combo_name):
    # Import steel section properties
    from steel_check import get_section_properties
    
    # Geometry
    E = params["E"]
    span = params["span"]
    h1 = params["h1"]; h2 = params["h2"]; ridge = params["ridge"]
    spacing = params["spacing"]
    
    # Get real section properties from database
    column_section = params["column_section"]
    beam_section = params["beam_section"]
    
    col_props = get_section_properties(column_section)
    beam_props = get_section_properties(beam_section)
    
    if col_props is None or beam_props is None:
        print(f"ERROR: Section properties not found for {column_section} or {beam_section}")
        # Fallback to manual values
        A_col = params["A_col"]; I_col = params["I_col"]
        A_raf = params["A_raf"]; I_raf = params["I_raf"]
    else:
        # Use real section properties (convert to SI units)
        steel_density = 7850  # kg/m³
        A_col = col_props["G"] / steel_density  # m² from weight (kg/m)
        I_col = col_props["I"] * 1e-8  # cm⁴ to m⁴
        A_raf = beam_props["G"] / steel_density  # m² from weight (kg/m)
        I_raf = beam_props["I"] * 1e-8  # cm⁴ to m⁴
        
        print(f"DEBUG - Gerçek kesit özellikleri:")
        print(f"  Column {column_section}: A={A_col:.6f} m², I={I_col:.2e} m⁴ (DB: I={col_props['I']} cm⁴)")
        print(f"  Beam {beam_section}: A={A_raf:.6f} m², I={I_raf:.2e} m⁴ (DB: I={beam_props['I']} cm⁴)")
        
        # Apply haunch enhancement if enabled
        haunch_enable = params.get("haunch_enable", False)
        if haunch_enable:
            haunch_height_increase = params.get("haunch_height_increase", 0.2)  # meters
            base_height = beam_props.get("h", 300) / 1000.0  # Convert mm to m
            I_raf_enhanced = calculate_haunch_properties(I_raf, base_height, haunch_height_increase)
            print(f"  Haunch aktif - Beam I: {I_raf*1e8:.1f} cm⁴ → {I_raf_enhanced*1e8:.1f} cm⁴ (artış: {(I_raf_enhanced/I_raf-1)*100:.1f}%)")
            I_raf = I_raf_enhanced  # Use enhanced value for analysis
        
        # Sanity check: IPE 300 should have I ≈ 8356 cm⁴ = 8.356e-5 m⁴
        # Real world IPE 300: I = 8356 cm⁴
        # If this is too low, we might need to check our reference data
        if beam_section == "IPE 300":
            print(f"  IPE 300 beklenen atalet momenti: 8356 cm⁴ = 8.356e-5 m⁴")
            print(f"  Hesaplanan: {I_raf:.2e} m⁴")
    
    # Sections (m^2, m^4)
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
    try:
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
            
        print(f"  Sol kiriş (wL): {wL:.2f} kN/m -> {wL*1000:.0f} N/m")
        print(f"  Sağ kiriş (wR): {wR:.2f} kN/m -> {wR*1000:.0f} N/m")
        
    except Exception as e:
        print(f"HATA: Yük hesaplama: {e}")
        print(f"G_L={G_L}, SW_L={SW_L}, S_L={S_L}")
        print(f"G_R={G_R}, SW_R={SW_R}, S_R={S_R}")
        raise

    # Mesnet tipine göre fix değerlerini belirle
    left_fix = (True, True, True) if params.get("left_support_type", "Sabit") == "Sabit" else (True, True, False)
    right_fix = (True, True, True) if params.get("right_support_type", "Sabit") == "Sabit" else (True, True, False)

    # Debug: Yük hesaplamaları
    print(f"\nDEBUG: Yük hesaplamaları ({combo_name}):")
    print(f"Kar yükü (s_k): {s_k} kN/m²")
    print(f"Rüzgar yükü (pn_kNm2): {params['pn_kNm2']} kN/m²") 
    print(f"Ölü yük (G_kNm2): {G_kNm2} kN/m²")
    print(f"Sol açı (aL): {aLdeg:.1f}°, Sağ açı (aR): {aRdeg:.1f}°")
    print(f"Kar katsayıları - μL: {muL:.2f}, μR: {muR:.2f}")
    print(f"Kar yükleri - sL: {sL:.2f} kN/m², sR: {sR:.2f} kN/m²")
    print(f"Dağıtılmış yükler:")
    
    # Build frame (units SI: convert kN/m -> N/m)
    kNpm_to_Npm = 1e3
    nodes = [
        Node(0.0, 0.0, fix=left_fix),
        Node(0.0, h1),
        Node(span/2.0, ridge),
        Node(span, h2),
        Node(span, 0.0, fix=right_fix),
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
    
    # Add displacement data to nodes for deflection checking
    for i, node in enumerate(nodes):
        node.D = model.D[i*3:(i+1)*3]  # Extract 3 DOFs for each node (x, y, rotation)
    
    samples = model.sample_internal(npts=60)
    return nodes, elems, samples, subtitle

class ModernPortalAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Portal Çerçeve Analizi - Modern UI")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Default parameters
        self.default_params = {
            "E": 210e9,
            "span": 20.0, "h1": 7.0, "h2": 7.0, "ridge": 8.5,
            "spacing": 6.0,
            "A_col": 0.020, "I_col": 8e-6,
            "A_raf": 0.020, "I_raf": 8e-6,
            "label_col": "HEA 240", "label_raf": "IPE 300",
            "G_kNm2": 0.5,
            "include_selfweight": True,
            "s_k": 0.8, "Ce": 1.0, "Ct": 1.0,
            "pn_kNm2": 0.3, "wind_upward": True,
            "haunch_enable": False,
            "haunch_length": 1.5,
            "haunch_height_increase": 0.2,
        }
        
        # Create UI
        self.create_sidebar()
        self.create_main_frame()
        
        # Initialize with default values
        self.load_defaults()
    
    def create_sidebar(self):
        """Create the left sidebar with input controls"""
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(20, weight=1)
        
        # Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Portal Çerçeve Analizi",
                                      font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))
        
        # Geometry Section
        self.geometry_label = ctk.CTkLabel(self.sidebar_frame, text="Geometri",
                                          font=ctk.CTkFont(size=16, weight="bold"))
        self.geometry_label.grid(row=1, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Mesnet Tipi Seçiciler
        self.left_support_label = ctk.CTkLabel(self.sidebar_frame, text="Sol Mesnet Tipi:")
        self.left_support_label.grid(row=2, column=0, padx=(20, 10), pady=(0, 0), sticky="w")
        self.left_support_combo = ctk.CTkComboBox(self.sidebar_frame, values=["Sabit", "Mafsallı"], width=100)
        self.left_support_combo.grid(row=2, column=1, padx=(0, 20), pady=(0, 0), sticky="w")
        self.left_support_combo.set("Sabit")

        self.right_support_label = ctk.CTkLabel(self.sidebar_frame, text="Sağ Mesnet Tipi:")
        self.right_support_label.grid(row=3, column=0, padx=(20, 10), pady=(0, 0), sticky="w")
        self.right_support_combo = ctk.CTkComboBox(self.sidebar_frame, values=["Sabit", "Mafsallı"], width=100)
        self.right_support_combo.grid(row=3, column=1, padx=(0, 20), pady=(0, 0), sticky="w")
        self.right_support_combo.set("Sabit")

        # Span
        self.span_label = ctk.CTkLabel(self.sidebar_frame, text="Açıklık [m]:")
        self.span_label.grid(row=4, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.span_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.span_entry.grid(row=4, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Left column height
        self.h1_label = ctk.CTkLabel(self.sidebar_frame, text="Sol kolon [m]:")
        self.h1_label.grid(row=5, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.h1_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.h1_entry.grid(row=5, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Right column height
        self.h2_label = ctk.CTkLabel(self.sidebar_frame, text="Sağ kolon [m]:")
        self.h2_label.grid(row=6, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.h2_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.h2_entry.grid(row=6, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Ridge height
        self.ridge_label = ctk.CTkLabel(self.sidebar_frame, text="Mahya [m]:")
        self.ridge_label.grid(row=7, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.ridge_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.ridge_entry.grid(row=7, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Frame spacing
        self.spacing_label = ctk.CTkLabel(self.sidebar_frame, text="Çerçeve aralığı [m]:")
        self.spacing_label.grid(row=8, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.spacing_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.spacing_entry.grid(row=8, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Sections
        self.sections_label = ctk.CTkLabel(self.sidebar_frame, text="Kesitler",
                                          font=ctk.CTkFont(size=16, weight="bold"))
        self.sections_label.grid(row=9, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Column section
        self.col_section_label = ctk.CTkLabel(self.sidebar_frame, text="Kolon kesit:")
        self.col_section_label.grid(row=10, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        
        # Get all sections from database (HEA + IPE for flexibility)
        all_sections = get_all_sections()
        all_sections.append("Özel")  # Add custom option
        
        self.col_section_combo = ctk.CTkComboBox(self.sidebar_frame, 
                                               values=all_sections,
                                               width=120,
                                               command=self.update_column_properties)
        self.col_section_combo.grid(row=10, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Column area
        self.Acol_label = ctk.CTkLabel(self.sidebar_frame, text="A_kolon [cm²]:")
        self.Acol_label.grid(row=11, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.Acol_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.Acol_entry.grid(row=11, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Column moment of inertia
        self.Icol_label = ctk.CTkLabel(self.sidebar_frame, text="I_kolon [cm⁴]:")
        self.Icol_label.grid(row=12, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.Icol_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.Icol_entry.grid(row=12, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Rafter section
        self.raf_section_label = ctk.CTkLabel(self.sidebar_frame, text="Kiriş kesit:")
        self.raf_section_label.grid(row=13, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        
        # Get all sections from database (IPE + HEA for flexibility)
        all_sections_beam = get_all_sections()
        all_sections_beam.append("Özel")  # Add custom option
        
        self.raf_section_combo = ctk.CTkComboBox(self.sidebar_frame,
                                               values=all_sections_beam,
                                               width=120,
                                               command=self.update_beam_properties)
        self.raf_section_combo.grid(row=13, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Rafter area
        self.Araf_label = ctk.CTkLabel(self.sidebar_frame, text="A_kiriş [cm²]:")
        self.Araf_label.grid(row=14, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.Araf_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.Araf_entry.grid(row=14, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Rafter moment of inertia
        self.Iraf_label = ctk.CTkLabel(self.sidebar_frame, text="I_kiriş [cm⁴]:")
        self.Iraf_label.grid(row=15, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.Iraf_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.Iraf_entry.grid(row=15, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Haunch parameters
        self.haunch_label = ctk.CTkLabel(self.sidebar_frame, text="Haunch (Konsol)",
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.haunch_label.grid(row=16, column=0, columnspan=2, padx=20, pady=(15, 5), sticky="w")
        
        # Haunch enable checkbox
        self.haunch_enable = ctk.CTkCheckBox(self.sidebar_frame, text="Haunch uygula",
                                           command=self.toggle_haunch_controls)
        self.haunch_enable.grid(row=17, column=0, columnspan=2, padx=20, pady=(5, 0), sticky="w")
        
        # Haunch length
        self.haunch_length_label = ctk.CTkLabel(self.sidebar_frame, text="Haunch uzunluk [m]:")
        self.haunch_length_label.grid(row=18, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.haunch_length_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.haunch_length_entry.grid(row=18, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        self.haunch_length_entry.insert(0, "1.5")  # Default 1.5m
        
        # Haunch height increase
        self.haunch_height_label = ctk.CTkLabel(self.sidebar_frame, text="Yükseklik artışı [cm]:")
        self.haunch_height_label.grid(row=19, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.haunch_height_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.haunch_height_entry.grid(row=19, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        self.haunch_height_entry.insert(0, "20")  # Default 20cm increase
        
        # Loads
        self.loads_label = ctk.CTkLabel(self.sidebar_frame, text="Yükler",
                                       font=ctk.CTkFont(size=16, weight="bold"))
        self.loads_label.grid(row=20, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Dead load
        self.G_label = ctk.CTkLabel(self.sidebar_frame, text="G (çatı) [kN/m²]:")
        self.G_label.grid(row=21, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.G_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.G_entry.grid(row=21, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Self-weight checkbox
        self.sw_check = ctk.CTkCheckBox(self.sidebar_frame, text="Kiriş öz-ağırlık dahil")
        self.sw_check.grid(row=22, column=0, columnspan=2, padx=20, pady=(10, 0), sticky="w")
        
        # Snow load
        self.sk_label = ctk.CTkLabel(self.sidebar_frame, text="s_k (kar) [kN/m²]:")
        self.sk_label.grid(row=23, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.sk_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.sk_entry.grid(row=23, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Wind load
        self.pn_label = ctk.CTkLabel(self.sidebar_frame, text="p_n (rüzgar) [kN/m²]:")
        self.pn_label.grid(row=24, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.pn_entry = ctk.CTkEntry(self.sidebar_frame, width=100)
        self.pn_entry.grid(row=24, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        
        # Wind direction
        self.wind_check = ctk.CTkCheckBox(self.sidebar_frame, text="Rüzgar yukarı (+)")
        self.wind_check.grid(row=25, column=0, columnspan=2, padx=20, pady=(10, 0), sticky="w")
        
        # Steel grade selection
        self.steel_grade_label = ctk.CTkLabel(self.sidebar_frame, text="Çelik sınıfı:")
        self.steel_grade_label.grid(row=26, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.steel_grade_combo = ctk.CTkComboBox(self.sidebar_frame,
                                               values=["S235", "S275", "S355"],
                                               width=120)
        self.steel_grade_combo.grid(row=26, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        self.steel_grade_combo.set("S275")  # Default value
        
        # Design code selection
        self.design_code_label = ctk.CTkLabel(self.sidebar_frame, text="Hesaplama yöntemi:")
        self.design_code_label.grid(row=27, column=0, padx=(20, 10), pady=(10, 0), sticky="w")
        self.design_code_combo = ctk.CTkComboBox(self.sidebar_frame,
                                               values=["Eurocode 3 (EN)", "TS 648 (Türk)", "ÇYTHYE (Türk)", "TBDY 2018 (Türk)"],
                                               width=140)
        self.design_code_combo.grid(row=27, column=1, padx=(0, 20), pady=(10, 0), sticky="w")
        self.design_code_combo.set("Eurocode 3 (EN)")  # Default value
        
        # Load combinations
        self.combos_label = ctk.CTkLabel(self.sidebar_frame, text="Kombinasyonlar",
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.combos_label.grid(row=28, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        self.uls_gs_check = ctk.CTkCheckBox(self.sidebar_frame, text="ULS (G+S)")
        self.uls_gs_check.grid(row=29, column=0, padx=(20, 10), pady=(5, 0), sticky="w")
        
        self.uls_gw_check = ctk.CTkCheckBox(self.sidebar_frame, text="ULS (G+W)")
        self.uls_gw_check.grid(row=29, column=1, padx=(0, 20), pady=(5, 0), sticky="w")
        
        self.sls_gs_check = ctk.CTkCheckBox(self.sidebar_frame, text="SLS (G+S)")
        self.sls_gs_check.grid(row=30, column=0, padx=(20, 10), pady=(5, 0), sticky="w")
        
        self.sls_gw_check = ctk.CTkCheckBox(self.sidebar_frame, text="SLS (G+W)")
        self.sls_gw_check.grid(row=30, column=1, padx=(0, 20), pady=(5, 0), sticky="w")
        
        # Calculate button
        self.calc_button = ctk.CTkButton(self.sidebar_frame, text="Hesapla ve Çiz",
                                        command=self.calculate_analysis,
                                        font=ctk.CTkFont(size=16, weight="bold"),
                                        height=40)
        self.calc_button.grid(row=31, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        
        # Steel verification button
        self.verify_btn = ctk.CTkButton(self.sidebar_frame, text="Tahkik Et", 
                                       command=self.verify_sections, 
                                       fg_color="darkgreen", 
                                       hover_color="green",
                                       font=ctk.CTkFont(size=14, weight="bold"))
        self.verify_btn.grid(row=32, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        
        # Optimization button
        self.optimize_btn = ctk.CTkButton(self.sidebar_frame, text="Optimize Et", 
                                         command=self.optimize_sections, 
                                         fg_color="orange", 
                                         hover_color="darkorange",
                                         font=ctk.CTkFont(size=14, weight="bold"))
        self.optimize_btn.grid(row=33, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        
        # Status label
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Hazır")
        self.status_label.grid(row=34, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="ew")
    
    def toggle_haunch_controls(self):
        """Toggle haunch parameter visibility based on checkbox state"""
        if self.haunch_enable.get():
            # Show haunch controls
            self.haunch_length_label.grid()
            self.haunch_length_entry.grid()
            self.haunch_height_label.grid()
            self.haunch_height_entry.grid()
        else:
            # Hide haunch controls
            self.haunch_length_label.grid_remove()
            self.haunch_length_entry.grid_remove()
            self.haunch_height_label.grid_remove()
            self.haunch_height_entry.grid_remove()
    
    def create_main_frame(self):
        """Create the main frame with graphics display"""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(20, 20), pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Initially show welcome message
        self.welcome_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.welcome_frame, text="Hoş Geldiniz")
        
        welcome_label = ctk.CTkLabel(self.welcome_frame, 
                                   text="Portal Çerçeve Analizi\n\nSol panelden parametreleri girin ve\n'Hesapla ve Çiz' butonuna basın.",
                                   font=ctk.CTkFont(size=18))
        welcome_label.pack(expand=True)
    
    def load_defaults(self):
        """Load default values into the UI"""
        self.span_entry.insert(0, str(self.default_params["span"]))
        self.h1_entry.insert(0, str(self.default_params["h1"]))
        self.h2_entry.insert(0, str(self.default_params["h2"]))
        self.ridge_entry.insert(0, str(self.default_params["ridge"]))
        self.spacing_entry.insert(0, str(self.default_params["spacing"]))
        
        self.col_section_combo.set(self.default_params["label_col"])
        self.raf_section_combo.set(self.default_params["label_raf"])
        
        self.Acol_entry.insert(0, str(self.default_params["A_col"]*1e4))
        self.Icol_entry.insert(0, str(self.default_params["I_col"]*1e8))
        self.Araf_entry.insert(0, str(self.default_params["A_raf"]*1e4))
        self.Iraf_entry.insert(0, str(self.default_params["I_raf"]*1e8))
        
        self.G_entry.insert(0, str(self.default_params["G_kNm2"]))
        self.sw_check.select() if self.default_params["include_selfweight"] else self.sw_check.deselect()
        self.sk_entry.insert(0, str(self.default_params["s_k"]))
        self.pn_entry.insert(0, str(self.default_params["pn_kNm2"]))
        self.wind_check.select() if self.default_params["wind_upward"] else self.wind_check.deselect()
        
        # Haunch defaults
        if self.default_params["haunch_enable"]:
            self.haunch_enable.select()
        else:
            self.haunch_enable.deselect()
        self.haunch_length_entry.insert(0, str(self.default_params["haunch_length"]))
        self.haunch_height_entry.insert(0, str(self.default_params["haunch_height_increase"]*100))  # Convert m to cm for display
        self.toggle_haunch_controls()  # Set initial visibility
        
        # Default combinations
        self.uls_gs_check.select()
        self.uls_gw_check.select()
    
    def update_column_properties(self, selected_section):
        """Update column A and I when section is selected"""
        if selected_section != "Özel":
            try:
                props = get_section_properties(selected_section)
                self.Acol_entry.delete(0, "end")
                self.Acol_entry.insert(0, f"{props['A']*1e4:.1f}")
                self.Icol_entry.delete(0, "end")
                self.Icol_entry.insert(0, f"{props['I']*1e8:.0f}")
            except:
                pass  # Keep existing values if section not found
    
    def update_beam_properties(self, selected_section):
        """Update beam A and I when section is selected"""
        if selected_section != "Özel":
            try:
                props = get_section_properties(selected_section)
                self.Araf_entry.delete(0, "end")
                self.Araf_entry.insert(0, f"{props['A']*1e4:.1f}")
                self.Iraf_entry.delete(0, "end")
                self.Iraf_entry.insert(0, f"{props['I']*1e8:.0f}")
            except:
                pass  # Keep existing values if section not found
    
    def get_parameters(self):
        """Get parameters from UI"""
        try:
            print(f"DEBUG: get_parameters çağrıldı")
            print(f"DEBUG: Araf_entry değeri: '{self.Araf_entry.get()}'")
            print(f"DEBUG: Iraf_entry değeri: '{self.Iraf_entry.get()}'")
            print(f"DEBUG: Acol_entry değeri: '{self.Acol_entry.get()}'")
            print(f"DEBUG: Icol_entry değeri: '{self.Icol_entry.get()}'")
            
            params = {
                "E": 210e9,
                "span": float(self.span_entry.get()),
                "h1": float(self.h1_entry.get()),
                "h2": float(self.h2_entry.get()),
                "ridge": float(self.ridge_entry.get()),
                "spacing": float(self.spacing_entry.get()),
                "A_col": float(self.Acol_entry.get())/1e4,
                "I_col": float(self.Icol_entry.get())/1e8,
                "A_raf": float(self.Araf_entry.get())/1e4,
                "I_raf": float(self.Iraf_entry.get())/1e8,
                "label_col": self.col_section_combo.get(),
                "label_raf": self.raf_section_combo.get(),
                "beam_section": self.raf_section_combo.get(),
                "column_section": self.col_section_combo.get(),
                "G_kNm2": float(self.G_entry.get()),
                "include_selfweight": self.sw_check.get(),
                "s_k": float(self.sk_entry.get()),
                "Ce": 1.0,
                "Ct": 1.0,
                "pn_kNm2": float(self.pn_entry.get()),
                "wind_upward": self.wind_check.get(),
                "left_support_type": self.left_support_combo.get(),
                "right_support_type": self.right_support_combo.get(),
                "haunch_enable": self.haunch_enable.get(),
                "haunch_length": float(self.haunch_length_entry.get()) if self.haunch_enable.get() else 1.5,
                "haunch_height_increase": float(self.haunch_height_entry.get())/100.0 if self.haunch_enable.get() else 0.2,  # Convert cm to m
            }
            print(f"DEBUG: Parametreler başarıyla oluşturuldu")
            return params
        except ValueError as e:
            print(f"DEBUG: get_parameters ValueError: {e}")
            self.status_label.configure(text=f"Hata: Geçersiz sayı girişi - {e}")
            return None
        except Exception as e:
            print(f"DEBUG: get_parameters Exception: {e}")
            self.status_label.configure(text=f"Hata: {e}")
            return None
    
    def get_selected_combinations(self):
        """Get selected load combinations"""
        combinations = []
        if self.uls_gs_check.get():
            combinations.append("ULS (G+S)")
        if self.uls_gw_check.get():
            combinations.append("ULS (G+W)")
        if self.sls_gs_check.get():
            combinations.append("SLS (G+S)")
        if self.sls_gw_check.get():
            combinations.append("SLS (G+W)")
        return combinations
    
    def calculate_analysis(self):
        """Run the analysis and display results"""
        # Clear existing tabs
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        
        # Get parameters
        params = self.get_parameters()
        if params is None:
            return
        
        combinations = self.get_selected_combinations()
        if not combinations:
            self.status_label.configure(text="En az bir kombinasyon seçin!")
            return
        
        self.status_label.configure(text="Hesaplama yapılıyor...")
        self.update()
        
        # Run analysis in a separate thread to prevent UI freezing
        thread = threading.Thread(target=self.run_analysis, args=(params, combinations))
        thread.daemon = True
        thread.start()
    
    def run_analysis(self, params, combinations):
        """Run the analysis in a separate thread"""
        try:
            # Show roof angles
            aL, aR = roof_angles(params["h1"], params["h2"], params["ridge"], params["span"])
            muL = snow_mu_duopitch(abs(aL*180/np.pi))
            muR = snow_mu_duopitch(abs(aR*180/np.pi))
            
            info_text = f"Çatı eğimleri: sol={abs(aL*180/np.pi):.2f}°, sağ={abs(aR*180/np.pi):.2f}°\n"
            info_text += f"Kar şekil katsayıları: μL={muL:.2f}, μR={muR:.2f}"
            
            # Store current analysis for verification
            max_moment_beam = 0
            max_moment_column = 0
            max_axial_column = 0
            
            # Run each combination to find maximum forces
            for combo_name in combinations:
                nodes, elems, samples, subtitle = build_and_run(params, combo_name)
                
                # Get maximum forces from this combination
                for sample in samples:
                    N_vals = np.atleast_1d(sample['N']) * 1e-3  # Convert to kN
                    M_vals = np.atleast_1d(sample['M']) * 1e-3  # Convert to kN⋅m
                    
                    max_moment_beam = max(max_moment_beam, np.max(np.abs(M_vals)))
                    max_moment_column = max(max_moment_column, np.max(np.abs(M_vals)))
                    max_axial_column = max(max_axial_column, np.max(np.abs(N_vals)))
            
            self.current_analysis = {
                'params': params,
                'combinations': combinations,
                'max_moment_beam': max_moment_beam,
                'max_moment_column': max_moment_column,
                'max_axial_column': max_axial_column
            }
            
            # Create results for each combination
            for combo_name in combinations:
                nodes, elems, samples, subtitle = build_and_run(params, combo_name)
                
                # Schedule UI update in main thread
                self.after(0, self.create_results_tab, combo_name, nodes, elems, samples, subtitle, info_text)
            
            self.after(0, lambda: self.status_label.configure(text="Analiz tamamlandı!"))
            
        except Exception as e:
            error_msg = f"Hata: {str(e)}"
            self.after(0, lambda msg=error_msg: self.status_label.configure(text=msg))
            
    def create_results_tab(self, combo_name, nodes, elems, samples, subtitle, info_text):
        """Create a tab with results for one combination"""
        # Create tab frame
        tab_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(tab_frame, text=combo_name)
        
        # Configure grid
        tab_frame.grid_rowconfigure(0, weight=1)
        tab_frame.grid_columnconfigure(0, weight=1)
        
        # Create scrollable frame for content
        scroll_frame = ctk.CTkScrollableFrame(tab_frame, width=800, height=600)
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add info text
        info_label = ctk.CTkLabel(scroll_frame, text=info_text, 
                                 font=ctk.CTkFont(size=12))
        info_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Create matplotlib figures
        try:
            self.create_diagram_plots(scroll_frame, nodes, elems, samples, combo_name, subtitle)
        except Exception as e:
            error_label = ctk.CTkLabel(scroll_frame, 
                                     text=f"Grafik çizim hatası: {str(e)}", 
                                     text_color="red")
            error_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
            
    def create_diagram_plots(self, parent_frame, nodes, elems, samples, combo_name, subtitle):
        """Create N-V-M diagrams in a 2x2 grid"""
        
        # Create main figure with subplots
        fig = Figure(figsize=(12, 8))
        fig.suptitle(f"{combo_name}\n{subtitle}", fontsize=14, fontweight='bold')
        
        # Create 2x2 subplot layout
        ax1 = fig.add_subplot(2, 2, 1)  # Geometry
        ax2 = fig.add_subplot(2, 2, 2)  # Normal forces
        ax3 = fig.add_subplot(2, 2, 3)  # Shear forces  
        ax4 = fig.add_subplot(2, 2, 4)  # Moments
        
        # Plot geometry
        self.plot_geometry(ax1, nodes, elems)
        ax1.set_title("Geometri")
        
        # Plot diagrams
        self.plot_diagram(ax2, nodes, elems, samples, "N", "Normal Kuvvet [kN]", "1 kN")
        self.plot_diagram(ax3, nodes, elems, samples, "V", "Kesme Kuvveti [kN]", "1 kN") 
        self.plot_diagram(ax4, nodes, elems, samples, "M", "Moment [kN⋅m]", "1 kN⋅m")
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas and add to parent frame
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, 
                                   sticky="nsew", padx=5, pady=5)
        
    def plot_geometry(self, ax, nodes, elems):
        """Plot frame geometry"""
        ax.set_aspect('equal', adjustable='datalim')
        
        # Draw members
        for e in elems:
            ni, nj = nodes[e.i], nodes[e.j]
            ax.plot([ni.x, nj.x], [ni.y, nj.y], 'k-', linewidth=2)
            
            # Add element label at midpoint
            xm = (ni.x + nj.x) / 2
            ym = (ni.y + nj.y) / 2
            ax.text(xm, ym, e.label, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Draw supports
        for i, node in enumerate(nodes):
            if any(node.fix):
                ax.plot(node.x, node.y, 's', markersize=8, color='red')
                ax.text(node.x, node.y-0.3, f'N{i}', ha='center', va='top', fontsize=8)
            else:
                ax.plot(node.x, node.y, 'o', markersize=6, color='blue')
                ax.text(node.x, node.y-0.3, f'N{i}', ha='center', va='top', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
    def plot_diagram(self, ax, nodes, elems, samples, diag_key, title, unit_text):
        """Plot N, V, or M diagram"""
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(title)
        
        # Draw frame geometry (light lines)
        for e in elems:
            ni, nj = nodes[e.i], nodes[e.j]
            ax.plot([ni.x, nj.x], [ni.y, nj.y], 'k-', linewidth=1, alpha=0.3)
        
        # Convert units
        if diag_key in ("N", "V"):
            conv = 1e-3  # N to kN
        else:
            conv = 1e-3  # N⋅m to kN⋅m
            
        # Find max value for scaling
        all_vals = []
        for s in samples:
            vals = np.atleast_1d(np.abs(s[diag_key])) * conv
            all_vals.extend(vals.flatten())
            
        if all_vals:
            vmax = max(all_vals)
        else:
            vmax = 1.0
            
        if vmax == 0:
            vmax = 1.0
            
        # Scale factor
        Lmax = max([s["L"] for s in samples]) if samples else 1.0
        scale = 0.15 * Lmax / vmax
        
        # Plot diagrams
        max_val = 0
        max_pos = None
        
        for s in samples:
            X = s["X"]
            Y = s["Y"] 
            val = np.atleast_1d(s[diag_key]) * conv
            nx, ny = s["nx"], s["ny"]
            
            # Offset points for diagram
            Xo = X + scale * val * nx
            Yo = Y + scale * val * ny
            
            # Plot diagram line
            ax.plot(Xo, Yo, 'b-', linewidth=2)
            
            # Connect to member
            ax.plot([X[0], Xo[0]], [Y[0], Yo[0]], 'b-', linewidth=1)
            ax.plot([X[-1], Xo[-1]], [Y[-1], Yo[-1]], 'b-', linewidth=1)
            
            # Find peak values for labeling
            abs_vals = np.abs(val)
            local_max_idx = np.argmax(abs_vals)
            if abs_vals[local_max_idx] > max_val:
                max_val = abs_vals[local_max_idx]
                max_pos = (Xo[local_max_idx], Yo[local_max_idx], val[local_max_idx])
        
        # Add peak value label
        if max_pos is not None:
            x_peak, y_peak, val_peak = max_pos
            ax.plot(x_peak, y_peak, 'ro', markersize=6)
            ax.text(x_peak, y_peak, f'{val_peak:.1f}', 
                   fontsize=10, fontweight='bold', 
                   ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8))
        
        # Draw supports
        for node in nodes:
            if any(node.fix):
                ax.plot(node.x, node.y, 's', markersize=6, color='red')
                
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        # Add scale info
        scale_text = f"Ölçek: {scale:.3e} m per {unit_text}"
        ax.text(0.02, 0.98, scale_text, transform=ax.transAxes, 
               fontsize=8, va='top',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
    def verify_sections(self):
        """Verify steel sections according to Eurocode 3"""
        if not hasattr(self, 'current_analysis'):
            self.status_label.configure(text="Önce analiz yapın!")
            return
            
        try:
            self.status_label.configure(text="Tahkikler yapılıyor...")
            self.update()
            
            analysis = self.current_analysis
            steel_grade = self.steel_grade_combo.get()
            design_code = self.design_code_combo.get()
            
            # Get yield strength
            fy = {"S235": 235, "S275": 275, "S355": 355}[steel_grade]
            
            # Show verification results window
            self.show_verification_results(analysis, fy, steel_grade, design_code)
            
            self.status_label.configure(text="Tahkikler tamamlandı")
            
        except Exception as e:
            self.status_label.configure(text=f"Tahkik hatası: {str(e)}")
            
    def show_verification_results(self, analysis, fy, steel_grade, design_code):
        """Show verification results in a new window"""
        
        # Create new window
        verify_window = ctk.CTkToplevel(self)
        verify_window.title(f"Çelik Tahkik Sonuçları - {steel_grade} - {design_code}")
        verify_window.geometry("900x700")
        
        # Create text widget with scrollbar
        text_frame = ctk.CTkFrame(verify_window)
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        text_widget = ctk.CTkTextbox(text_frame, width=860, height=640)
        text_widget.pack(fill="both", expand=True)
        
        # Get current sections from UI (not from stored analysis)
        current_beam_section = self.raf_section_combo.get()
        current_column_section = self.col_section_combo.get()
        
        # Add verification content
        content = f"ÇELIK KESİT TAHKİK SONUÇLARI\n"
        content += f"{'='*60}\n\n"
        content += f"Tasarım Kodu: {design_code}\n"
        content += f"Çelik Sınıfı: {steel_grade} (fy = {fy} MPa)\n"
        content += f"Kiriş Kesiti: {current_beam_section}\n"
        content += f"Kolon Kesiti: {current_column_section}\n\n"
        
        # Check each combination
        overall_safe = True
        
        for combo_name in analysis['combinations']:
            content += f"{combo_name} Kombinasyonu:\n"
            content += f"{'─'*40}\n"
            
            try:
                # Build and analyze this combination
                nodes, elems, samples, subtitle = build_and_run(analysis['params'], combo_name)
                
                # Get maximum forces from samples
                max_N = 0
                max_V = 0 
                max_M = 0
                
                for sample in samples:
                    N_vals = np.atleast_1d(sample['N']) * 1e-3  # Convert to kN
                    V_vals = np.atleast_1d(sample['V']) * 1e-3  # Convert to kN
                    M_vals = np.atleast_1d(sample['M']) * 1e-3  # Convert to kN⋅m
                    
                    max_N = max(max_N, np.max(np.abs(N_vals)))
                    max_V = max(max_V, np.max(np.abs(V_vals)))
                    max_M = max(max_M, np.max(np.abs(M_vals)))
                
                # Check beam section (only moment, no axial load) - use current UI selection
                beam_check = check_section_resistance(current_beam_section, fy, max_M, 0, design_code)
                
                # Check column section (moment + axial load) - use current UI selection
                column_check = check_section_resistance(current_column_section, fy, max_M, max_N, design_code)
                
                # Add results to content
                content += f"Maksimum İç Kuvvetler:\n"
                content += f"  Normal Kuvvet (N): {max_N:.2f} kN\n"
                content += f"  Kesme Kuvveti (V): {max_V:.2f} kN\n"
                content += f"  Moment (M): {max_M:.2f} kNm\n\n"
                
                # Format and add results to content with detailed formulas
                content += f"{'═'*50}\n"
                content += f"DETAYLI KESİT TAHKİK SONUÇLARI\n"
                content += f"{'═'*50}\n\n"
                
                # Add detailed beam check results
                content += format_detailed_check_results(beam_check)
                content += f"\n{'═'*50}\n\n"
                
                # Add detailed column check results  
                content += format_detailed_check_results(column_check)
                content += f"\n{'═'*50}\n\n"
                
                # Check if this combination is safe
                combo_safe = beam_check['safety'] and column_check['safety']
                
                # Add deflection check for all combinations
                # Get frame geometry from analysis parameters
                span = analysis['params']['span']
                h1 = analysis['params']['h1'] 
                h2 = analysis['params']['h2']
                
                # Check deflections
                deflection_check = check_deflections(nodes, span, h1, h2, design_code)
                
                content += f"DETAYLI SEHİM KONTROLÜ\n"
                content += f"{'═'*50}\n\n"
                
                if 'error' in deflection_check:
                    content += f"⚠ Hata: {deflection_check['error']}\n"
                    combo_safe = False
                else:
                    content += f"Tasarım Kodu: {deflection_check['design_code']} ({deflection_check['standard_ref']})\n"
                    content += f"Çerçeve Geometrisi: Açıklık = {span:.1f} m, Ortalama Yükseklik = {(h1+h2)/2:.1f} m\n\n"
                    
                    # Detailed vertical deflection check
                    if 'detailed_calculations' in deflection_check:
                        v_calc = deflection_check['detailed_calculations']['vertical']
                        content += f"1. DÜŞEY SEHİM KONTROLÜ:\n"
                        content += f"  Formül: {v_calc['formula']}\n"
                        content += f"  Hesaplama: {v_calc['calculation']}\n"
                        content += f"  Ölçülen: {v_calc['measured']}\n"
                        content += f"  Kontrol: {v_calc['check']}\n"
                        content += f"  Durum: {v_calc['status']}\n"
                        content += f"  Kullanım Oranı: {v_calc['utilization']:.3f}\n"
                        content += f"  Referans: {v_calc['reference']}\n\n"
                        
                        h_calc = deflection_check['detailed_calculations']['horizontal']
                        content += f"2. YATAY SEHİM KONTROLÜ:\n"
                        content += f"  Formül: {h_calc['formula']}\n"
                        content += f"  Hesaplama: {h_calc['calculation']}\n"
                        content += f"  Ölçülen: {h_calc['measured']}\n"
                        content += f"  Kontrol: {h_calc['check']}\n"
                        content += f"  Durum: {h_calc['status']}\n"
                        content += f"  Kullanım Oranı: {h_calc['utilization']:.3f}\n"
                        content += f"  Referans: {h_calc['reference']}\n\n"
                    
                    # Update combination safety with deflection results
                    combo_safe = combo_safe and deflection_check['overall_ok']
                    
                content += f"{'═'*50}\n\n"
                
                overall_safe = overall_safe and combo_safe
                
                content += f"\n{combo_name} Sonucu: {'✓ GÜVENLİ' if combo_safe else '✗ GÜVENSİZ'}\n\n"
                
            except Exception as e:
                content += f"Hata: {str(e)}\n\n"
                overall_safe = False
        
        # Overall result
        content += f"{'='*60}\n"
        if overall_safe:
            content += f"GENEL SONUÇ: ✓ TÜM KONTROLLER GÜVENLİ\n"
            content += f"Seçilen kesitler tüm yük kombinasyonları için mukavemet ve sehim açısından yeterlidir.\n"
        else:
            content += f"GENEL SONUÇ: ✗ BİR VEYA DAHA FAZLA KONTROL GÜVENSİZ\n"
            content += f"Kesit boyutları artırılmalı, çelik sınıfı yükseltilmeli veya sehim limitleri gözden geçirilmelidir.\n"
        content += f"{'='*60}\n"
        
        # Insert content into text widget
        text_widget.insert("0.0", content)
        text_widget.configure(state="disabled")

    def optimize_sections(self):
        """Run optimization to find the lightest steel sections for total frame weight"""
        if not hasattr(self, 'current_analysis') or not self.current_analysis:
            self.status_label.configure(text="Önce analiz yapın!")
            return
        
        try:
            # Get current parameters
            steel_grade = self.steel_grade_combo.get()
            design_code = self.design_code_combo.get()
            fy = {"S235": 235, "S275": 275, "S355": 355}[steel_grade]
            
            # Calculate frame geometry
            geometry = self.calculate_frame_geometry()
            if geometry is None:
                self.status_label.configure(text="Geometri hesaplama hatası!")
                return
            
            # Get current section selections to determine optimization types
            current_beam_section = self.raf_section_combo.get()
            current_column_section = self.col_section_combo.get()
            
            # Determine section types for optimization - Always include both types for comprehensive optimization
            beam_types = ["IPE", "HEA"]  # Always optimize both types for beams
            column_types = ["IPE", "HEA"]  # Always optimize both types for columns
            
            # Optional: respect current selection if you want to limit search space
            # if current_beam_section.startswith("IPE") or current_beam_section == "Özel":
            #     beam_types.append("IPE")
            # if current_beam_section.startswith("HEA"):
            #     beam_types.append("HEA")
            # if not beam_types:  # Default to both types
            #     beam_types = ["IPE", "HEA"]
            
            self.status_label.configure(text="Toplam ağırlık optimizasyonu yapılıyor...")
            
            # Debug information
            print(f"Current analysis: {self.current_analysis}")
            print(f"Max moment beam: {self.current_analysis['max_moment_beam']}")
            print(f"Max moment column: {self.current_analysis['max_moment_column']}")
            print(f"Max axial column: {self.current_analysis['max_axial_column']}")
            print(f"Beam optimization types: {beam_types} (current: {current_beam_section})")
            print(f"Column optimization types: {column_types} (current: {current_column_section})")
            
            # Get haunch parameters
            haunch_enable = self.haunch_enable.get()
            haunch_length = float(self.haunch_length_entry.get()) if haunch_enable else 0.0
            haunch_height_increase = float(self.haunch_height_entry.get())/100.0 if haunch_enable else 0.0  # Convert cm to m
            
            print(f"Haunch parameters - Enable: {haunch_enable}, Length: {haunch_length}m, Height increase: {haunch_height_increase}m")
            
            # Run total weight optimization
            optimization_result = optimize_portal_frame_total_weight(
                max_N_beam=0,  # Beam has no axial load
                max_V_beam=50,  # Approximate shear force (N)
                max_M_beam=self.current_analysis['max_moment_beam'],  # kNm
                max_N_column=self.current_analysis['max_axial_column'],  # kN
                max_V_column=50,  # Approximate shear force (N)
                max_M_column=self.current_analysis['max_moment_column'],  # kNm
                steel_grade=steel_grade,
                design_code=design_code,
                beam_length=geometry['beam_length'],
                column_length=geometry['column_length'],
                num_columns=geometry['num_columns'],
                beam_section_types=beam_types,
                column_section_types=column_types,
                haunch_enable=haunch_enable,
                haunch_length=haunch_length,
                haunch_height_increase=haunch_height_increase
            )
            
            print(f"Total weight optimization result: {optimization_result}")
            
            # Show optimization results
            self.show_total_weight_optimization_results(optimization_result, design_code, geometry)
            
        except Exception as e:
            self.status_label.configure(text=f"Optimizasyon hatası: {str(e)}")
            print(f"Optimization error: {e}")
            
    def show_optimization_results(self, beam_result, column_result, design_code):
        """Show optimization results in a new window"""
        opt_window = ctk.CTkToplevel(self)
        opt_window.title(f"Optimizasyon Sonuçları - {design_code}")
        opt_window.geometry("800x700")  # Increased height for debug info
        
        # Create scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(opt_window)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Add title
        title_label = ctk.CTkLabel(scrollable_frame, text=f"Optimizasyon Sonuçları ({design_code})", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(0, 20))
        
        # Debug information
        debug_frame = ctk.CTkFrame(scrollable_frame)
        debug_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(debug_frame, text="DEBUG BİLGİSİ", 
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        
        debug_text = f"Kiriş Sonucu: {beam_result}\n\n"
        debug_text += f"Kolon Sonucu: {column_result}"
        
        debug_label = ctk.CTkLabel(debug_frame, text=debug_text, justify="left")
        debug_label.pack(padx=10, pady=5)
        
        # Beam optimization results
        beam_frame = ctk.CTkFrame(scrollable_frame)
        beam_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(beam_frame, text="KİRİŞ OPTİMİZASYONU", 
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        if beam_result.get('optimal_section'):
            result_text = f"En Hafif Kesit: {beam_result['optimal_section']}\n"
            result_text += f"Ağırlık: {beam_result['weight']:.2f} kg/m\n"
            result_text += f"Kullanım Oranı: {beam_result['utilization']:.2f}"
            ctk.CTkLabel(beam_frame, text=result_text, justify="left").pack(padx=20, pady=5)
        else:
            error_msg = beam_result.get('message', 'Bilinmeyen hata')
            ctk.CTkLabel(beam_frame, text=f"Uygun kesit bulunamadı!\nNeden: {error_msg}", 
                        text_color="red", justify="left").pack(padx=20, pady=5)
        
        # Column optimization results
        column_frame = ctk.CTkFrame(scrollable_frame)
        column_frame.pack(fill="x", pady=(10, 0))
        
        ctk.CTkLabel(column_frame, text="KOLON OPTİMİZASYONU", 
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        if column_result.get('optimal_section'):
            result_text = f"En Hafif Kesit: {column_result['optimal_section']}\n"
            result_text += f"Ağırlık: {column_result['weight']:.2f} kg/m\n"
            result_text += f"Kullanım Oranı: {column_result['utilization']:.2f}"
            ctk.CTkLabel(column_frame, text=result_text, justify="left").pack(padx=20, pady=5)
        else:
            error_msg = column_result.get('message', 'Bilinmeyen hata')
            ctk.CTkLabel(column_frame, text=f"Uygun kesit bulunamadı!\nNeden: {error_msg}", 
                        text_color="red", justify="left").pack(padx=20, pady=5)
            
        # Update sections button (only if both results are successful)
        if beam_result.get('optimal_section') and column_result.get('optimal_section'):
            update_button = ctk.CTkButton(scrollable_frame, text="Optimum Kesitleri Uygula",
                                          command=lambda: self.apply_optimal_sections(beam_result, column_result))
            update_button.pack(pady=20)

    def show_total_weight_optimization_results(self, optimization_result, design_code, geometry):
        """Show total weight optimization results in a new window"""
        opt_window = ctk.CTkToplevel(self)
        opt_window.title(f"Toplam Ağırlık Optimizasyonu - {design_code}")
        opt_window.geometry("900x800")  
        
        # Create scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(opt_window)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Add title
        title_label = ctk.CTkLabel(scrollable_frame, text=f"Portal Çerçeve Toplam Ağırlık Optimizasyonu ({design_code})", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(0, 20))
        
        # Geometry information
        geom_frame = ctk.CTkFrame(scrollable_frame)
        geom_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(geom_frame, text="ÇERÇEVE GEOMETRİSİ", 
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        
        geom_text = f"Toplam Kiriş Uzunluğu: {geometry['beam_length']:.2f} m\n"
        geom_text += f"Toplam Kolon Uzunluğu: {geometry['total_column_length']:.2f} m\n" 
        geom_text += f"Kolon Sayısı: {geometry['num_columns']}\n"
        geom_text += f"Çerçeve Açıklığı: {geometry['span']:.2f} m"
        
        ctk.CTkLabel(geom_frame, text=geom_text, justify="left").pack(padx=20, pady=5)
        
        if optimization_result['status'] == 'FAILED':
            # Show error message
            error_frame = ctk.CTkFrame(scrollable_frame)
            error_frame.pack(fill="x", pady=(10, 0))
            
            ctk.CTkLabel(error_frame, text="OPTİMİZASYON BAŞARISIZ", 
                         font=ctk.CTkFont(size=14, weight="bold"), text_color="red").pack(pady=10)
            
            error_msg = optimization_result.get('error', 'Bilinmeyen hata')
            ctk.CTkLabel(error_frame, text=f"Hata: {error_msg}", 
                        text_color="red", justify="left").pack(padx=20, pady=5)
            return
        
        # Show optimization summary
        summary_frame = ctk.CTkFrame(scrollable_frame)
        summary_frame.pack(fill="x", pady=(10, 0))
        
        ctk.CTkLabel(summary_frame, text="OPTİMİZASYON ÖZETİ", 
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        optimal = optimization_result['best_combination']
        summary_text = f"Kontrol Edilen Kombinasyon Sayısı: {optimization_result['total_combinations_checked']}\n"
        summary_text += f"Güvenli Kombinasyon Sayısı: {optimization_result['valid_combinations_count']}\n\n"
        
        # Add haunch information if enabled
        if optimization_result.get('haunch', {}).get('enabled', False):
            haunch_info = optimization_result['haunch']
            summary_text += f"HAUNCH (KONSOL) BİLGİSİ:\n"
            summary_text += f"Haunch Uzunluğu: {haunch_info['length']:.2f} m\n"
            summary_text += f"Yükseklik Artışı: {haunch_info['height_increase']*100:.0f} cm\n"
            summary_text += f"Etkin Moment (Haunch ile): {optimal.get('effective_max_M_beam', 0):.1f} kNm\n\n"
        
        summary_text += f"OPTIMAL KOMBINASYON:\n"
        summary_text += f"Kiriş: {optimal['beam_section']} ({optimal['beam_weight_per_m']:.1f} kg/m)\n"
        summary_text += f"Kolon: {optimal['column_section']} ({optimal['column_weight_per_m']:.1f} kg/m)\n\n"
        summary_text += f"AĞIRLIK DETAYI:\n"
        summary_text += f"Toplam Kiriş Ağırlığı: {optimal['total_beam_weight']:.1f} kg\n"
        summary_text += f"Toplam Kolon Ağırlığı: {optimal['total_column_weight']:.1f} kg\n"
        summary_text += f"Toplam Çerçeve Ağırlığı: {optimal['total_weight']:.1f} kg\n\n"
        summary_text += f"KULLANIM ORANLARI:\n"
        summary_text += f"Kiriş Kullanım Oranı: {optimal['beam_check'].get('utilization', 0):.2f}\n"
        summary_text += f"Kolon Kullanım Oranı: {optimal['column_check'].get('utilization', 0):.2f}\n"
        
        # Add buckling information if available
        if optimal.get('column_buckling_check'):
            buckling = optimal['column_buckling_check']
            summary_text += f"Kolon Burkulma Oranı: {buckling.get('utilization', 0):.2f}\n"
            summary_text += f"Normalleştirilmiş Narinlik: {buckling.get('normalized_slenderness', 0):.2f}\n"
            summary_text += f"Burkulma Azaltma Faktörü: {buckling.get('reduction_factor', 0):.2f}"
        
        ctk.CTkLabel(summary_frame, text=summary_text, justify="left").pack(padx=20, pady=5)
        
        # Show alternatives
        if optimization_result['all_combinations']:
            alt_frame = ctk.CTkFrame(scrollable_frame)
            alt_frame.pack(fill="x", pady=(10, 0))
            
            ctk.CTkLabel(alt_frame, text="ALTERNATİF ÇÖZÜMLER (İlk 10)", 
                         font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
            
            alt_text = ""
            for i, combo in enumerate(optimization_result['all_combinations'][:10], 1):
                beam_util = combo['beam_check'].get('utilization', 0)
                column_util = combo['column_check'].get('utilization', 0)
                buckling_util = combo.get('column_buckling_check', {}).get('utilization', 0)
                alt_text += f"{i:2d}. {combo['beam_section']} + {combo['column_section']}: "
                alt_text += f"{combo['total_weight']:.1f} kg "
                alt_text += f"(K:{beam_util:.2f}, S:{column_util:.2f}, B:{buckling_util:.2f})\n"
            
            ctk.CTkLabel(alt_frame, text=alt_text, justify="left", 
                        font=ctk.CTkFont(family="Courier", size=10)).pack(padx=20, pady=5)
        
        # Apply optimal sections button
        if optimization_result['status'] == 'SUCCESS':
            apply_button = ctk.CTkButton(scrollable_frame, text="Optimal Kombinasyonu Uygula",
                                          command=lambda: self.apply_optimal_combination(optimization_result['best_combination']))
            apply_button.pack(pady=20)

    def apply_optimal_sections(self, beam_result, column_result):
        """Apply the optimal sections to the UI dropdowns and recalculate analysis"""
        print(f"DEBUG: apply_optimal_sections başlatıldı")
        print(f"DEBUG: Beam result: {beam_result.get('optimal_section', 'None')}")
        print(f"DEBUG: Column result: {column_result.get('optimal_section', 'None')}")
        
        if beam_result.get('optimal_section'):
            old_beam = self.raf_section_combo.get()
            print(f"DEBUG: Eski beam section: {old_beam}")
            
            # Set new beam section
            self.raf_section_combo.set(beam_result['optimal_section'])
            new_beam = self.raf_section_combo.get()
            print(f"DEBUG: Yeni beam section: {new_beam}")
            print(f"DEBUG: Beam section değişikliği başarılı: {old_beam} -> {new_beam}")
            
            # Manually update beam properties
            try:
                props = get_section_properties(beam_result['optimal_section'])
                print(f"DEBUG: Beam properties alındı: {props}")
                
                old_A = self.Araf_entry.get()
                old_I = self.Iraf_entry.get()
                print(f"DEBUG: Eski beam A: {old_A}, I: {old_I}")
                
                # Update area with forced focus
                self.Araf_entry.focus()
                self.update_idletasks()
                self.Araf_entry.delete(0, "end")
                self.Araf_entry.insert(0, f"{props['A']*1e4:.1f}")
                self.update_idletasks()
                new_A = self.Araf_entry.get()
                
                # Update inertia with forced focus
                self.Iraf_entry.focus()
                self.update_idletasks()
                self.Iraf_entry.delete(0, "end")
                self.Iraf_entry.insert(0, f"{props['I']*1e8:.0f}")
                self.update_idletasks()
                new_I = self.Iraf_entry.get()
                
                print(f"DEBUG: Yeni beam A: {new_A}, I: {new_I}")
                print(f"DEBUG: Beam A değişikliği: {old_A} -> {new_A}")
                print(f"DEBUG: Beam I değişikliği: {old_I} -> {new_I}")
            except Exception as e:
                print(f"DEBUG: Beam properties update error: {e}")
            
        if column_result.get('optimal_section'):
            old_column = self.col_section_combo.get()
            print(f"DEBUG: Eski column section: {old_column}")
            
            # Set new column section
            self.col_section_combo.set(column_result['optimal_section'])
            new_column = self.col_section_combo.get()
            print(f"DEBUG: Yeni column section: {new_column}")
            print(f"DEBUG: Column section değişikliği başarılı: {old_column} -> {new_column}")
            
            # Manually update column properties
            try:
                props = get_section_properties(column_result['optimal_section'])
                print(f"DEBUG: Column properties alındı: {props}")
                
                
                old_A = self.Acol_entry.get()
                old_I = self.Icol_entry.get()
                print(f"DEBUG: Eski column A: {old_A}, I: {old_I}")
                
                # Update area with forced focus
                self.Acol_entry.focus()
                self.update_idletasks()
                self.Acol_entry.delete(0, "end")
                self.Acol_entry.insert(0, f"{props['A']*1e4:.1f}")
                self.update_idletasks()
                new_A = self.Acol_entry.get()
                
                # Update inertia with forced focus
                self.Icol_entry.focus()
                self.update_idletasks()
                self.Icol_entry.delete(0, "end")
                self.Icol_entry.insert(0, f"{props['I']*1e8:.0f}")
                self.update_idletasks()
                new_I = self.Icol_entry.get()
                
                print(f"DEBUG: Yeni column A: {new_A}, I: {new_I}")
                print(f"DEBUG: Column A değişikliği: {old_A} -> {new_A}")
                print(f"DEBUG: Column I değişikliği: {old_I} -> {new_I}")
            except Exception as e:
                print(f"DEBUG: Column properties update error: {e}")
        
        # Force UI update
        print("DEBUG: UI güncelleme yapılıyor...")
        self.update()
        
        # Recalculate analysis with new sections
        print("DEBUG: Yeni kesitlerle analiz başlatılıyor...")
        self.status_label.configure(text="Yeni kesitlerle analiz yapılıyor...")
        self.update()
        
        try:
            # Get updated parameters with new sections
            params = self.get_parameters()
            if params is None:
                print("DEBUG: Parametre hatası!")
                self.status_label.configure(text="Parametre hatası!")
                return
            
            print(f"DEBUG: Yeni parametreler:")
            print(f"  Beam section: {params['beam_section']}")
            print(f"  Column section: {params['column_section']}")
            print(f"  Beam A: {params['A_raf']*1e4:.1f} cm², I: {params['I_raf']*1e8:.0f} cm⁴")
            print(f"  Column A: {params['A_col']*1e4:.1f} cm², I: {params['I_col']*1e8:.0f} cm⁴")
            
            combinations = self.get_selected_combinations()
            if not combinations:
                print("DEBUG: Kombinasyon seçilmemiş!")
                self.status_label.configure(text="En az bir kombinasyon seçin!")
                return
            
            print(f"DEBUG: Seçilen kombinasyonlar: {combinations}")
            
            # Recalculate maximum forces with new sections
            max_moment_beam = 0
            max_moment_column = 0
            max_axial_column = 0
            
            for combo_name in combinations:
                print(f"DEBUG: {combo_name} kombinasyonu analiz ediliyor...")
                nodes, elems, samples, subtitle = build_and_run(params, combo_name)
                
                # Get maximum forces from this combination
                for sample in samples:
                    N_vals = np.atleast_1d(sample['N']) * 1e-3  # Convert to kN
                    M_vals = np.atleast_1d(sample['M']) * 1e-3  # Convert to kN⋅m
                    
                    max_moment_beam = max(max_moment_beam, np.max(np.abs(M_vals)))
                    max_moment_column = max(max_moment_column, np.max(np.abs(M_vals)))
                    max_axial_column = max(max_axial_column, np.max(np.abs(N_vals)))
            
            # Update current analysis with new data
            self.current_analysis = {
                'params': params,
                'combinations': combinations,
                'max_moment_beam': max_moment_beam,
                'max_moment_column': max_moment_column,
                'max_axial_column': max_axial_column
            }
            
            print(f"DEBUG: Analiz güncellendi:")
            print(f"  Max moment beam: {max_moment_beam:.2f} kNm")
            print(f"  Max moment column: {max_moment_column:.2f} kNm")  
            print(f"  Max axial column: {max_axial_column:.2f} kN")
            
            self.status_label.configure(text="Optimum kesitler uygulandı ve analiz güncellendi!")
            print("DEBUG: apply_optimal_sections başarıyla tamamlandı!")
            
        except Exception as e:
            print(f"DEBUG: Analysis update error: {e}")
            self.status_label.configure(text=f"Analiz güncelleme hatası: {str(e)}")
    
    def calculate_frame_geometry(self):
        """Calculate portal frame geometry for total weight optimization"""
        try:
            # Get parameters
            span = float(self.span_entry.get())
            h1 = float(self.h1_entry.get()) 
            h2 = float(self.h2_entry.get())
            ridge = float(self.ridge_entry.get())
            spacing = float(self.spacing_entry.get())
            
            # Calculate beam length (rafter length)
            # Each rafter goes from eave to ridge
            rafter_length = ((span/2)**2 + (ridge - h1)**2)**0.5
            total_beam_length = 2 * rafter_length  # Two rafters per frame
            
            # Calculate column lengths
            total_column_length = h1 + h2  # Two columns per frame
            num_columns = 2
            
            geometry = {
                'beam_length': total_beam_length,
                'column_length': h1,  # Single column length
                'total_column_length': total_column_length,
                'num_columns': num_columns,
                'rafter_length': rafter_length,
                'span': span,
                'spacing': spacing
            }
            
            print(f"DEBUG: Frame geometry calculated:")
            print(f"  Span: {span} m")
            print(f"  Column heights: {h1}, {h2} m") 
            print(f"  Ridge height: {ridge} m")
            print(f"  Rafter length: {rafter_length:.2f} m")
            print(f"  Total beam length: {total_beam_length:.2f} m")
            print(f"  Total column length: {total_column_length:.2f} m")
            
            return geometry
            
        except Exception as e:
            print(f"DEBUG: Error calculating frame geometry: {e}")
            return None

    def apply_optimal_combination(self, optimal_combo):
        """Apply optimal section combination to UI"""
        try:
            # Apply beam section
            beam_section = optimal_combo['beam_section'] 
            if beam_section != self.raf_section_combo.get():
                self.raf_section_combo.set(beam_section)
                # Update beam properties when section changes
                self.update_beam_properties(beam_section)
            
            # Apply column section
            column_section = optimal_combo['column_section']
            if column_section != self.col_section_combo.get():
                self.col_section_combo.set(column_section)
                # Update column properties when section changes
                self.update_column_properties(column_section)
                
            messagebox.showinfo("Başarılı", 
                                f"Optimal kesitler uygulandı:\nKiriş: {beam_section}\nKolon: {column_section}\nToplam Ağırlık: {optimal_combo['total_weight']:.1f} kg")
        except Exception as e:
            messagebox.showerror("Hata", f"Optimal kesitler uygulanırken hata: {str(e)}")
        
if __name__ == "__main__":
    app = ModernPortalAnalyzer()
    app.mainloop()
