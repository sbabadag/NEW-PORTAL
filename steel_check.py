# Steel section database and verification functions for Eurocode 3
import numpy as np

# Steel section database (typical values)
STEEL_SECTIONS = {
    # HEA sections: [A(cm²), I_y(cm⁴), W_y(cm³), h(mm), b(mm), t_f(mm), t_w(mm)]
    "HEA 100": [21.2, 349, 69.8, 96, 100, 8.0, 5.0],
    "HEA 120": [25.3, 606, 101, 114, 120, 8.0, 5.0],
    "HEA 140": [31.4, 1033, 148, 133, 140, 8.5, 5.5],
    "HEA 160": [38.8, 1673, 220, 152, 160, 9.0, 6.0],
    "HEA 180": [45.3, 2510, 294, 171, 180, 9.5, 6.0],
    "HEA 200": [53.8, 3692, 369, 190, 200, 10.0, 6.5],
    "HEA 220": [64.3, 5410, 491, 210, 220, 11.0, 7.0],
    "HEA 240": [76.8, 7763, 647, 230, 240, 12.0, 7.5],
    "HEA 260": [86.8, 10450, 804, 250, 260, 12.5, 7.5],
    "HEA 280": [97.3, 13670, 977, 270, 280, 13.0, 8.0],
    "HEA 300": [112.5, 18263, 1260, 290, 300, 14.0, 8.5],
    "HEA 320": [124.4, 22930, 1479, 310, 300, 15.5, 9.0],
    "HEA 340": [133.5, 27690, 1678, 330, 350, 16.5, 9.5],
    "HEA 360": [142.8, 33090, 1891, 350, 350, 17.5, 10.0],
    "HEA 400": [159.0, 45070, 2307, 390, 300, 19.0, 11.0],
    "HEA 450": [178.0, 63720, 2896, 440, 300, 21.0, 11.5],
    "HEA 500": [197.8, 86970, 3550, 490, 300, 23.0, 12.0],
    "HEA 550": [212.9, 111900, 4309, 540, 300, 24.0, 12.5],
    "HEA 600": [226.0, 141200, 4747, 590, 300, 25.0, 13.0],
    "HEA 650": [242.8, 175000, 5387, 640, 300, 26.0, 13.5],
    "HEA 700": [260.4, 214800, 6143, 690, 300, 27.0, 14.5],
    "HEA 800": [286.4, 303400, 7586, 790, 300, 28.0, 15.0],
    "HEA 900": [320.5, 422100, 9384, 890, 300, 30.0, 16.0],
    "HEA 1000": [347.0, 553000, 11070, 990, 300, 31.0, 16.5],
    
    # IPE sections: [A(cm²), I_y(cm⁴), W_y(cm³), h(mm), b(mm), t_f(mm), t_w(mm)]
    "IPE 80": [7.64, 80.1, 20.0, 80, 46, 5.2, 3.8],
    "IPE 100": [10.32, 171, 34.2, 100, 55, 5.7, 4.1],
    "IPE 120": [13.21, 318, 53.0, 120, 64, 6.3, 4.4],
    "IPE 140": [16.43, 541, 77.3, 140, 73, 6.9, 4.7],
    "IPE 160": [20.09, 869, 108.7, 160, 82, 7.4, 5.0],
    "IPE 180": [23.95, 1317, 146.3, 180, 91, 8.0, 5.3],
    "IPE 200": [28.48, 1943, 194.3, 200, 100, 8.5, 5.6],
    "IPE 220": [33.37, 2772, 252.0, 220, 110, 9.2, 5.9],
    "IPE 240": [39.12, 3892, 324.3, 240, 120, 9.8, 6.2],
    "IPE 270": [45.95, 5790, 428.9, 270, 135, 10.2, 6.6],
    "IPE 300": [53.81, 8356, 557.0, 300, 150, 10.7, 7.1],
    "IPE 330": [62.61, 11770, 713.0, 330, 160, 11.5, 7.5],
    "IPE 360": [72.73, 16270, 903.6, 360, 170, 12.7, 8.0],
    "IPE 400": [84.46, 23130, 1156.0, 400, 180, 13.5, 8.6],
    "IPE 450": [98.82, 33740, 1500.0, 450, 190, 14.6, 9.4],
    "IPE 500": [116.0, 48200, 1928.0, 500, 200, 16.0, 10.2],
    "IPE 550": [134.4, 67120, 2441.0, 550, 210, 17.2, 11.1],
    "IPE 600": [156.0, 92080, 3069.0, 600, 220, 19.0, 12.0],
    "IPE 750": [196.8, 173000, 4610.0, 750, 265, 23.0, 14.0],
}

# Material properties
STEEL_GRADE = {
    "S235": {"f_y": 235, "f_u": 360},  # N/mm²
    "S275": {"f_y": 275, "f_u": 430},
    "S355": {"f_y": 355, "f_u": 510},
}

def get_section_properties(section_name):
    """Get section properties from database"""
    if section_name in STEEL_SECTIONS:
        props = STEEL_SECTIONS[section_name]
        return {
            "A": props[0] * 1e-4,      # cm² to m²
            "I": props[1] * 1e-8,      # cm⁴ to m⁴
            "W": props[2] * 1e-6,      # cm³ to m³
            "h": props[3] * 1e-3,      # mm to m
            "b": props[4] * 1e-3,      # mm to m
            "t_f": props[5] * 1e-3,    # mm to m
            "t_w": props[6] * 1e-3,    # mm to m
        }
    return None

def check_section_resistance(section_name, fy, M_Ed, N_Ed=0, design_code="Eurocode 3 (EN)", V_Ed=0):
    """
    Check section resistance according to specified design code
    
    Parameters:
    - section_name: Steel section designation  
    - fy: Yield strength (MPa)
    - M_Ed: Design moment (kN⋅m)
    - N_Ed: Design axial force (kN)
    - design_code: Design code ("Eurocode 3 (EN)", "TS 648 (Türk)", etc.)
    - V_Ed: Design shear force (kN) 
    
    Returns:
    - dict with check results
    """
    # Convert to SI units
    M_Ed = M_Ed * 1000  # kN⋅m to N⋅m
    N_Ed = N_Ed * 1000  # kN to N  
    V_Ed = V_Ed * 1000  # kN to N
    fy = fy * 1e6  # MPa to Pa
    # Get section properties
    props = get_section_properties(section_name)
    if props is None:
        return {"error": f"Section {section_name} not found in database"}
    
    # Material properties and safety factors based on design code
    f_y = fy  # Already in Pa from parameter conversion above
    
    # Set safety factors based on design code
    if "Eurocode" in design_code:
        gamma_M0 = 1.0   # Eurocode 3
        gamma_M1 = 1.0
        code_label = "EC3"
    elif "TS 648" in design_code:
        gamma_M0 = 1.15  # Turkish Standard TS 648
        gamma_M1 = 1.15
        code_label = "TS648"
    elif "ÇYTHYE" in design_code:
        gamma_M0 = 1.1   # Turkish Steel Design Code
        gamma_M1 = 1.1
        code_label = "ÇYTHYE"
    elif "TBDY" in design_code:
        gamma_M0 = 1.0   # Turkish Seismic Design Code (similar to Eurocode)
        gamma_M1 = 1.0
        code_label = "TBDY"
    else:
        gamma_M0 = 1.0   # Default to Eurocode
        gamma_M1 = 1.0
        code_label = "EC3"
    
    A = props["A"]
    I = props["I"]
    W = props["W"]
    h = props["h"]
    t_w = props["t_w"]
    
    results = {
        "section": section_name,
        "steel_fy": fy/1e6,  # Pa to MPa
        "design_code": design_code,
        "code_label": code_label,
        "safety_factors": {"gamma_M0": gamma_M0, "gamma_M1": gamma_M1},
        "forces": {"N_Ed": N_Ed/1000, "V_Ed": V_Ed/1000, "M_Ed": M_Ed/1000},  # kN, kN⋅m
        "checks": {},
        "overall_status": "SAFE"
    }
    
    # 1. Tension resistance (if N_Ed > 0)
    if N_Ed > 0:
        N_t_Rd = A * f_y / gamma_M0
        ratio_tension = N_Ed / N_t_Rd
        results["checks"]["tension"] = {
            "N_Ed": N_Ed/1000,
            "N_t_Rd": N_t_Rd/1000,
            "ratio": ratio_tension,
            "status": "SAFE" if ratio_tension <= 1.0 else "UNSAFE"
        }
        if ratio_tension > 1.0:
            results["overall_status"] = "UNSAFE"
    
    # 2. Compression resistance (if N_Ed < 0)
    if N_Ed < 0:
        # Simplified: assuming no buckling (short member)
        N_c_Rd = A * f_y / gamma_M0
        ratio_compression = abs(N_Ed) / N_c_Rd
        results["checks"]["compression"] = {
            "N_Ed": N_Ed/1000,
            "N_c_Rd": N_c_Rd/1000,
            "ratio": ratio_compression,
            "status": "SAFE" if ratio_compression <= 1.0 else "UNSAFE"
        }
        if ratio_compression > 1.0:
            results["overall_status"] = "UNSAFE"
    
    # 3. Shear resistance
    A_v = A  # Simplified: total area (conservative)
    V_pl_Rd = A_v * (f_y / np.sqrt(3)) / gamma_M0
    ratio_shear = abs(V_Ed) / V_pl_Rd
    results["checks"]["shear"] = {
        "V_Ed": V_Ed/1000,
        "V_pl_Rd": V_pl_Rd/1000,
        "ratio": ratio_shear,
        "status": "SAFE" if ratio_shear <= 1.0 else "UNSAFE"
    }
    if ratio_shear > 1.0:
        results["overall_status"] = "UNSAFE"
    
    # 4. Bending resistance
    M_c_Rd = W * f_y / gamma_M0
    ratio_bending = abs(M_Ed) / M_c_Rd
    results["checks"]["bending"] = {
        "M_Ed": M_Ed/1000,
        "M_c_Rd": M_c_Rd/1000,
        "ratio": ratio_bending,
        "status": "SAFE" if ratio_bending <= 1.0 else "UNSAFE"
    }
    if ratio_bending > 1.0:
        results["overall_status"] = "UNSAFE"
    
    # 5. Combined stresses (simplified interaction)
    max_utilization = 0
    if N_Ed != 0 and M_Ed != 0:
        # Simplified interaction formula
        if N_Ed > 0:  # Tension + Bending
            ratio_combined = ratio_tension + ratio_bending
        else:  # Compression + Bending
            ratio_combined = ratio_compression + ratio_bending
        
        results["checks"]["combined"] = {
            "ratio": ratio_combined,
            "status": "SAFE" if ratio_combined <= 1.0 else "UNSAFE"
        }
        max_utilization = ratio_combined
        if ratio_combined > 1.0:
            results["overall_status"] = "UNSAFE"
    else:
        # Find maximum individual ratio
        all_ratios = []
        if "tension" in results["checks"]:
            all_ratios.append(results["checks"]["tension"]["ratio"])
        if "compression" in results["checks"]:
            all_ratios.append(results["checks"]["compression"]["ratio"])
        if "bending" in results["checks"]:
            all_ratios.append(results["checks"]["bending"]["ratio"])
        if "shear" in results["checks"]:
            all_ratios.append(results["checks"]["shear"]["ratio"])
        max_utilization = max(all_ratios) if all_ratios else 0
    
    # Add convenience fields for UI compatibility
    results["safety"] = results["overall_status"] == "SAFE"
    results["utilization"] = max_utilization
    
    return results

def format_check_results(check_results):
    """Format check results for display"""
    if "error" in check_results:
        return f"❌ HATA: {check_results['error']}"
    
    section = check_results["section"]
    fy = check_results.get("steel_fy", 275)  # Default to S275
    design_code = check_results.get("design_code", "Eurocode 3 (EN)")
    code_label = check_results.get("code_label", "EC3")
    status = check_results["overall_status"]
    safety_factors = check_results.get("safety_factors", {})
    
    # Status icon
    status_icon = "✅ GÜVENLİ" if status == "SAFE" else "❌ GÜVENSİZ"
    
    result_text = f"\n{'='*60}\n"
    result_text += f"KESİT TAHKİKİ: {section} (fy={fy} MPa)\n"
    result_text += f"Hesaplama Yöntemi: {design_code}\n"
    if safety_factors:
        result_text += f"Güvenlik Faktörü: γM0={safety_factors.get('gamma_M0', 1.0)}\n"
    result_text += f"{'='*60}\n"
    result_text += f"GENEL DURUM: {status_icon}\n\n"
    
    # Individual checks
    checks = check_results["checks"]
    
    if "tension" in checks:
        c = checks["tension"]
        icon = "✅" if c["status"] == "SAFE" else "❌"
        result_text += f"{icon} Çekme: N_Ed={c['N_Ed']:.1f} kN ≤ N_t_Rd={c['N_t_Rd']:.1f} kN (Oran: {c['ratio']:.2f})\n"
    
    if "compression" in checks:
        c = checks["compression"]
        icon = "✅" if c["status"] == "SAFE" else "❌"
        result_text += f"{icon} Basınç: |N_Ed|={abs(c['N_Ed']):.1f} kN ≤ N_c_Rd={c['N_c_Rd']:.1f} kN (Oran: {c['ratio']:.2f})\n"
    
    if "shear" in checks:
        c = checks["shear"]
        icon = "✅" if c["status"] == "SAFE" else "❌"
        result_text += f"{icon} Kesme: |V_Ed|={abs(c['V_Ed']):.1f} kN ≤ V_pl_Rd={c['V_pl_Rd']:.1f} kN (Oran: {c['ratio']:.2f})\n"
    
    if "bending" in checks:
        c = checks["bending"]
        icon = "✅" if c["status"] == "SAFE" else "❌"
        result_text += f"{icon} Eğilme: |M_Ed|={abs(c['M_Ed']):.1f} kN⋅m ≤ M_c_Rd={c['M_c_Rd']:.1f} kN⋅m (Oran: {c['ratio']:.2f})\n"
    
    if "combined" in checks:
        c = checks["combined"]
        icon = "✅" if c["status"] == "SAFE" else "❌"
        result_text += f"{icon} Kombine: Toplam Oran = {c['ratio']:.2f} ≤ 1.0\n"
    
    result_text += f"\n{'='*50}\n"
    
    return result_text

def optimize_section_selection(max_N, max_V, max_M, steel_grade="S275", section_type="all", design_code="Eurocode 3 (EN)"):
    """
    Find the most optimal (lightest) section that satisfies all requirements
    
    Parameters:
    - max_N: Maximum axial force (kN)
    - max_V: Maximum shear force (N) 
    - max_M: Maximum moment (kN⋅m)
    - steel_grade: Steel grade (S235, S275, S355)
    - section_type: "IPE", "HEA", or "all"
    - design_code: Design code for verification
    
    Returns:
    - dict with optimization results
    """
    
    # Filter sections based on type
    if section_type == "IPE":
        sections_to_check = {k: v for k, v in STEEL_SECTIONS.items() if k.startswith("IPE")}
    elif section_type == "HEA":
        sections_to_check = {k: v for k, v in STEEL_SECTIONS.items() if k.startswith("HEA")}
    else:
        sections_to_check = STEEL_SECTIONS
    
    safe_sections = []
    checked_sections = []
    
    # Get yield strength
    fy = {"S235": 235, "S275": 275, "S355": 355}[steel_grade]
    
    print(f"DEBUG: Optimizasyon başlatılıyor")
    print(f"  Steel grade: {steel_grade} (fy = {fy} MPa)")
    print(f"  Section type: {section_type}")
    print(f"  Design code: {design_code}")
    print(f"  Forces: N = {max_N:.1f} kN, V = {max_V:.1f} N, M = {max_M:.1f} kN⋅m")
    print(f"  Checking {len(sections_to_check)} sections...")
    
    # Check each section
    for section_name in sections_to_check:
        check_result = check_section_resistance(section_name, fy, max_M, max_N, design_code, max_V/1000)  # Convert V to kN
        
        checked_sections.append({
            "section": section_name,
            "safety": check_result.get("safety", False),
            "utilization": check_result.get("utilization", 999),
            "overall_status": check_result.get("overall_status", "UNKNOWN")
        })
        
        if check_result.get("safety"):
            props = get_section_properties(section_name)
            weight_per_meter = props["A"] * 7850  # kg/m (steel density ≈ 7850 kg/m³)
            
            safe_sections.append({
                "section": section_name,
                "weight": weight_per_meter,
                "area": props["A"] * 1e4,  # m² to cm²
                "utilization": check_result.get("utilization", 0),
                "check_result": check_result
            })
    
    print(f"DEBUG: {len(safe_sections)} güvenli kesit bulundu")
    if len(safe_sections) == 0:
        print("DEBUG: İlk 5 kontrol edilen kesit:")
        for i, chk in enumerate(checked_sections[:5]):
            print(f"  {chk['section']}: safety={chk['safety']}, util={chk['utilization']:.3f}, status={chk['overall_status']}")
    
    if not safe_sections:
        return {
            "optimal_section": None,
            "weight": None,
            "utilization": None,
            "status": "NO_SAFE_SECTION",
            "message": "Hiçbir kesit güvenli değil! Daha büyük kesitler deneyin veya çelik sınıfını yükseltin."
        }
    
    # Sort by weight (lightest first)
    safe_sections.sort(key=lambda x: x["weight"])
    
    # Get optimization results
    optimal = safe_sections[0]
    
    return {
        "optimal_section": optimal["section"],
        "weight": optimal["weight"],
        "utilization": optimal["utilization"],
        "status": "SUCCESS",
        "alternatives": safe_sections[1:6],  # Top 5 alternatives
        "total_safe_sections": len(safe_sections),
        "steel_grade": steel_grade,
        "forces": {"N": max_N/1000, "V": max_V/1000, "M": max_M/1000}
    }

def format_optimization_results(opt_results):
    """Format optimization results for display"""
    
    if opt_results["status"] == "NO_SAFE_SECTION":
        return f"❌ {opt_results['message']}"
    
    result_text = f"\n{'='*60}\n"
    result_text += f"OPTİMİZASYON SONUÇLARI\n"
    result_text += f"{'='*60}\n"
    result_text += f"Çelik Sınıfı: {opt_results['steel_grade']}\n"
    result_text += f"İç Kuvvetler: N={opt_results['forces']['N']:.1f} kN, "
    result_text += f"V={opt_results['forces']['V']:.1f} kN, "
    result_text += f"M={opt_results['forces']['M']:.1f} kN⋅m\n"
    result_text += f"Güvenli Kesit Sayısı: {opt_results['total_safe_sections']}\n\n"
    
    # Optimal section
    optimal = opt_results["optimal_section"]
    result_text += f"🏆 OPTİMAL KESİT (En Hafif):\n"
    result_text += f"{'─'*40}\n"
    result_text += f"Kesit: {optimal['section']}\n"
    result_text += f"Ağırlık: {optimal['weight']:.2f} kg/m\n"
    result_text += f"Alan: {optimal['area']:.1f} cm²\n\n"
    
    # Show detailed check for optimal section
    check = optimal["check_result"]["checks"]
    result_text += f"Güvenlik Kontrolleri:\n"
    
    for check_type, data in check.items():
        if check_type == "combined":
            icon = "✅" if data["status"] == "SAFE" else "❌"
            result_text += f"  {icon} Kombine: Oran = {data['ratio']:.3f}\n"
        elif check_type == "tension":
            icon = "✅" if data["status"] == "SAFE" else "❌"
            result_text += f"  {icon} Çekme: Oran = {data['ratio']:.3f}\n"
        elif check_type == "compression":
            icon = "✅" if data["status"] == "SAFE" else "❌"
            result_text += f"  {icon} Basınç: Oran = {data['ratio']:.3f}\n"
        elif check_type == "shear":
            icon = "✅" if data["status"] == "SAFE" else "❌"
            result_text += f"  {icon} Kesme: Oran = {data['ratio']:.3f}\n"
        elif check_type == "bending":
            icon = "✅" if data["status"] == "SAFE" else "❌"
            result_text += f"  {icon} Eğilme: Oran = {data['ratio']:.3f}\n"
    
    # Alternative sections
    if opt_results["alternatives"]:
        result_text += f"\n📋 ALTERNATİF KESİTLER:\n"
        result_text += f"{'─'*40}\n"
        for i, alt in enumerate(opt_results["alternatives"], 1):
            weight_increase = ((alt["weight"] - optimal["weight"]) / optimal["weight"]) * 100
            result_text += f"{i}. {alt['section']}: {alt['weight']:.2f} kg/m (+{weight_increase:.1f}%)\n"
    
    result_text += f"\n{'='*60}\n"
    
    return result_text
