import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from portal_modern_ui import *

# Test the build_and_run function
test_params = {
    "E": 210e9,
    "span": 20.0, "h1": 7.0, "h2": 7.0, "ridge": 8.5,
    "spacing": 6.0,
    "A_col": 0.020, "I_col": 8e-6,
    "A_raf": 0.020, "I_raf": 8e-6,
    "label_col": "HEA 240", "label_raf": "IPE 300",
    "beam_section": "IPE 300", "column_section": "HEA 240",
    "G_kNm2": 0.5,
    "include_selfweight": True,
    "s_k": 0.8, "Ce": 1.0, "Ct": 1.0,
    "pn_kNm2": 0.3, "wind_upward": True,
}

print("Testing build_and_run function...")
try:
    nodes, elems, samples, subtitle = build_and_run(test_params, "ULS (G+S)")
    print("✓ build_and_run works")
    print(f"  Number of nodes: {len(nodes)}")
    print(f"  Number of elements: {len(elems)}")
    print(f"  Number of samples: {len(samples)}")
    
    # Test get_max_forces
    max_N, max_V, max_M = get_max_forces(elems, samples)
    print(f"✓ get_max_forces works")
    print(f"  Max N: {max_N:.2f} kN")
    print(f"  Max V: {max_V:.2f} kN") 
    print(f"  Max M: {max_M:.2f} kNm")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting steel checking...")
try:
    from steel_check import check_section_resistance, format_check_results
    
    # Test steel check
    check_result = check_section_resistance("IPE 300", 275, 50, 25, 150)
    print("✓ Steel checking works")
    print(f"  Section safe: {check_result['safe']}")
    
except Exception as e:
    print(f"✗ Steel check error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test completed ===")
