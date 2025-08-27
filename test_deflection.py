#!/usr/bin/env python3
"""
Portal çerçeve sehim testi - Frame2D ile basit hesaplama
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portal_modern_ui import Frame2D, Node, Element
import numpy as np

def test_simple_beam():
    """Basit kiriş sehimi testi"""
    print("=== BASIT KİRİŞ SEHİM TESTİ ===")
    
    # Basit kiriş: L=10m, q=10 kN/m
    L = 10.0  # m
    q = 10000  # N/m (10 kN/m)
    E = 2.1e11  # Pa
    I = 8.36e-5  # m⁴ (IPE 300)
    A = 0.006855  # m²
    
    # Teorik sehim: δ = 5qL⁴/(384EI)
    delta_theory = (5 * q * L**4) / (384 * E * I)
    print(f"Teorik sehim: {delta_theory*1000:.2f} mm")
    
    # Frame2D ile hesaplama
    nodes = [
        Node(0, 0, fix=(True, True, False)),  # Sol basit mesnet
        Node(L, 0, fix=(False, True, False))  # Sağ basit mesnet
    ]
    
    elems = [
        Element(0, 1, E, A, I, w=q)
    ]
    
    model = Frame2D(nodes, elems)
    model.assemble()
    model.solve()
    
    # Node 1 (orta nokta) dikey sehimi 
    # Nodes: 0=sol, 1=sağ, orta nokta ayrı hesaplanacak
    # Basit kiriş için orta nokta interpolasyonu gerekli
    # Bu test için sadece Frame2D'nin çalıştığını kontrol edelim
    
    print(f"Frame2D çözüm tamamlandı")
    print(f"Node 0 sehimi: x={model.D[0]:.3f}, y={model.D[1]:.3f}")
    print(f"Node 1 sehimi: x={model.D[3]:.3f}, y={model.D[4]:.3f}")
    
    # Samples ile element üzerindeki değerleri alalım
    samples = model.sample_internal(npts=21)
    if samples:
        print(f"Element sample sayısı: {len(samples)}")
        print(f"Mevcut anahtarlar: {list(samples[0].keys())}")
    print()

def test_portal_frame():
    """Portal çerçeve sehimi testi"""
    print("=== PORTAL ÇERÇEVE SEHİM TESTİ ===")
    
    # Portal çerçeve parametreleri
    span = 20.0  # m
    h = 7.0     # m
    ridge = 8.5 # m
    q = 9710    # N/m (yaklaşık ULS yükü)
    E = 2.1e11  # Pa
    I_beam = 8.36e-5  # m⁴ (IPE 300)
    I_col = 7.76e-5   # m⁴ (HEA 240)
    A_beam = 0.006855 # m²
    A_col = 0.009783  # m²
    
    print(f"Açıklık: {span} m, Yükseklik: {h} m, Mahya: {ridge} m")
    print(f"Yük: {q/1000:.1f} kN/m")
    print(f"Kiriş I: {I_beam:.2e} m⁴, Kolon I: {I_col:.2e} m⁴")
    
    # Nodes
    nodes = [
        Node(0.0, 0.0, fix=(True, True, True)),      # Sol alt (sabit)
        Node(0.0, h),                                # Sol üst
        Node(span/2.0, ridge),                       # Mahya
        Node(span, h),                               # Sağ üst
        Node(span, 0.0, fix=(True, True, True)),     # Sağ alt (sabit)
    ]
    
    # Elements
    elems = [
        Element(0, 1, E, A_col, I_col, w=0.0),      # Sol kolon
        Element(1, 2, E, A_beam, I_beam, w=q),      # Sol kiriş
        Element(2, 3, E, A_beam, I_beam, w=q),      # Sağ kiriş
        Element(3, 4, E, A_col, I_col, w=0.0),      # Sağ kolon
    ]
    
    model = Frame2D(nodes, elems)
    model.assemble()
    model.solve()
    
    # Sehim değerleri
    print("\nNode sehimleri:")
    for i, node in enumerate(nodes):
        D = model.D[i*3:(i+1)*3]
        print(f"Node {i}: x={D[0]*1000:.2f} mm, y={D[1]*1000:.2f} mm, θ={D[2]*1000:.2f} mrad")
    
    # Kritik sehimler
    ridge_vertical = abs(model.D[2*3 + 1]) * 1000  # Node 2, y yönü
    col_top_horizontal = abs(model.D[1*3 + 0]) * 1000  # Node 1, x yönü
    
    print(f"\nKritik sehimler:")
    print(f"Mahya dikey sehim: {ridge_vertical:.2f} mm")
    print(f"Kolon başı yatay sehim: {col_top_horizontal:.2f} mm")
    
    # Limitler
    vert_limit = span/2 / 250 * 1000  # L/250
    hor_limit = h / 300 * 1000        # H/300
    
    print(f"\nLimitler:")
    print(f"Dikey limit (L/250): {vert_limit:.2f} mm")
    print(f"Yatay limit (H/300): {hor_limit:.2f} mm")
    
    print(f"\nKontrol:")
    print(f"Dikey: {ridge_vertical:.2f} / {vert_limit:.2f} = {ridge_vertical/vert_limit:.2f} {'✓' if ridge_vertical <= vert_limit else '✗'}")
    print(f"Yatay: {col_top_horizontal:.2f} / {hor_limit:.2f} = {col_top_horizontal/hor_limit:.2f} {'✓' if col_top_horizontal <= hor_limit else '✗'}")

if __name__ == "__main__":
    test_simple_beam()
    test_portal_frame()
