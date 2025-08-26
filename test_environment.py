#!/usr/bin/env python3
"""
Test script for portal frame analysis environment
"""

print("Testing portal frame analysis environment...")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
    
    try:
        import ipywidgets as widgets
        from IPython.display import display
        print("✓ IPywidgets and IPython imported successfully")
        jupyter_available = True
    except ImportError:
        print("⚠ IPywidgets/IPython available but may need Jupyter environment for full functionality")
        jupyter_available = False
    
    # Test basic functionality
    print("\nTesting basic calculations...")
    x = np.array([1, 2, 3, 4, 5])
    y = x ** 2
    print(f"✓ NumPy calculation works: {x} -> {y}")
    
    # Test matplotlib
    print("\nTesting matplotlib...")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, 'b-', label='y = x²')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    plt.title('Environment Test Plot')
    plt.savefig('test_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Matplotlib plot saved as 'test_plot.png'")
    
    print("\n" + "="*50)
    print("🎉 Environment setup complete!")
    print("Your portal frame analysis tool is ready to run.")
    if jupyter_available:
        print("💡 For best experience, run 'portal.py' in a Jupyter notebook environment.")
    else:
        print("💡 The tool will run in basic mode without interactive widgets.")
    print("="*50)
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("Please check your environment setup.")
