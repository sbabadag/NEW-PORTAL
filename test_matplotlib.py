import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

print("Matplotlib backend:", matplotlib.get_backend())
print("Available backends:", matplotlib.backend_bases.Backend.__subclasses__())

# Test plot
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title("Test Plot")
ax.grid(True)

# Save to file to test
plt.savefig("matplotlib_test.png", dpi=100, bbox_inches='tight')
print("✓ Test plot saved as 'matplotlib_test.png'")
plt.close()

print("✓ Matplotlib test completed successfully")
