import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV (from repo data/)
data = pd.read_csv("data/field_validation_soc.csv")

# Scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(data['Trend.Earth SOC (kg C m⁻²)'], data['Measured SOC (kg C m⁻²)'], 
            s=60, edgecolors='k', alpha=0.8)

# 1:1 line
x = np.linspace(0, 2, 100)
plt.plot(x, x, '--', color='gray', label='1:1 line')

# Regression
z = np.polyfit(data['Trend.Earth SOC (kg C m⁻²)'], data['Measured SOC (kg C m⁻²)'], 1)
p = np.poly1d(z)
plt.plot(x, p(x), '-', color='black', label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

plt.xlabel('Trend.Earth SOC (kg C m⁻²)')
plt.ylabel('Measured SOC (Walkley-Black, kg C m⁻²)')
plt.title('Validation of Trend.Earth SOC Estimates (n=30)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(0.1, 1.7, 'R² = 0.78\nRMSE = 0.32', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('output/soc_validation_plot.png', dpi=300)
plt.show()

print("SOC validation plot saved in output/")
