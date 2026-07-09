import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Load the data, skipping the 16 metadata lines
file_name = "D:\work_pc_IITH\JJ\my work\qrn extraction\Tek001_ALL_elec.csv"
# header=18 points to the row: TIME, CH1, CH2, CH3
df = pd.read_csv(file_name, skiprows=17, header=0)

# Extract time and channel data
time = df['TIME'].values

ch3 = df['CH3'].values

print(f"Loaded {len(ch3)} points from {file_name}")

# 2. Plot Distribution and check for Gaussian fit
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
count, bins, ignored = plt.hist(ch3, bins=100, density=True, alpha=0.6, color='g')
mu, sigma = norm.fit(ch3)
plt.plot(bins, norm.pdf(bins, mu, sigma), 'r--', linewidth=2)
plt.title(f'CH3 Distribution{mu:.4f}, {sigma:.4f}')
plt.xlabel('Voltage (V)')

# 3. Check Autocorrelation to identify oversampling
plt.subplot(1, 2, 2)
# Using a pandas wrapper or numpy to check the first 100 lags
lags = 100
acf = [pd.Series(ch3).autocorr(lag=i) for i in range(lags)]
plt.stem(range(lags), acf)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lags')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()
plt.show()
