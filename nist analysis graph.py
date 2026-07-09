import matplotlib.pyplot as plt

# 1. Data extracted from your finalAnalysisReport.txt
# For tests with multiple sub-results (like Cumulative Sums or Non-Overlapping Templates),
# the standard convention is to plot the minimum or average P-value.
data = {
    "Frequency": 0.122325,
    "Block Frequency": 0.122325,
    "Cumulative Sums": 0.213309,
    "Runs": 0.534146,
    "Longest Run": 0.739918,
    "Rank": 0.911413,
    "FFT": 0.350485,
    "Non-Overlapping Template": 0.534146, # Representative value
    "Overlapping Template": 0.350485,
    "Universal": 0.350485,
    "Approximate Entropy": 0.534146,
    "Serial": 0.350485,
    "Linear Complexity": 0.213309
}

# Sort alphabetically to match the style of your example plot
sorted_data = dict(sorted(data.items()))

tests = list(sorted_data.keys())
p_values = list(sorted_data.values())

# 2. Initialize the plot
plt.figure(figsize=(8, 7))

# 3. Create the bars (using a deep blue color matching your image)
plt.bar(tests, p_values, color='#001f9c', edgecolor='none', width=0.8)

# 4. Set Y-axis to Logarithmic Scale to match the exponents (10^-3 to 10^0)
plt.yscale('log')
plt.ylim(1e-3, 1.1)

# 5. Add the red dashed significance threshold line at 0.01 (10^-2)
plt.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5)

# 6. Format axes and text styling
plt.ylabel('P-value', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(False) # Clean background matching your reference image

# Adjust layout so labels don't get cut off at the bottom
plt.tight_layout()

# 7. Save and show the plot
plt.savefig('nist_results_histogram.png', dpi=300)
plt.show()
