import pandas as pd
import numpy as np

def calculate_entropy_per_bit(file_path):
    print(f"Loading CH3 from {file_path}...")
    
    # Locate header dynamically
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if "TIME" in line:
                header_row = idx
                break
                
    df = pd.read_csv(file_path, skiprows=header_row)
    ch3_raw = df['CH3'].values
    
    # 1. 12-bit Quantization (mapping analog voltages to whole integers 0-4095)
    v_min, v_max = np.min(ch3_raw), np.max(ch3_raw)
    quantized = ((ch3_raw - v_min) / (v_max - v_min) * 4095).astype(np.uint16)
    
    # 2. Extract the Least Significant Bit (LSB)
    bits = quantized & 1
    total_bits = len(bits)
    
    # 3. Count 0s and 1s to find probabilities
    count_ones = np.sum(bits)
    count_zeros = total_bits - count_ones
    
    p0 = count_zeros / total_bits
    p1 = count_ones / total_bits
    
    # 4. Calculate Shannon Entropy
    # We add a tiny epsilon (1e-12) to prevent log2(0) crashes if a bit doesn't exist
    entropy = - (p0 * np.log2(p0 + 1e-12) + p1 * np.log2(p1 + 1e-12))
    
    print(f"--- Entropy Report for {file_path} ---")
    print(f"  Total bits analyzed: {total_bits:,}")
    print(f"  Zeros '0': {count_zeros:,} ({p0*100:.2f}%)")
    print(f"  Ones '1':  {count_ones:,} ({p1*100:.2f}%)")
    print(f"  **Entropy per bit**: {entropy:.6f} bits")
    print("-" * 50)
    return entropy

# Run the calculation for both experimental runs
entropy_500 = calculate_entropy_per_bit("Tek002-500mv_ALL.csv")
entropy_7dbm = calculate_entropy_per_bit("Tek002-max7dbm_ALL.csv")
