import pandas as pd
import numpy as np

def calculate_true_bit_entropy(file_path):
    print(f"Analyzing hidden memory in {file_path}...")
    
    # Locate header dynamically
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if "TIME" in line:
                header_row = idx
                break
                
    df = pd.read_csv(file_path, skiprows=header_row)
    ch3_raw = df['CH3'].values
    
    # 1. 12-bit Quantization and LSB extraction
    v_min, v_max = np.min(ch3_raw), np.max(ch3_raw)
    quantized = ((ch3_raw - v_min) / (v_max - v_min) * 4095).astype(np.uint16)
    bits = (quantized & 1).astype(np.uint8)
    
    # 2. Check Bit Autocorrelation (Lag-1)
    # Checks how correlated a bit is with the very next bit
    bit_series = pd.Series(bits)
    bit_corr = bit_series.autocorr(lag=1)
    
    # 3. Create 2-bit blocks to calculate Block Entropy
    # Pair up bit[t] and bit[t+1]
    bits_t = bits[:-1]
    bits_next = bits[1:]
    
    # Combine pairs into a single integer representation (0, 1, 2, or 3)
    pairs = (bits_t << 1) + bits_next
    total_pairs = len(pairs)
    
    # Count occurrences of 00, 01, 10, 11
    counts = np.bincount(pairs, minlength=4)
    probs = counts / total_pairs
    
    # Calculate 2-Bit Shannon Entropy
    # Perfect independence should yield exactly 2.0 bits
    block_entropy_2bit = -np.sum([p * np.log2(p + 1e-12) for p in probs])
    
    print(f"\n================ HIDDEN MEMORY REPORT: {file_path} ================")
    print(f"  Consecutive Bit Autocorrelation (Lag 1): {bit_corr:.4f}")
    print(f"  (Target for true QRNG should be near 0.0000)")
    print(f"\n  Pair Distribution Frequencies:")
    print(f"    Pattern '00': {counts[0]:,} ({probs[0]*100:.2f}%) [Target: 25%]")
    print(f"    Pattern '01': {counts[1]:,} ({probs[1]*100:.2f}%) [Target: 25%]")
    print(f"    Pattern '10': {counts[2]:,} ({probs[2]*100:.2f}%) [Target: 25%]")
    print(f"    Pattern '11': {counts[3]:,} ({probs[3]*100:.2f}%) [Target: 25%]")
    print(f"\n  **True 2-Bit Block Entropy**: {block_entropy_2bit:.6f} bits / 2 bits")
    print(f"  **True Entropy Rate per bit**: {block_entropy_2bit / 2.0:.6f} bits/bit")
    print("==================================================================\n")

# Run for both configurations
calculate_true_bit_entropy("Tek002-500mv_ALL.csv")
calculate_true_bit_entropy("Tek002-max7dbm_ALL.csv")
