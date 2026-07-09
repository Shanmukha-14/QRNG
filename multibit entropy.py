import numpy as np
import pandas as pd

def calculate_paper_style_entropy(file_path):
    print(f"Loading CH3 from {file_path}...")
    
    # Locate header dynamically to avoid KeyErrors
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if "TIME" in line:
                header_row = idx
                break
    df = pd.read_csv(file_path, skiprows=header_row)
    ch3_raw = df['CH3'].values
    
    # 1. Map analog voltages to full 12-bit integers (0 to 4095)
    v_min, v_max = np.min(ch3_raw), np.max(ch3_raw)
    quantized_words = ((ch3_raw - v_min) / (v_max - v_min) * 4095).astype(np.uint16)
    
    # 2. Count frequencies of every active ADC bin
    bins, counts = np.unique(quantized_words, return_counts=True)
    probabilities = counts / len(quantized_words)
    
    # 3. Paper Metric A: Shannon Entropy of the Multi-Bit Distribution
    shannon_word_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    
    # 4. Paper Metric B: Shannon Min-Entropy (H_min = -log2(P_max))
    max_prob = np.max(probabilities)
    h_min = -np.log2(max_prob)
    
    print(f"\n================ 12-BIT MULTI-BIT REPORT: {file_path} ================")
    print(f"  Total Active ADC Bins Used:        {len(bins)} out of 4096")
    print(f"  Peak Bin Probability (P_max):      {max_prob:.4f}")
    print(f"  ----------------------------------------------------------------")
    print(f"  **Shannon Word Entropy**:          {shannon_word_entropy:.4f} bits/sample")
    print(f"  **Shannon Min-Entropy (H_min)**:    {h_min:.4f} bits/sample")
    print(f"  (Theoretical maximum limit for your 12-bit ADC is 12.0000)")
    print("======================================================================\n")

# Execute for your configurations
calculate_paper_style_entropy("Tek002-500mv_ALL.csv")
calculate_paper_style_entropy("Tek002-max7dbm_ALL.csv")
