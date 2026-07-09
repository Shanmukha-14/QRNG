import os
import pandas as pd
import numpy as np
import secrets
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import welch
from randextract import ToeplitzHashing

# ==============================================================================
# PATHS AND PARAMETERS
# ==============================================================================
DOWNLOADS_DIR = r"C:\Users\HP\Downloads"      #path to make the plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Input CSV files for analysis
POWER_GROUPS = {
    "Elec_Noise": ["Tek000_ALL.csv", "Tek001_ALL.csv", "Tek002_ALL.csv"],
    "0.5_mW":     ["Tek003_ALL.csv", "Tek004_ALL.csv", "Tek005_ALL.csv"],
    "1.0_mW":     ["Tek006_ALL.csv", "Tek007_ALL.csv", "Tek008_ALL.csv"],
    "2.0_mW":     ["Tek009_ALL.csv", "Tek010_ALL.csv", "Tek011_ALL.csv"],
    "3.0_mW":     ["Tek012_ALL.csv", "Tek013_ALL.csv", "Tek014_ALL.csv"],
    "4.0_mW":     ["Tek015_ALL.csv", "Tek016_ALL.csv", "Tek017_ALL.csv"],
    "5.0_mW":     ["Tek018_ALL.csv", "Tek019_ALL.csv", "Tek020_ALL.csv"],
    "6.5_mW_Max": ["Tek021_ALL.csv", "Tek022_ALL.csv", "Tek023_ALL.csv"]
}

SAMPLING_INTERVAL = 8.00e-11  # ~12.5 GHz
SAMPLING_FREQUENCY = 1.0 / SAMPLING_INTERVAL  
BITS_PER_SAMPLE = 4          
COMPRESSION_RATIO = 0.75      
NIST_N = 1000000             

print("=== Starting Official Multi-Case Integrated Physics & Information-Theory Pipeline ===\n")

baseline_variance = None

for group_name, file_list in POWER_GROUPS.items():
    print(f"\n====================================================================")
    print(f"PROCESSING POWER LEVEL DOMAIN: {group_name.upper()}")
    print(f"====================================================================")
    
    group_voltages = []
    
    # Concatenate the files in this group
    for base_filename in file_list:
        full_path = os.path.join(DOWNLOADS_DIR, base_filename)
        if not os.path.exists(full_path):
            alt_path = os.path.join(DOWNLOADS_DIR, base_filename.replace(".csv", " (1).csv"))
            if os.path.exists(alt_path):
                full_path = alt_path
                
        if not os.path.exists(full_path):
            print(f"⚠️  Missing component file: {base_filename}")
            continue
            
        try:
            header_row = 0
            with open(full_path, 'r') as f:
                for idx, line in enumerate(f):
                    if "TIME" in line:
                        header_row = idx
                        break
            df = pd.read_csv(full_path, skiprows=header_row)
            group_voltages.extend(df['CH3'].values)
        except Exception as e:
            print(f"❌ Failed to parse {base_filename}: {e}")
            
    if len(group_voltages) == 0:
        print(f"❌ No valid raw data recovered for group {group_name}.")
        continue
        
    ch3_raw = np.array(group_voltages)
    total_samples = len(ch3_raw)
    current_variance = np.var(ch3_raw)
    
    print(f"Loaded {total_samples:,} total continuous data points across group files.")
    
    # --------------------------------------------------------------------------
    # CODE 1 OUTPUT COPY: DISTRIBUTION AND GAUSSIAN FIT / ACF PLOTTING
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    count, bins, ignored = ax1.hist(ch3_raw, bins=100, density=True, alpha=0.6, color='g')
    mu, sigma = norm.fit(ch3_raw)
    ax1.plot(bins, norm.pdf(bins, mu, sigma), 'r--', linewidth=2)
    ax1.set_title(f'CH3 Distribution $\mu$={mu:.4f}, $\sigma$={sigma:.4f}')
    ax1.set_xlabel('Voltage (V)')
    
    lags = 100
    sub_sampled_series = pd.Series(ch3_raw[:min(20000, total_samples)])
    acf_array = [sub_sampled_series.autocorr(lag=i) for i in range(lags)]
    ax2.stem(range(lags), acf_array)
    ax2.set_title('Autocorrelation Function (ACF)')
    ax2.set_xlabel('Lags')
    ax2.set_ylabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{group_name}_distribution_acf.png"), dpi=300)
    plt.close()
    
    # --------------------------------------------------------------------------
    # CODE 2 OUTPUT COPY: POWER SPECTRAL DENSITY ANALYSIS VIA WELCH
    # --------------------------------------------------------------------------
    freqs, psd = welch(ch3_raw, fs=SAMPLING_FREQUENCY, nperseg=2048)
    plt.figure(figsize=(11, 6))
    plt.loglog(freqs, psd, label=f'CH3: {group_name}', color='#d62728', alpha=0.8)
    plt.title(f'Power Spectral Density (PSD) Analysis — Channel 3 ({group_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power Spectral Density ($V^2/Hz$)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    psd_plot_path = os.path.join(PLOTS_DIR, f"{group_name}_noise_psd_analysis.png")
    plt.savefig(psd_plot_path, dpi=300)
    plt.close()
    print(f"Done! The PSD analysis plot has been saved as '{psd_plot_path}'")
    
    # --------------------------------------------------------------------------
    # CODE 3 OUTPUT COPY: QUANTUM-TO-CLASSICAL RATIO (QCNR) REPORT
    # --------------------------------------------------------------------------
    print("\n================ STATISTICAL POWER (CH3) ================")
    if group_name == "Elec_Noise":
        baseline_variance = current_variance
        print(f"Classical Noise Power (σ²_elec):  {baseline_variance:.8e} V²")
    else:
        print(f"Classical Noise Power (σ²_elec):  {baseline_variance:.8e} V²" if baseline_variance else "Classical Noise Power: Not Available")
        print(f"Total Noise Power (σ²_{group_name}):     {current_variance:.8e} V²")
        
    print("\n================ QUANTUM-TO-CLASSICAL RATIO (QCNR) ================")
    if group_name == "Elec_Noise":
        print(" [!] Current set defined as baseline floor. QCNR is relative 0 dB.")
    else:
        if baseline_variance is not None:
            var_quantum = current_variance - baseline_variance
            if var_quantum > 0:
                qcnr_linear = var_quantum / baseline_variance
                qcnr_db = 10 * np.log10(qcnr_linear)
            else:
                var_quantum, qcnr_linear, qcnr_db = 0, 0, float('-inf')
            print(f"--- {group_name} Configuration ---")
            print(f"  Isolated Quantum Noise (σ²_quantum): {max(0, var_quantum):.8e} V²")
            print(f"  QCNR (Linear Ratio):              {qcnr_linear:.4f}")
            print(f"  QCNR (in dB):                     {qcnr_db:.2f} dB")
        else:
            print("  QCNR Calculation Pending baseline parameter assignment.")
    print("====================================================================")

    # --------------------------------------------------------------------------
    # CODE 4 OUTPUT COPY: SHANNON ENTROPY REPRODUCIBILITY BLOCK
    # --------------------------------------------------------------------------
    v_min, v_max = np.min(ch3_raw), np.max(ch3_raw)
    quantized = ((ch3_raw - v_min) / (v_max - v_min) * 4095).astype(np.uint16)
    
    lsb_bits = quantized & 1
    total_bits_analyzed = len(lsb_bits)
    count_ones = np.sum(lsb_bits)
    count_zeros = total_bits_analyzed - count_ones
    p0 = count_zeros / total_bits_analyzed
    p1 = count_ones / total_bits_analyzed
    shannon_entropy = - (p0 * np.log2(p0 + 1e-12) + p1 * np.log2(p1 + 1e-12))
    
    print(f"\n--- Entropy Report for {group_name} ---")
    print(f"  Total bits analyzed: {total_bits_analyzed:,}")
    print(f"  Zeros '0': {count_zeros:,} ({p0*100:.2f}%)")
    print(f"  Ones '1':  {count_ones:,} ({p1*100:.2f}%)")
    print(f"  **Entropy per bit**: {shannon_entropy:.6f} bits")
    print("-" * 50)

    # --------------------------------------------------------------------------
    # CODE 5 OUTPUT COPY: TRACE HIDDEN MEMORY BLOCK ENTROPY RATE
    # --------------------------------------------------------------------------
    print(f"Analyzing hidden memory in {group_name}...")
    bit_series = pd.Series(lsb_bits)
    bit_corr = bit_series.autocorr(lag=1)
    
    bits_t = lsb_bits[:-1]
    bits_next = lsb_bits[1:]
    pairs = (bits_t << 1) + bits_next
    total_pairs = len(pairs)
    counts = np.bincount(pairs, minlength=4)
    probs = counts / total_pairs
    block_entropy_2bit = -np.sum([p * np.log2(p + 1e-12) for p in probs])
    
    print(f"\n================ HIDDEN MEMORY REPORT: {group_name} ================")
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

    # --------------------------------------------------------------------------
    # CODE 6 OUTPUT COPY: TOEPLITZ EXTRACTION SUB-PIPELINE
    # --------------------------------------------------------------------------
    raw_bit_pool = []
    for val in quantized:
        for b in range(BITS_PER_SAMPLE):
            raw_bit_pool.append((val >> b) & 1)
    raw_bits_array = np.array(raw_bit_pool, dtype=np.uint8)
    
    num_raw_bits = len(raw_bits_array)
    num_output_bits = int(num_raw_bits * COMPRESSION_RATIO)
    num_output_bits = num_output_bits - (num_output_bits % 8)
    
    print("-> Passing data into randextract.ToeplitzHashing...")
    seed_len = num_raw_bits + num_output_bits - 1
    seed_bytes = secrets.token_bytes((seed_len + 7) // 8)
    seed_array = np.unpackbits(np.frombuffer(seed_bytes, dtype=np.uint8))[:seed_len]
    
    output_bin = f"toeplitz_{group_name}.bin"
    try:
        extractor = ToeplitzHashing(num_raw_bits, num_output_bits)
        extracted_bits = np.array(extractor.extract(raw_bits_array, seed_array), dtype=np.uint8)
        
        packed_bytes = np.packbits(extracted_bits.reshape(-1, 8))
        with open(output_bin, "wb") as f:
            f.write(packed_bytes.tobytes())
            
        NIST_M = len(extracted_bits) // NIST_N
        
        print(f"\n================ EXTRACTION SUCCESS ================")
        print(f" Extracted Pure Bits: {len(extracted_bits):,}")
        print(f" Saved Binary File:   {output_bin}")
        print("====================================================\n")
        print("👉 TO RUN IN NIST, TYPE THIS EXACT COMMAND:")
        print(f" ./assess {NIST_N}")
        print(f"\n Then input these settings when prompted:")
        print(f" 1. Select File Format: 0 (Binary Stream)")
        print(f" 2. Length of Bit Stream (n): {NIST_N}")
        print(f" 3. Number of Bit Streams (m): {NIST_M}")
        print("====================================================")
    except Exception as e:
        print(f"❌ Extraction Module Aborted for {group_name}: {e}")

print("\n======================= SYSTEM EXECUTION END =======================")
