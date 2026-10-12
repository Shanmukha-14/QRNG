import os
import pandas as pd
import numpy as np
import secrets
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import welch
from randextract import ToeplitzHashing

FILES_DIR = "D:/"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================================================================
# CHUNK 1: INPUT POWERS & PIPELINE CONFIGURATION
# ==============================================================================

# 1. Experimental Power Domain Mapping
INPUT_POWERS = {
    "Elec_Noise": ["Tek000_ALL.csv", "Tek001_ALL.csv", "Tek002_ALL.csv"],
    "0.5_mW":     ["Tek003_ALL.csv", "Tek004_ALL.csv", "Tek005_ALL.csv"],
    "1.0_mW":     ["Tek006_ALL.csv", "Tek007_ALL.csv", "Tek008_ALL.csv"],
    "2.0_mW":     ["Tek009_ALL.csv", "Tek010_ALL.csv", "Tek011_ALL.csv"],
    "3.0_mW":     ["Tek012_ALL.csv", "Tek013_ALL.csv", "Tek014_ALL.csv"],
    "4.0_mW":     ["Tek015_ALL.csv", "Tek016_ALL.csv", "Tek017_ALL.csv"],
    "5.0_mW":     ["Tek018_ALL.csv", "Tek019_ALL.csv", "Tek020_ALL.csv"],
    "6.5_mW_Max": ["Tek021_ALL.csv", "Tek022_ALL.csv", "Tek023_ALL.csv"]
}

# ==============================================================================
# CHUNK 2: SYSTEM CONFIGURATION & GLOBAL PARAMETERS
# ==============================================================================

# Hardware Oscilloscope Settings
SAMPLING_INTERVAL = 8.00e-11   # 80 picoseconds (from scope header)
SAMPLING_FREQ = 1.0 / SAMPLING_INTERVAL  # 12.5 GHz Nyquist sampling rate
ADC_BITS = 12                  # 12-bit ADC (quantization levels: 0 to 4095)

# Extraction Algorithm Settings
BITS_PER_SAMPLE = 4      # Bottom 4 LSBs per sample (captures quantum noise floor)
COMPRESSION_RATIO = 0.75  # Toeplitz privacy amplification factor (50% yield)
NIST_STREAM_SIZE = 1000000  # 1 Megabit sequence size for NIST SP 800-22

# Sanity check: a real scope record is ~1e6 points; fewer than this means the CSV was misread
MIN_SAMPLES_PER_FILE = 1000

print("Chunk 2: Global parameters and power group mappings loaded successfully!")

# ==============================================================================
# CHUNK 3: DATA LOADING & PREPROCESSING
# ==============================================================================

def load_CH3_data(file_path):
    """
    Dynamically scans a Tektronix CSV file to locate the data boundary,
    parses CH3 voltage values safely, and returns them as a clean NumPy array.
    """
    # The scope writes metadata lines (record length, trigger point, ...) above the
    # waveform, and one of them can mention "CH3". The real column-header row is the
    # first line that has BOTH "TIME" and "CH3"; matching on "CH3" alone picks a
    # metadata line and reads only a handful of header numbers as "voltages".
    lines = []
    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            lines.append(line)
            if "TIME" in line and "CH3" in line:
                break

    if lines and "TIME" in lines[-1] and "CH3" in lines[-1]:
        header_row = len(lines) - 1
    else:
        # Fallback: first line containing "TIME" (the rule the reference code used)
        header_row = 0
        with open(file_path, 'r', errors='replace') as f:
            for idx, line in enumerate(f):
                if "TIME" in line:
                    header_row = idx
                    break

    # Read CSV starting from detected header row; low_memory=False prevents mixed-type warnings
    df = pd.read_csv(file_path, skiprows=header_row, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]

    if 'CH3' not in df.columns:
        raise KeyError(f"Column 'CH3' missing in {file_path}. Found: {list(df.columns)}")

    # Convert CH3 column to numeric values safely and drop any residual metadata/NaN rows
    ch3_series = pd.to_numeric(df['CH3'], errors='coerce').dropna()
    ch3_data = ch3_series.values.astype(np.float64)

    # Fail loudly instead of silently analysing a few header numbers
    if len(ch3_data) < MIN_SAMPLES_PER_FILE:
        raise ValueError(
            f"Only {len(ch3_data)} numeric CH3 values found (header detected at line "
            f"{header_row}, columns {list(df.columns)[:6]}); expected at least "
            f"{MIN_SAMPLES_PER_FILE}. The header row was probably misdetected."
        )

    return ch3_data


def load_power_group_data(power_group):
    """
    Loads and concatenates raw CH3 voltage vectors for all CSV files
    belonging to a specified optical power group.
    Missing files are skipped with a warning (the " (1).csv" browser-duplicate
    name is tried as well); an error is raised only if nothing could be loaded.
    """
    if power_group not in INPUT_POWERS:
        raise ValueError(f"Power group '{power_group}' not found in INPUT_POWERS.")

    aggregated_data = []
    for file_name in INPUT_POWERS[power_group]:
        file_path = os.path.join(FILES_DIR, file_name)

        if not os.path.exists(file_path):
            alt_path = os.path.join(FILES_DIR, file_name.replace(".csv", " (1).csv"))
            if os.path.exists(alt_path):
                file_path = alt_path

        if not os.path.exists(file_path):
            print(f"  [!] Missing component file: {file_name}")
            continue

        try:
            ch3_data = load_CH3_data(file_path)
            aggregated_data.append(ch3_data)
        except Exception as e:
            print(f"  [X] Failed to parse {file_name}: {e}")

    if len(aggregated_data) == 0:
        raise FileNotFoundError(f"No valid raw data recovered for group '{power_group}'.")

    return np.concatenate(aggregated_data)

print("Chunk 3: Loading and group aggregation functions defined successfully!")

# ==============================================================================
# CHUNK 4: DATA VISUALIZATION (styled to match the reference code's plots)
# ==============================================================================

def gaussian_plot(voltages, power_group):
    # Green histogram, red dashed Gaussian fit, mu/sigma in the title
    plt.figure(figsize=(6, 5))

    count, bins, ignored = plt.hist(voltages, bins=100, density=True, alpha=0.6, color='g')
    mu, sigma = norm.fit(voltages)
    plt.plot(bins, norm.pdf(bins, mu, sigma), 'r--', linewidth=2)

    plt.title(rf'CH3 Distribution $\mu$={mu:.4f}, $\sigma$={sigma:.4f}  ({power_group})')
    plt.xlabel('Voltage (V)')
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"Gaussian_{power_group}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def autocorr(voltages, power_group, max_lags = 100):
    # Stem plot computed on the first 20,000 samples, default stem colours
    sub_sampled_series = pd.Series(voltages[:min(20000, len(voltages))])
    acf_array = [sub_sampled_series.autocorr(lag=i) for i in range(max_lags)]

    plt.figure(figsize=(6, 5))
    plt.stem(range(max_lags), acf_array)

    plt.title(f'Autocorrelation Function (ACF)  ({power_group})')
    plt.xlabel('Lags')
    plt.ylabel('Correlation Coefficient')
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"ACF_{power_group}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_welch_psd(voltages, power_group):
    nperseg_val = min(2048, len(voltages))

    if nperseg_val < 10:
        print(f"Warning: Not enough data points for Welch PSD estimation in {power_group}. Skipping PSD plot.")
        return
    freqs, psd = welch(voltages, fs=SAMPLING_FREQ, nperseg=nperseg_val)

    # Log-log axes in Hz, red trace, bold title, legend
    plt.figure(figsize=(11, 6))
    plt.loglog(freqs, psd, label=f'CH3: {power_group}', color='#d62728', alpha=0.8)

    plt.title(f'Power Spectral Density (PSD) Analysis — Channel 3 ({power_group})', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power Spectral Density ($V^2/Hz$)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"PSD_{power_group}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Done! The PSD analysis plot has been saved as '{plot_path}'")

def run_physical_diagnostics(power_group, data=None):
    """
    Executes the full suite of physical diagnostics for a given optical power group:
    1. Loads and aggregates CH3 voltage data (or reuses `data` if already loaded).
    2. Generates Gaussian fit, autocorrelation, and Welch PSD plots.
    """
    print(f"Running diagnostics for power group: {power_group}")
    if data is None:
        data = load_power_group_data(power_group)

    print(f"Loaded {len(data):,} total continuous data points across group files.")

    print(f"Generating Gaussian fit plot for {power_group}...")
    gaussian_plot(data, power_group)

    print(f"Generating autocorrelation plot for {power_group}...")
    autocorr(data, power_group)

    print(f"Generating Welch PSD plot for {power_group}...")
    plot_welch_psd(data, power_group)

    print(f"Diagnostics completed for power group: {power_group}. Plots saved in '{PLOTS_DIR}' directory.")

print("Chunk 4: Data visualization functions defined successfully!")

# ==============================================================================
# CHUNK 5: QCNR
# ==============================================================================

def qcnr(voltages, power_group, elec_noise_data=None):
    """
    Quantum-to-classical noise ratio (linear).
    Pass `elec_noise_data` to avoid reloading the baseline from disk on every call.
    """
    if elec_noise_data is None:
        elec_noise_data = load_power_group_data("Elec_Noise")
    classical_variance = np.var(elec_noise_data)

    total_variance = np.var(voltages)

    # Quantum noise variance is the difference between total and classical variances
    quantum_variance = total_variance - classical_variance

    if quantum_variance <= 0:
        print(f"Warning: Non-positive quantum variance for {power_group}. QCNR cannot be computed.")
        return None

    qcnr_value = quantum_variance / classical_variance
    return qcnr_value

def qcnr_report(voltages, power_group, elec_noise_data):
    """Prints the statistical-power and QCNR report (linear + dB) like the reference code."""
    classical_variance = np.var(elec_noise_data)
    total_variance = np.var(voltages)

    print("\n================ STATISTICAL POWER (CH3) ================")
    print(f"Classical Noise Power (sigma^2_elec):  {classical_variance:.8e} V^2")
    if power_group == "Elec_Noise":
        print(" [!] Current set defined as baseline floor. QCNR is relative 0 dB.")
        print("====================================================================")
        return None, float('-inf'), 0.0

    print(f"Total Noise Power (sigma^2_{power_group}):     {total_variance:.8e} V^2")

    print("\n================ QUANTUM-TO-CLASSICAL RATIO (QCNR) ================")
    var_quantum = total_variance - classical_variance
    qcnr_linear = qcnr(voltages, power_group, elec_noise_data)
    if qcnr_linear is None:
        var_quantum, qcnr_lin_out, qcnr_db = 0.0, 0.0, float('-inf')
    else:
        qcnr_lin_out = qcnr_linear
        qcnr_db = 10 * np.log10(qcnr_linear)

    print(f"--- {power_group} Configuration ---")
    print(f"  Isolated Quantum Noise (sigma^2_quantum): {max(0.0, var_quantum):.8e} V^2")
    print(f"  QCNR (Linear Ratio):              {qcnr_lin_out:.4f}")
    print(f"  QCNR (in dB):                     {qcnr_db:.2f} dB")
    print("====================================================================")
    return qcnr_linear, qcnr_db, max(0.0, var_quantum)

# ==============================================================================
# CHUNK 6: QUANTITATIVE ENTROPY EVALUATION
# ==============================================================================

def extract_lsbs(voltages, bits_per_sample=BITS_PER_SAMPLE):

    # For a uniform distribution across 2^bits states, max entropy is 'bits_per_sample'
    # We use empirical frequency analysis of the extracted LSB integers:
    v_min, v_max = np.min(voltages), np.max(voltages)
    normalized = (voltages - v_min) / (v_max - v_min)
    adc_ints = np.floor(normalized * ((1 << ADC_BITS) - 1)).astype(np.uint32)
    bitmask = (1 << bits_per_sample) - 1
    lsb_values = adc_ints & bitmask


    # Vectorized bit unpacking (MSB-first within each sample), same order as the
    # original per-sample format() loop but orders of magnitude faster
    shifts = np.arange(bits_per_sample - 1, -1, -1, dtype=np.uint32)
    raw_bits = ((lsb_values[:, None] >> shifts) & 1).astype(np.uint8).ravel()

    return raw_bits, lsb_values

def shannon_entropy(lsb_values, bits_per_sample=BITS_PER_SAMPLE):
    num_states = 1 << bits_per_sample  # 2^bits_per_sample
    counts = np.bincount(lsb_values, minlength=num_states)
    probs = counts / np.sum(counts)

    # Filter out zero probabilities to avoid log2(0)
    non_zero_probs = probs[probs > 0]
    shannon_h = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    print(f"Shannon entropy calculated: {shannon_h:.4f} bits per sample.")

    return shannon_h

def min_entropy(lsb_values, bits_per_sample=BITS_PER_SAMPLE):
    """Min-entropy per sample: -log2(max probability). The right quantity for choosing
    a safe Toeplitz compression ratio (Shannon entropy overestimates unpredictability)."""
    num_states = 1 << bits_per_sample
    counts = np.bincount(lsb_values, minlength=num_states)
    probs = counts / np.sum(counts)
    h_min = -np.log2(np.max(probs))
    print(f"Min-entropy calculated: {h_min:.4f} bits per sample "
          f"(max possible {bits_per_sample}; compression ratio {COMPRESSION_RATIO} "
          f"keeps {COMPRESSION_RATIO * bits_per_sample:.2f} bits/sample, "
          f"{'OK' if h_min >= COMPRESSION_RATIO * bits_per_sample else 'TOO AGGRESSIVE'}).")
    return h_min

def von_neumann_entropy(voltages, Elec_Noise):

    var_total = np.var(voltages)
    var_elec = np.var(Elec_Noise)

    if var_elec > 0:
        n_thermal = var_elec / max(1e-12, (var_total - var_elec))  # Avoid division by zero
    else:
        n_thermal = 0.0

    if n_thermal > 0:
        von_neumann_h = (n_thermal + 1) * np.log2(n_thermal + 1) - n_thermal * np.log2(n_thermal)
    else:
        von_neumann_h = 0.0

    print(f"Von Neumann entropy calculated: {von_neumann_h:.4f} bits per sample.")

    return von_neumann_h

def single_bit_entropy_report(lsb_ints, power_group):
    """Shannon entropy of the single least-significant bit (reference code's Code 4 block)."""
    lsb_bits = (lsb_ints & 1).astype(np.uint8)
    total_bits_analyzed = len(lsb_bits)
    count_ones = int(np.sum(lsb_bits))
    count_zeros = total_bits_analyzed - count_ones
    p0 = count_zeros / total_bits_analyzed
    p1 = count_ones / total_bits_analyzed
    h_bit = -(p0 * np.log2(p0 + 1e-12) + p1 * np.log2(p1 + 1e-12))

    print(f"\n--- Entropy Report for {power_group} ---")
    print(f"  Total bits analyzed: {total_bits_analyzed:,}")
    print(f"  Zeros '0': {count_zeros:,} ({p0*100:.2f}%)")
    print(f"  Ones '1':  {count_ones:,} ({p1*100:.2f}%)")
    print(f"  **Entropy per bit**: {h_bit:.6f} bits")
    print("-" * 50)
    return lsb_bits, h_bit

def hidden_memory_report(lsb_bits, power_group):
    """Lag-1 bit autocorrelation, 2-bit pair frequencies and block entropy
    (reference code's Code 5 block)."""
    print(f"Analyzing hidden memory in {power_group}...")
    bit_corr = pd.Series(lsb_bits).autocorr(lag=1)

    pairs = (lsb_bits[:-1].astype(np.uint8) << 1) + lsb_bits[1:]
    total_pairs = len(pairs)
    counts = np.bincount(pairs, minlength=4)
    probs = counts / total_pairs
    block_entropy_2bit = -np.sum([p * np.log2(p + 1e-12) for p in probs])

    print(f"\n================ HIDDEN MEMORY REPORT: {power_group} ================")
    print(f"  Consecutive Bit Autocorrelation (Lag 1): {bit_corr:.4f}")
    print(f"  (Target for true QRNG should be near 0.0000)")
    print(f"\n  Pair Distribution Frequencies:")
    for i, pat in enumerate(["00", "01", "10", "11"]):
        print(f"    Pattern '{pat}': {counts[i]:,} ({probs[i]*100:.2f}%) [Target: 25%]")
    print(f"\n  **True 2-Bit Block Entropy**: {block_entropy_2bit:.6f} bits / 2 bits")
    print(f"  **True Entropy Rate per bit**: {block_entropy_2bit / 2.0:.6f} bits/bit")
    print("==================================================================\n")
    return bit_corr, block_entropy_2bit / 2.0

print("Chunk 5/6: QCNR and entropy functions defined successfully!")

# ==============================================================================
# CHUNK 7: TOEPLITZ PRIVACY AMPLIFICATION & NIST STREAM GENERATION
# ==============================================================================

def generate_nist_bitstream(raw_bits, compression_ratio=COMPRESSION_RATIO, target_size=None):
    """
    Applies Toeplitz Privacy Amplification hashing to distill raw LSB bits
    into a uniformly random, cryptographically secure NIST sequence.

    target_size=None  -> use ALL raw bits (as in the reference code), giving
                         len(raw_bits) * compression_ratio output bits.
    target_size=<int> -> only use enough raw bits to yield that many output bits.
    """
    if target_size is None:
        required_raw_bits = len(raw_bits)
        target_size = int(required_raw_bits * compression_ratio)
    else:
        required_raw_bits = int(target_size / compression_ratio)
        if len(raw_bits) < required_raw_bits:
            print(f"  Warning: Available raw bits ({len(raw_bits):,}) is less than required ({required_raw_bits:,}). Adjusting target size.")
            required_raw_bits = len(raw_bits)
            target_size = int(required_raw_bits * compression_ratio)

    # Output length must be a whole number of bytes so it can be packed to a .bin file
    target_size = target_size - (target_size % 8)

    input_bit_stream = raw_bits[:required_raw_bits]
    num_raw_bits = len(input_bit_stream)

    # Cryptographically secure seed for the Toeplitz matrix: n + m - 1 bits
    seed_len = num_raw_bits + target_size - 1
    seed_bytes = secrets.token_bytes((seed_len + 7) // 8)
    seed_array = np.unpackbits(np.frombuffer(seed_bytes, dtype=np.uint8))[:seed_len]

    # Same call pattern as the reference code (positional lengths, seed passed to extract)
    extractor = ToeplitzHashing(num_raw_bits, target_size)
    final_bits = np.array(extractor.extract(input_bit_stream, seed_array), dtype=np.uint8)

    print(f"  Final NIST Bitstream Generated: {len(final_bits):,} bits secured via Toeplitz Hashing.")

    return final_bits

def save_and_report_nist(final_bits, power_group):
    """Writes the packed .bin file and prints the exact NIST STS instructions."""
    output_bin = f"toeplitz_{power_group}.bin"
    usable = len(final_bits) - (len(final_bits) % 8)
    packed_bytes = np.packbits(final_bits[:usable].reshape(-1, 8))
    with open(output_bin, "wb") as f:
        f.write(packed_bytes.tobytes())

    nist_m = len(final_bits) // NIST_STREAM_SIZE

    print(f"\n================ EXTRACTION SUCCESS ================")
    print(f" Extracted Pure Bits: {len(final_bits):,}")
    print(f" Saved Binary File:   {output_bin}")
    print("====================================================\n")
    print("TO RUN IN NIST, TYPE THIS EXACT COMMAND:")
    print(f" ./assess {NIST_STREAM_SIZE}")
    print(f"\n Then input these settings when prompted:")
    print(f" 1. Select File Format: 0 (Binary Stream)")
    print(f" 2. Length of Bit Stream (n): {NIST_STREAM_SIZE}")
    print(f" 3. Number of Bit Streams (m): {nist_m}")
    print("====================================================")
    return output_bin

# ==============================================================================
# CHUNK 8: MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=== Starting Multi-Case Integrated Physics & Information-Theory Pipeline ===\n")

    # 1. Load the electronic noise baseline once (reference for all powers)
    print("Loading electronic noise baseline...")
    elec_data = load_power_group_data("Elec_Noise")

    summary_results = {}

    # 2. Loop through EVERY group, including the Elec_Noise baseline
    for power_group in INPUT_POWERS:
        print(f"\n" + "=" * 68)
        print(f"PROCESSING POWER LEVEL DOMAIN: {power_group.upper()}")
        print("=" * 68)

        try:
            optical_data = elec_data if power_group == "Elec_Noise" else load_power_group_data(power_group)
        except FileNotFoundError as e:
            print(f"  [X] {e}")
            continue

        # Guard: a power level whose noise variance is bit-identical to the baseline
        # almost certainly means the same (or wrongly parsed) data was loaded
        if power_group != "Elec_Noise" and np.var(optical_data) == np.var(elec_data):
            print(f"  [!] WARNING: {power_group} has exactly the same variance as Elec_Noise. "
                  f"Check the input files / parsing before trusting these results.")

        # Physical diagnostics (Gaussian, ACF, PSD plots)
        run_physical_diagnostics(power_group, data=optical_data)

        # QCNR report (linear + dB + isolated quantum variance)
        qcnr_lin, qcnr_db, var_q = qcnr_report(optical_data, power_group, elec_data)

        # Bit extraction and entropy metrics
        raw_bits, lsb_ints = extract_lsbs(optical_data, BITS_PER_SAMPLE)
        shannon_val = shannon_entropy(lsb_ints, BITS_PER_SAMPLE)
        min_ent_val = min_entropy(lsb_ints, BITS_PER_SAMPLE)
        vn_val = von_neumann_entropy(optical_data, elec_data)

        # Single-bit entropy and hidden-memory checks
        lsb_bits, h_bit = single_bit_entropy_report(lsb_ints, power_group)
        bit_corr, entropy_rate = hidden_memory_report(lsb_bits, power_group)

        # Toeplitz extraction + NIST instructions
        print("-> Passing data into randextract.ToeplitzHashing...")
        try:
            final_bits = generate_nist_bitstream(raw_bits, COMPRESSION_RATIO)
            save_and_report_nist(final_bits, power_group)
            extracted = len(final_bits)
        except Exception as e:
            print(f"[X] Extraction Module Aborted for {power_group}: {e}")
            extracted = 0

        summary_results[power_group] = {
            "QCNR": qcnr_lin,
            "QCNR_dB": qcnr_db,
            "Shannon_Entropy": shannon_val,
            "Min_Entropy": min_ent_val,
            "Von_Neumann_Entropy": vn_val,
            "Bit_Entropy": h_bit,
            "Lag1_Corr": bit_corr,
            "Extracted_Bits": extracted,
        }

    # 3. Final summary table
    print("\n" + "=" * 68)
    print("FINAL SUMMARY")
    print("=" * 68)
    header = f"{'Group':<12}{'QCNR(dB)':>10}{'Shannon':>10}{'MinEnt':>10}{'VonNeu':>10}{'BitH':>10}{'Lag1':>9}{'OutBits':>12}"
    print(header)
    print("-" * len(header))
    for grp, r in summary_results.items():
        db = "base" if grp == "Elec_Noise" else f"{r['QCNR_dB']:.2f}"
        print(f"{grp:<12}{db:>10}{r['Shannon_Entropy']:>10.4f}{r['Min_Entropy']:>10.4f}"
              f"{r['Von_Neumann_Entropy']:>10.4f}{r['Bit_Entropy']:>10.6f}{r['Lag1_Corr']:>9.4f}"
              f"{r['Extracted_Bits']:>12,}")

    print("\n======================= SYSTEM EXECUTION END =======================")
