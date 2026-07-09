import pandas as pd
import numpy as np
import secrets
from randextract import ToeplitzHashing

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "Tek002-max7dbm_ALL.csv"
OUTPUT_FILE = "toeplitz_library_bits.bin"
BITS_PER_SAMPLE = 4      # Extract bottom 4 LSBs
COMPRESSION_RATIO = 0.75  # Secure extraction fraction

print("=== Starting Official 'randextract' Library Pipeline ===")

# 1. Locate header dynamically
with open(INPUT_FILE, 'r') as f:
    for idx, line in enumerate(f):
        if "TIME" in line:
            header_row = idx
            break

# 2. Load the raw analog voltages from CH3
df = pd.read_csv(INPUT_FILE, skiprows=header_row)
ch3_raw = df['CH3'].values
print(f"Loaded {len(ch3_raw):,} raw points from CH3.")

# 3. Quantization
v_min, v_max = np.min(ch3_raw), np.max(ch3_raw)
quantized = ((ch3_raw - v_min) / (v_max - v_min) * 4095).astype(np.uint16)

# 4. Gather the multi-bit LSB pool into a NumPy array
raw_bit_pool = []
for val in quantized:
    for b in range(BITS_PER_SAMPLE):
        raw_bit_pool.append((val >> b) & 1)

raw_bits_array = np.array(raw_bit_pool, dtype=np.uint8)

# Calculate Lengths
num_raw_bits = len(raw_bits_array)
num_output_bits = int(num_raw_bits * COMPRESSION_RATIO)
num_output_bits = num_output_bits - (num_output_bits % 8) # Align to byte boundary

print(f"-> Raw bits gathered: {num_raw_bits:,}")
print(f"-> Target output bits: {num_output_bits:,}")

# 5. Generate the physical random seed
seed_len = num_raw_bits + num_output_bits - 1
seed_bytes = secrets.token_bytes((seed_len + 7) // 8)
seed_array = np.unpackbits(np.frombuffer(seed_bytes, dtype=np.uint8))[:seed_len]

print("-> Passing data into randextract.ToeplitzHashing...")

# --- THE FIX: Passing INTEGERS (num_raw_bits) instead of arrays ---
extractor = ToeplitzHashing(num_raw_bits, num_output_bits)

# Execute extraction by passing the DATA arrays
extracted_bits_array = extractor.extract(raw_bits_array, seed_array)
extracted_bits = np.array(extracted_bits_array, dtype=np.uint8)
# ------------------------------------------------------------------

# 6. Pack into standard 8-bit bytes and save to disk
bit_matrix = extracted_bits.reshape(-1, 8)
packed_bytes = np.packbits(bit_matrix)

with open(OUTPUT_FILE, "wb") as f:
    f.write(packed_bytes.tobytes())

# ==========================================
# NIST CONFIGURATION & MATH CHECK
# ==========================================
NIST_N = 1000000
NIST_M = len(extracted_bits) // NIST_N

print(f"\n================ EXTRACTION SUCCESS ================")
print(f" Extracted Pure Bits: {len(extracted_bits):,}")
print(f" Saved Binary File:   {OUTPUT_FILE}")
print("====================================================\n")

print("👉 TO RUN IN NIST, TYPE THIS EXACT COMMAND:")
print(f" ./assess {NIST_N}")
print(f"\n Then input these settings when prompted:")
print(f" 1. Select File Format: 0 (Binary Stream)")
print(f" 2. Length of Bit Stream (n): {NIST_N}")
print(f" 3. Number of Bit Streams (m): {NIST_M}")

if NIST_M < 10:
    print("\n⚠️  NOTICE REGARDING YOUR 10-STREAM REQUIREMENT ⚠️")
    print(f" Your current CSV file only provided enough data for {NIST_M} streams of 1,000,000.")
    print(" To reach 10 streams (10,000,000 bits), you will need to process a much larger CSV file")
    print(" or combine multiple CSV files together before extraction.")
print("====================================================")
