import pandas as pd
import numpy as np

def calculate_qcnr_for_ch3():
    # 1. Define file paths
    file_elec = "Tek001_ALL_elec.csv"       # Pure classical electronic noise floor
    file_500mv = "Tek002-500mv_ALL.csv"     # Total noise at 500mV
    file_7dbm = "Tek002-max7dbm_ALL.csv"    # Total noise at max 7dBm

    # Helper to load CH3 safely avoiding KeyErrors
    def get_ch3_data(file_path):
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if "TIME" in line:
                    header_row = idx
                    break
        df = pd.read_csv(file_path, skiprows=header_row)
        return df['CH3'].values

    print("Loading datasets...")
    ch3_elec = get_ch3_data(file_elec)
    ch3_500mv = get_ch3_data(file_500mv)
    ch3_7dbm = get_ch3_data(file_7dbm)

    # 2. Calculate the variance (\sigma^2) for each file
    # Variance represents the total electrical noise power of the signal
    var_classical = np.var(ch3_elec)
    var_total_500 = np.var(ch3_500mv)
    var_total_7dbm = np.var(ch3_7dbm)

    print("\n================ STATISTICAL POWER (CH3) ================")
    print(f"Classical Noise Power (\u03c3\u00b2_elec):  {var_classical:.8e} V\u00b2")
    print(f"Total Noise Power (\u03c3\u00b2_500mv):    {var_total_500:.8e} V\u00b2")
    print(f"Total Noise Power (\u03c3\u00b2_7dbm):     {var_total_7dbm:.8e} V\u00b2")

    # 3. Calculate QCNR for 500mV File
    var_quantum_500 = var_total_500 - var_classical
    if var_quantum_500 > 0:
        qcnr_linear_500 = var_quantum_500 / var_classical
        qcnr_db_500 = 10 * np.log10(qcnr_linear_500)
    else:
        qcnr_linear_500, qcnr_db_500 = 0, float('-inf')

    # 4. Calculate QCNR for max7dBm File
    var_quantum_7dbm = var_total_7dbm - var_classical
    if var_quantum_7dbm > 0:
        qcnr_linear_7dbm = var_quantum_7dbm / var_classical
        qcnr_db_7dbm = 10 * np.log10(qcnr_linear_7dbm)
    else:
        qcnr_linear_7dbm, qcnr_db_7dbm = 0, float('-inf')

    print("\n================ QUANTUM-TO-CLASSICAL RATIO (QCNR) ================")
    print(f"--- 500mV Configuration ---")
    print(f"  Isolated Quantum Noise (\u03c3\u00b2_quantum): {max(0, var_quantum_500):.8e} V\u00b2")
    print(f"  QCNR (Linear Ratio):              {qcnr_linear_500:.4f}")
    print(f"  QCNR (in dB):                     {qcnr_db_500:.2f} dB")
    
    print(f"\n--- max7dBm Configuration ---")
    print(f"  Isolated Quantum Noise (\u03c3\u00b2_quantum): {max(0, var_quantum_7dbm):.8e} V\u00b2")
    print(f"  QCNR (Linear Ratio):              {qcnr_linear_7dbm:.4f}")
    print(f"  QCNR (in dB):                     {qcnr_db_7dbm:.2f} dB")
    print("====================================================================")

# Execute
calculate_qcnr_for_ch3()
