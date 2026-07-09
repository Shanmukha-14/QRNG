import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_ch3_psd():
    # 1. Define filenames
    file_500mv = "Tek002-500mv_ALL.csv"
    file_7dbm = "Tek002-max7dbm_ALL.csv"
    file_elec = "Tek001_ALL_elec.csv"
    # Sampling parameters from the Tektronix header
    sampling_interval = 8.00e-11 
    sampling_frequency = 1.0 / sampling_interval  # ~12.5 GHz
    
    print("Loading CH3 from 500mV file...")
    df_500 = pd.read_csv(file_500mv, skiprows=17)
    ch3_500 = df_500['CH3'].values
    
    print("Loading CH3 from 7dBm file...")
    df_7dbm = pd.read_csv(file_7dbm, skiprows=17)
    ch3_7dbm = df_7dbm['CH3'].values

    print("Loading CH3 from electrical file...")
    df_electric = pd.read_csv(file_elec, skiprows=17)
    ch3_elec = df_electric['CH3'].values
    
    # 2. Compute Power Spectral Density via Welch's method
    print("Computing PSD for CH3...")
    freqs_500, psd_500 = welch(ch3_500, fs=sampling_frequency, nperseg=2048)
    freqs_7dbm, psd_7dbm = welch(ch3_7dbm, fs=sampling_frequency, nperseg=2048)
    freqs_elec, psd_elec = welch(ch3_elec, fs=sampling_frequency, nperseg=2048)
    # 3. Create the log-log plot
    plt.figure(figsize=(11, 6))
    
    plt.loglog(freqs_500, psd_500, label='CH3: 500mV Configuration', color='#d62728', alpha=0.8)
    plt.loglog(freqs_7dbm, psd_7dbm, label='CH3: max7dBm Configuration', color='#9467bd', alpha=0.8)
    plt.loglog(freqs_elec, psd_elec, label='CH3: 500mV Configuration', color="#50d627", alpha=0.8)
    plt.title('Power Spectral Density (PSD) Analysis — Channel 3', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power Spectral Density ($V^2/Hz$)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    # Save the output plot
    output_filename = "CH3_noise_psd_analysis.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()
    
    print(f"Done! The plot for your postdoc has been saved as '{output_filename}'")

# Execute the script
plot_ch3_psd()
