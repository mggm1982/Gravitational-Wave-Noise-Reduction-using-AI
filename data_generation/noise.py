import sys
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

print("Step 2: Acquiring Detector Noise...")

try:
    start_time = '2017-08-01 10:00:00'
    end_time = '2017-08-01 10:17:04'
    detector = 'H1'

    print(f"Fetching {detector} data from {start_time} to {end_time}...")

    noise_data = TimeSeries.fetch_open_data(
        detector, start_time, end_time, cache=True)

    print("Data fetched successfully!")
    print(f"Data details: {noise_data}")

    output_filename = '../../data/raw/H1_noise.hdf5'
    noise_data.write(output_filename, format='hdf5', overwrite=True)

    print(f"Noise data saved to {output_filename}")

    print("Plotting first 5 seconds of noise...")

    crop_start = noise_data.t0.value
    crop_end = noise_data.t0.value + 5

    plot = noise_data.crop(crop_start, crop_end).plot()

    plt.title("LIGO H1 Detector Noise (First 5 Seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain (Amplitude)")

    plt.savefig('../../results/H1_noise_sample.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved as 'results/H1_noise_sample.png'")
    plt.close()

    print("\nStep 2 Complete. You now have the file 'H1_noise.hdf5'.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your internet connection and if all libraries are installed.")
    sys.exit(1)
