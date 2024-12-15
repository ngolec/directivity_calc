from pathlib import Path

import numpy as np
from scipy.fft import fft
from scipy.integrate import quad


def interpolate_csv(data, n):
    """
    Reads CSV file, performs linear interpolation, and returns interpolated array.
    Parameters:
        data (numpy structured array): CSV file containing `phi` and `intensity_dB` columns.
        num_points_out (int): Number of uniformly distributed points for interpolation.
    Returns:
        interpolated_array (numpy structured array): Interpolated array with columns `phi` and `intensity_dB`.
    """
    # Extract the phi and intensity_dB columns
    x = data['phi']
    y = data['intensity_dB']

    # Generate n uniformly distributed x values
    x_uniform = np.linspace(x.min(), x.max(), n)

    # Perform linear interpolation
    y_interpolated = np.interp(x_uniform, x, y)

    # Combine x and y into an array
    interpolated_array = np.column_stack((x_uniform, y_interpolated))

    return interpolated_array


def fourier_coefficients(interpolated_data):
    """
    Calculates and returns the Fourier coefficients of a array with evenly spaced x values
    Parameter:
        interpolated_data (numpy structured array): array containing `phi` and `intensity_dB` with evenly spaced 'phi' values
    Returns:
        fourier_coefficients (numpy structured array): The Fourier coefficients of interpolated data
    """
    phi = interpolated_data[:, 0]  # Angles (assume in radians)
    intensity_dB = interpolated_data[:, 1]  # Radiation intensity in dB

    intensity_dB = 10 ** (intensity_dB / 10)

    # Double checks that x axis values are evenly spaced
    dx = np.diff(phi)
    if not np.allclose(dx, dx[0]):
        raise ValueError("The phi values must be evenly spaced.")

    # Calculates Fourier coefficients
    n = len(intensity_dB)
    fourier_coefficients = fft(intensity_dB) / n  # Normalizes coefficients
    return fourier_coefficients


def fourier_series(x, T, fourier_coefficients):
    """
    Parameters:
        x (variable)
        T (float): period of function
        fourier_coefficients (numpy structured array): Fourier coefficients for Fourier series being constructed
    Returns: fourier series
    """
    result = np.real(fourier_coefficients[0])
    if len(fourier_coefficients) % 2 == 0:
        for k in range(1, (len(fourier_coefficients) // 2) - 1):
            result += 2 * np.real(fourier_coefficients[k]) * np.cos((2 * np.pi * k * (x - np.pi)) / T)
            result += (-2) * np.imag(fourier_coefficients[k]) * np.sin((2 * np.pi * k * (x - np.pi)) / T)
        result += np.real(fourier_coefficients[len(fourier_coefficients) // 2]) * np.cos((np.pi * (x - np.pi)) / T)
    else:
        for k in range(1, (len(fourier_coefficients) - 1) // 2):
            result += 2 * np.real(fourier_coefficients[k]) * np.cos((2 * np.pi * k * (x - np.pi)) / T)
            result += (-2) * np.imag(fourier_coefficients[k]) * np.sin((2 * np.pi * k * (x - np.pi)) / T)
    return result


def directivity(data, points):
    """
    Calculates the directivity of the data array with assumption that data has been normalized
    Parameters:
        data:
    :param points:
    :return:
    """

    def intensity_function(phi):
        return fourier_series(phi, 2 * np.pi, fourier_coefficients(interpolate_csv(data, n=points)))

    U_max = 1  # Normalization ensures max intensity is 1
    # Integrate combined function
    U_avg = 0
    for i in range(36):
        U_part, _ = quad(intensity_function, -np.pi + (i * np.pi / 18), -np.pi + ((i + 1) * np.pi / 18))
        U_avg += U_part
    U_avg /= (2 * np.pi)  # Normalize by 2π
    directivity = U_max / U_avg
    return directivity


# Load data from CSV files
data_dir = Path('csv_canonical')
csv_files = list(data_dir.glob('*.csv'))
print(f"Found {len(csv_files)} CSV files to process")
# Calculates directivity for all CSV files
for csv_file in csv_files:
    data = np.genfromtxt(csv_file, delimiter=',', names=True)
    directivity_data = directivity(data, 1000)
    print(f"Directivity {csv_file}: {directivity_data}")
    directivity_data_dBi = 10 * np.log10(directivity_data)
    print(f"Directivity {csv_file} dBi: {directivity_data_dBi}")
