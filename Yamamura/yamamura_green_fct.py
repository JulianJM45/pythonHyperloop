# %%
import numpy as np
from scipy.integrate import cumulative_trapezoid

from modules.my_plots import plot

# %%
# velocity
v = 100

# constants
mu_0 = 4 * np.pi * 1e-7
sigma = 10.3e6

L = 20  # Domain length
N = 1000  # Number of points in the domain
l = 5  # magnet length
d = 0.1  # rail width
a = 0.1  # rail thickness
g = 0.02  # air gap

B0 = 1

dx = L / N  # Spatial step size
x = np.arange(-L / 2, L / 2, dx)
# omega = 2*np.pi*fftfreq(N, d=dx)
k = mu_0 * sigma * d * v / g
# k = mu_0 * sigma * d * v
k = 200
print(f"K={k}")

# %%


def main():
    b0 = b0_function()
    # b = b0
    b = b_function(b0)
    plot(x, b)


main()

# %%


def b0_function():
    b0 = np.zeros_like(x)
    b0[int((L / 2 - l / 2) / dx) : int((L / 2 + l / 2) / dx)] = 1
    # b0hat = fft(b0)
    return b0


# %%
def b_function(b0):
    bi = np.zeros_like(x)
    n = 10  # order
    bi = np.sum([Xn(i, b0) for i in range(n)], axis=0)

    return bi + b0


# %%
def Xn(n, b0):
    """
    Compute the nth term in the Yamamura model with numerical stability.

    Parameters:
    n : int
        Mode number
    b0 : ndarray
        Initial magnetic field distribution

    Returns:
    y : ndarray
        Solution for the nth mode, or zeros if numerically unstable
    """
    c_n = c(n)
    lambda_n = lambda_function(n)
    r1, r2, r1_r2 = calculate_rs(lambda_n)

    # Check for numerical stability - if lambda_n is too large, the exponentials will overflow
    domain_length = x[-1] - x[0]
    max_exp_arg = 700  # exp(700) is close to the limit before overflow

    # Estimate maximum exponential arguments that will occur
    max_r1_arg = abs(r1) * domain_length
    max_r2_arg = abs(r2) * domain_length

    # If the exponential arguments are too large, return zeros (this mode contributes negligibly)
    if max_r1_arg > max_exp_arg or max_r2_arg > max_exp_arg:
        print(
            f"Warning: Mode n={n} numerically unstable (lambda_n={lambda_n:.2f}, max_args={max_r1_arg:.1f}, {max_r2_arg:.1f}). Returning zeros."
        )
        return np.zeros_like(b0)

    # Suppress overflow warnings for this calculation
    with np.errstate(over="ignore", invalid="ignore"):
        # Left integral: ∫_{x0}^{x} e^{r1(x - s)} b(s) ds
        exp_arg1 = -r1 * (x - x[0])
        exp_arg1 = np.clip(exp_arg1, -max_exp_arg, max_exp_arg)
        I1 = cumulative_trapezoid(b0 * np.exp(exp_arg1), x, initial=0)

        # Right integral: ∫_{x}^{xN} e^{r2(x - s)} b(s) ds
        exp_arg2 = -r2 * (x[-1] - x[::-1])
        exp_arg2 = np.clip(exp_arg2, -max_exp_arg, max_exp_arg)
        I2 = cumulative_trapezoid((b0[::-1]) * np.exp(exp_arg2), x[::-1], initial=0)
        I2 = I2[::-1]  # flip back

        # Final exponential terms with overflow protection
        final_exp_arg1 = np.clip(r1 * (x - x[0]), -max_exp_arg, max_exp_arg)
        final_exp_arg2 = np.clip(r2 * (x[-1] - x), -max_exp_arg, max_exp_arg)

        exp1 = np.exp(final_exp_arg1)
        exp2 = np.exp(final_exp_arg2)

        # Replace any inf or nan values with 0
        exp1 = np.where(np.isfinite(exp1), exp1, 0)
        exp2 = np.where(np.isfinite(exp2), exp2, 0)
        I1 = np.where(np.isfinite(I1), I1, 0)
        I2 = np.where(np.isfinite(I2), I2, 0)

        y = -b0 - (lambda_n**2 / r1_r2) * (exp1 * I1 - exp2 * I2)

        # Final cleanup of any remaining numerical issues
        y = np.where(np.isfinite(y), y, 0)

    return y


# %%
def Xn_robust(n, b0, tolerance=1e-10):
    """
    A more robust version that checks the contribution magnitude.

    Parameters:
    n : int
        Mode number
    b0 : ndarray
        Initial magnetic field distribution
    tolerance : float
        Minimum relative contribution to consider (default 1e-10)

    Returns:
    y : ndarray
        Solution for the nth mode, or zeros if contribution is below tolerance
    """
    c_n = c(n)

    # If the coefficient is already very small, don't bother computing
    if abs(c_n) < tolerance:
        return np.zeros_like(b0)

    return Xn(n, b0)


# %%
def analyze_mode_contributions(b0, max_modes=50):
    """
    Analyze the relative contributions of different modes to help determine
    how many modes are needed for convergence.

    Parameters:
    b0 : ndarray
        Initial magnetic field distribution
    max_modes : int
        Maximum number of modes to analyze

    Returns:
    dict : Analysis results including coefficients, stability, and recommendations
    """
    coefficients = []
    stable_modes = []
    lambda_values = []

    print("Analyzing mode contributions...")
    print("Mode |   c_n   | lambda_n | Stable?")
    print("-" * 35)

    for n in range(1, max_modes + 1):
        c_n = c(n)
        lambda_n = lambda_function(n)
        r1, r2, _ = calculate_rs(lambda_n)

        # Check stability
        domain_length = x[-1] - x[0]
        max_r1_arg = abs(r1) * domain_length
        max_r2_arg = abs(r2) * domain_length
        is_stable = max_r1_arg < 700 and max_r2_arg < 700

        coefficients.append(abs(c_n))
        lambda_values.append(lambda_n)
        stable_modes.append(is_stable)

        if n <= 20 or not is_stable:  # Print first 20 modes and all unstable modes
            status = "Yes" if is_stable else "No"
            print(f"{n:4d} | {c_n:7.4f} | {lambda_n:8.2f} | {status}")

    # Find recommended number of modes
    coeffs_array = np.array(coefficients)
    stable_array = np.array(stable_modes)

    # Find where coefficients become negligible (< 1% of max)
    cutoff_coeff = coeffs_array[0] * 0.01
    coeff_cutoff_idx = np.where(coeffs_array < cutoff_coeff)[0]
    coeff_cutoff = coeff_cutoff_idx[0] + 1 if len(coeff_cutoff_idx) > 0 else max_modes

    # Find first unstable mode
    unstable_idx = np.where(~stable_array)[0]
    stability_cutoff = unstable_idx[0] + 1 if len(unstable_idx) > 0 else max_modes

    recommended_modes = min(coeff_cutoff, stability_cutoff)

    print("\nRecommendations:")
    print(f"- Coefficient-based cutoff: {coeff_cutoff} modes")
    print(f"- Stability-based cutoff: {stability_cutoff} modes")
    print(f"- Recommended number of modes: {recommended_modes}")

    return {
        "coefficients": coefficients,
        "lambda_values": lambda_values,
        "stable_modes": stable_modes,
        "recommended_modes": recommended_modes,
        "coefficient_cutoff": coeff_cutoff,
        "stability_cutoff": stability_cutoff,
    }


# %%
def compute_series_convergence(b0, max_modes=None, tolerance=1e-8):
    """
    Compute the series solution and monitor convergence.

    Parameters:
    b0 : ndarray
        Initial magnetic field distribution
    max_modes : int, optional
        Maximum number of modes to use (if None, uses analyze_mode_contributions)
    tolerance : float
        Convergence tolerance for relative change between iterations

    Returns:
    tuple : (solution, convergence_info)
    """
    if max_modes is None:
        analysis = analyze_mode_contributions(b0, max_modes=50)
        max_modes = analysis["recommended_modes"]
        print(f"\nUsing {max_modes} modes based on analysis.")

    solution = np.zeros_like(b0)
    convergence_history = []

    print(f"\nComputing series with {max_modes} modes...")

    for n in range(1, max_modes + 1):
        mode_contribution = Xn_robust(n, b0)
        solution += mode_contribution

        # Monitor convergence
        if n > 1:
            relative_change = np.linalg.norm(mode_contribution) / np.linalg.norm(
                solution
            )
            convergence_history.append(relative_change)

            if n % 5 == 0:  # Print every 5th mode
                print(f"Mode {n:3d}: relative contribution = {relative_change:.2e}")

            # Check for convergence
            if relative_change < tolerance and n > 10:
                print(f"Converged at mode {n} (relative change < {tolerance})")
                break

    return solution, {
        "modes_used": n,
        "convergence_history": convergence_history,
        "final_relative_change": convergence_history[-1] if convergence_history else 0,
    }


# %%
def c(n):
    return 4 / (np.pi * (2 * n - 1)) * np.sin((2 * n - 1) * np.pi / 2)


def lambda_function(n):
    return (2 * n - 1) * np.pi / (2 * a)


def calculate_rs(lambda_n):
    r1 = (k + np.sqrt(k**2 + 4 * lambda_n**2)) / 2
    r2 = (k - np.sqrt(k**2 + 4 * lambda_n**2)) / 2
    r1_r2 = r1 - r2
    return r1, r2, r1_r2


# %%

if __name__ == "__main__":
    main()
