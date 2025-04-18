"""
Author: Rong Jin, University of Kentucky
Date: 04-18-2025
Description: 
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def johnson_cook(E, A, B, n, C, m,
                 D1, D2, D3, D4, D5,
                 eps1, T,
                 eps_tol=0.5,
                 eps_p1_0=1e-3,
                 T0=298.0, T_mlt=905.0,
                 nstep=int(1e5)):
    """
    Compute the Johnson–Cook stress–strain curve up to failure (damage D >= 1).
    
    Parameters
    ----------
    E         : Young's modulus [MPa]
    A, B, n   : strain hardening parameters
    C, m      : strain rate and thermal softening parameters
    D1–D5     : damage model parameters
    eps1      : applied plastic strain rate [1/s]
    T         : temperature [K]
    eps_tol   : tolerance for failure strain integration
    eps_p1_0  : reference plastic strain rate [1/s]
    T0        : reference temperature [K]
    T_mlt     : melting temperature [K]
    nstep     : number of integration steps
    
    Returns
    -------
    sig : ndarray
        Array of stress values [MPa]
    eps : ndarray
        Array of corresponding plastic strain values
    """
    # Normalize temperature and strain rate
    T_star    = (T - T0) / (T_mlt - T0)
    eps1_star = eps1 / eps_p1_0
    
    # Compute time step and strain increment
    dt = (eps_tol / eps1) / nstep
    de = eps1 * dt

    # Initialize lists for stress and strain
    sig = []
    eps = []

    # Initialize plastic strain, total strain, stress, and damage accumulator
    ep      = 0.0
    e_total = 0.0
    sigma   = E * de
    D       = 0.0

    # Time-stepping loop for constitutive integration
    for _ in range(nstep):
        # Compute current flow stress with thermal softening
        sigma_flow = (A + B * ep**n) * (1 - T_star**m)
        # Compute plastic strain increment if stress exceeds flow stress
        dep = (eps_p1_0 * np.exp((sigma - sigma_flow) / (C * sigma_flow))
               if sigma > sigma_flow else 0.0)

        # Update plastic and total strain, then recompute stress
        ep      += dep * dt
        e_total += de
        sigma    = E * (e_total - ep)

        # Record stress and strain
        sig.append(sigma)
        eps.append(e_total)

        # Compute failure strain for damage accumulation
        eps_f = ((D1 + D2 * np.exp(D3 / 3))
                 * (1 + D4 * np.log(eps1_star))
                 * (1 + D5 * T_star))
        # Update damage
        D += dep * dt / eps_f
        if D >= 1.0:
            # Stop when damage reaches 1
            break

    # Append final failure point (zero stress)
    eps.append(eps[-1])
    sig.append(0.0)

    return np.array(sig), np.array(eps)


def residuals(params,
              A, uts, fs_mean,
              datasets,
              E, C, m, D4, D5, eps_dot, T,
              failure_strains):
    """
    Compute residual vector for optimization.
    
    params = [B, n, D1, D2, D3]
    Includes:
      1) Stress–strain residuals for each dataset
      2) Failure strain residuals for each experiment
      3) n consistency residual via three-point fit
    """
    B, n, D1, D2, D3 = params

    # 1) Generate model stress–strain curve
    sig_mod, eps_mod = johnson_cook(
        E, A, B, n, C, m,
        D1, D2, D3, D4, D5,
        eps_dot, T
    )

    res = []
    # 2) Append stress residuals: model vs. experimental
    for data in datasets:
        eps_exp, sig_exp = data[:, 0], data[:, 1]
        sig_int = np.interp(eps_exp, eps_mod, sig_mod)
        res.append(sig_int - sig_exp)

    # 3) Append failure strain residuals
    eps_f_model = D1 + D2 * np.exp(D3 / 3)
    for fs in failure_strains:
        res.append(np.atleast_1d(eps_f_model - fs))

    # 4) Append n consistency residual (three-point fit)
    n_calc = np.log((uts - A) / B) / np.log(fs_mean - A / E)
    res.append(np.atleast_1d(n - n_calc))

    # Concatenate into single residual vector
    return np.concatenate(res)


if __name__ == "__main__":
    # 1) Change working directory to script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # 2) Load experimental data files
    data1 = np.loadtxt('data1.txt')
    data2 = np.loadtxt('data2.txt')
    data3 = np.loadtxt('data3.txt')
    datasets = [data1, data2, data3]

    # Extract failure strains (last strain value) and compute mean
    failure_strains = [d[-1, 0] for d in datasets]
    fs_mean = np.mean(failure_strains)

    # 3) Fixed material and loading constants
    ys = 46.40660067       # yield strength [MPa]
    uts = 160.2778867      # ultimate tensile strength [MPa]
    fs_mean = np.mean(failure_strains)  # mean failure strain

    E = 45000.0            # Young's modulus [MPa]
    C = 0.013              # strain‑rate sensitivity coefficient
    m = 1.550              # thermal softening exponent
    D4 = 0.4738            # damage rate sensitivity
    D5 = 7.2               # damage temperature sensitivity
    eps_dot = 1e-3         # applied strain rate [1/s]
    T = 298.15             # temperature [K]

    # 4) Initial guess for [B, n, D1, D2, D3]
    B0 = 400.0             # hardening modulus
    n0 = 0.5374            # hardening exponent
    D10 = -0.196           # damage parameter D1
    D20 = 0.3374           # damage parameter D2
    D30 = -0.4537          # damage parameter D3

    x0 = np.array([B0, n0, D10, D20, D30])


    # 5) Set parameter bounds
    lb = [0.0, 0.0, -1.0, 0.0, -2.0]
    ub = [2000.0, 2.0, 0.0, 2.0, 0.0]

    # 6) Run least-squares optimization
    result = least_squares(
        residuals, x0, bounds=(lb, ub),
        args=(
            ys,           # A
            uts,          # ultimate tensile strength
            fs_mean,      # mean failure strain
            datasets,     # list of experimental data
            E, C, m, D4, D5,
            eps_dot, T,
            failure_strains  
        ),
        xtol=1e-8, ftol=1e-8, verbose=2
    )


    # 7) Extract and print optimized parameters
    B_opt, n_opt, D1_opt, D2_opt, D3_opt = result.x
    eps_f_opt = D1_opt + D2_opt * np.exp(D3_opt / 3)
    print("Fixed A =", ys)
    print(f"Optimized B = {B_opt:.4f}, n = {n_opt:.4f}")
    print(f"Optimized D1 = {D1_opt:.4f}, D2 = {D2_opt:.4f}, D3 = {D3_opt:.4f}")
    print(f"Model failure strain = {eps_f_opt:.4f}")
    print("Experimental failure strains:", failure_strains)

    # 8) Plot comparison of experiments vs. fitted JC curve
    sig_fit, eps_fit = johnson_cook(
        E, ys, B_opt, n_opt,
        C, m, D1_opt, D2_opt, D3_opt, D4, D5,
        eps_dot, T
    )

    plt.figure()
    exp_colors = ['#2b83ba','#d7191c','#fdae61']  
    for (data, lbl, col) in zip(datasets, ['Exp1','Exp2','Exp3'], exp_colors):
        plt.plot(
            data[:,0], data[:,1],
            marker='o', linestyle='-',
            color=col,
            ms=3, lw=1.5,
            label=lbl
        )

    # JC fit curve in a contrasting color
    plt.plot(
        eps_fit, sig_fit,
        linestyle='--',
        color='k',   
        lw=3,
        label='JC Fit'
    )

    plt.xlabel('Strain')
    plt.ylabel('Stress [MPa]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
