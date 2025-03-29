import numpy as np
import time
from ce_perm_c.ceviche import fdfd_ez, my_fdfd_ez


def timing_fdfd_ez_vs_my_fdfd_ez():
    """Test that my_fdfd_ez produces the same results as fdfd_ez"""
    # Setup parameters
    omega = 2 * np.pi * 200e12  # 200 THz / 1.5 um
    dL = 5e-8  # 50 nanometers
    Nx, Ny = 400, 400  # grid size
    eps_r = np.random.random((Nx, Ny))
    source = np.zeros((Nx, Ny))
    source[Nx // 2, Ny // 2] = 1
    npml = [20, 20]

    # Solve using fdfd_ez
    F_original = fdfd_ez(omega, dL, eps_r, npml)

    # Solve using my_fdfd_ez
    F_my = my_fdfd_ez(omega, dL, eps_r, npml)
    # warm up
    F_my.solve(source)

    # timing:
    times = 5
    start = time.time()
    for _ in range(times):
        F_original.solve(source)
    end = time.time()
    time_fdfd_ez = end - start
    print(f"Time taken by fdfd_ez: {time_fdfd_ez} seconds")

    # timing my_fdfd_ez
    start = time.time()
    for _ in range(times):
        F_my.solve(source)
    end = time.time()
    time_my_fdfd_ez = end - start
    print(f"Time taken by my_fdfd_ez: {time_my_fdfd_ez} seconds")

    # compare
    print(f"Speedup: {time_fdfd_ez / time_my_fdfd_ez}")


if __name__ == "__main__":
    timing_fdfd_ez_vs_my_fdfd_ez()
