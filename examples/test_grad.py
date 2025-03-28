import numpy as np
import autograd.numpy as npa

from ce_perm_c.ceviche import fdfd_ez, my_fdfd_ez, jacobians


def test_grad():
    """Test that the gradient is correct"""
    # Setup parameters
    omega = 2 * np.pi * 200e12  # 200 THz / 1.5 um
    dL = 5e-8  # 50 nanometers
    Nx, Ny = 80, 80  # grid size
    eps_r = np.random.random((Nx, Ny))
    source = np.zeros((Nx, Ny))
    source[Nx // 2, Ny // 2] = 1
    npml = [20, 20]

    # Solve using fdfd_ez
    F_original = fdfd_ez(omega, dL, eps_r, npml)
    # _, _, _ = F_original.solve(source)

    # Solve using my_fdfd_ez
    # Simply use my_fdfd_ez to create the FDFD object instead of fdfd_ez
    F_my = my_fdfd_ez(omega, dL, eps_r, npml)
    # Importantly, we need to call `F_my.solve(source)` to create the perm_c
    _, _, _ = F_my.solve(source)

    def objective_original(eps_r):
        F_original.eps_r = eps_r
        Hx, Hy, Ez = F_original.solve(source)
        return (
            npa.square(npa.abs(Hx)) + npa.square(npa.abs(Hy)) + npa.square(npa.abs(Ez))
        ).sum()

    def objective_my(eps_r):
        F_my.eps_r = eps_r
        Hx, Hy, Ez = F_my.solve(source)
        return (
            npa.square(npa.abs(Hx)) + npa.square(npa.abs(Hy)) + npa.square(npa.abs(Ez))
        ).sum()

    # Prepare gradient functions
    grad_original = jacobians.jacobian(objective_original, mode="reverse")
    grad_my = jacobians.jacobian(objective_my, mode="reverse")

    for _ in range(10):
        eps_r = np.random.random((Nx, Ny))
        # Compare gradients
        assert np.allclose(
            grad_original(eps_r), grad_my(eps_r), rtol=1e-10, atol=1e-10
        ), "Gradients don't match"


if __name__ == "__main__":
    test_grad()
