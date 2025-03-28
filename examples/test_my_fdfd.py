import numpy as np
import sys

sys.path.append("../ceviche")

from ce_perm_c.ceviche import fdfd_ez, my_fdfd_ez


def test_fdfd_ez_vs_my_fdfd_ez():
    """Test that my_fdfd_ez produces the same results as fdfd_ez"""
    # Setup parameters
    omega = 2 * np.pi * 200e12  # 200 THz / 1.5 um
    dL = 5e-8  # 50 nanometers
    Nx, Ny = 141, 141  # grid size
    eps_r = np.ones((Nx, Ny))
    source = np.zeros((Nx, Ny))
    source[Nx // 2, Ny // 2] = 1
    npml = [20, 20]

    # Solve using fdfd_ez
    F_original = fdfd_ez(omega, dL, eps_r, npml)
    Hx_original, Hy_original, Ez_original = F_original.solve(source)

    # Solve using my_fdfd_ez
    F_my = my_fdfd_ez(omega, dL, eps_r, npml)
    Hx_my, Hy_my, Ez_my = F_my.solve(source)

    # Compare results
    # Using np.allclose with a reasonable tolerance to account for minor numerical differences
    assert np.allclose(Hx_original, Hx_my, rtol=1e-10, atol=1e-10), (
        "Hx fields don't match"
    )
    assert np.allclose(Hy_original, Hy_my, rtol=1e-10, atol=1e-10), (
        "Hy fields don't match"
    )
    assert np.allclose(Ez_original, Ez_my, rtol=1e-10, atol=1e-10), (
        "Ez fields don't match"
    )

    # Test that my_fdfd_ez is reusing the permutation vector
    # Run a second solve and verify it's still correct
    # Change the source slightly
    source2 = np.zeros((Nx, Ny))
    source2[Nx // 2 + 1, Ny // 2] = 1

    # Solve again with both methods
    Hx_original2, Hy_original2, Ez_original2 = F_original.solve(source2)
    Hx_my2, Hy_my2, Ez_my2 = F_my.solve(source2)

    # Verify results still match
    assert np.allclose(Hx_original2, Hx_my2, rtol=1e-10, atol=1e-10), (
        "Hx fields don't match on second solve"
    )
    assert np.allclose(Hy_original2, Hy_my2, rtol=1e-10, atol=1e-10), (
        "Hy fields don't match on second solve"
    )
    assert np.allclose(Ez_original2, Ez_my2, rtol=1e-10, atol=1e-10), (
        "Ez fields don't match on second solve"
    )

    # Verify that perm_c was created and reused
    assert F_my.perm_c is not None, "perm_c was not created"


if __name__ == "__main__":
    test_fdfd_ez_vs_my_fdfd_ez()
    print("All tests passed!")
