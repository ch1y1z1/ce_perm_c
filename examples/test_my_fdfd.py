import numpy as np

from ce_perm_c.ceviche import fdfd_ez, my_fdfd_ez


def test_fdfd_ez_vs_my_fdfd_ez():
    """Test that my_fdfd_ez produces the same results as fdfd_ez"""
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

    # Solve using my_fdfd_ez
    F_my = my_fdfd_ez(omega, dL, eps_r, npml)

    for _ in range(10):
        Hx_original, Hy_original, Ez_original = F_original.solve(source)
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

    # Verify that perm_c was created and reused
    assert F_my.perm_c is not None, "perm_c was not created"


if __name__ == "__main__":
    test_fdfd_ez_vs_my_fdfd_ez()
    print("All tests passed!")
