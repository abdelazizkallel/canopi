"""Scene metadata must match the RAMI-5 HET09_JBS_SUM specification.

These are constants and table lookups — no stand is built, so they are fast.
"""
import numpy as np

from canopi.scenes.jbs_loader import (
    LX, LY, H, LAI_EFF, JBS_GROUPS, GROUP_TO_SPECIES, ILLUMINATION,
    get_band_params,
)
from canopi.scenes.jbs_species_loader import (
    SPECIES_LIST, N_SPECIES, GROUP_LIST, N_GROUPS,
)


def test_stand_dimensions():
    assert np.isclose(LX, 105.5115)
    assert np.isclose(LY, 106.1535)
    assert np.isclose(H, 30.513)
    assert np.isclose(LAI_EFF, 3.442)


def test_species_and_group_counts():
    assert N_SPECIES == 7 and len(SPECIES_LIST) == 7
    assert N_GROUPS == 18 and len(GROUP_LIST) == 18
    assert len(JBS_GROUPS) == 18
    assert set(SPECIES_LIST) == {"BEPE", "TICO", "ALGL", "POTR", "PIAB", "FREX", "ACPL"}


def test_every_group_maps_to_a_known_species():
    for grp in GROUP_LIST:
        assert GROUP_TO_SPECIES[grp] in SPECIES_LIST, grp


def test_illumination_geometry():
    assert np.isclose(ILLUMINATION["Jul"]["sun_zenith_deg"], 41.0)
    assert np.isclose(ILLUMINATION["Apr"]["sun_zenith_deg"], 56.0)


def test_band_params_are_physical():
    for band in ("O08", "O17"):
        bp = get_band_params(band)
        assert {"rho", "tau", "rsoil"} <= set(bp)
        assert 0.0 <= bp["rho"] <= 1.0
        assert 0.0 <= bp["tau"] <= 1.0
        assert 0.0 <= bp["rho"] + bp["tau"] <= 1.0


def test_red_is_darker_than_nir():
    assert get_band_params("O08")["rho"] < get_band_params("O17")["rho"]
