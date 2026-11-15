"""AME values checked by hand against data/mass_1.mas20.txt (mass excesses in keV copied from the file)."""
import numpy as np
import pandas as pd
import pytest

from xsml import features

N, H, HE4 = 8071.31806, 7288.971064, 2424.91587
O16, O17 = -4737.00217, -808.76421
S34, S35, CL35, P32 = -29931.689, -28846.210, -29013.532, -24304.876
AU197, AU198 = -31139.751, -29580.793


@pytest.fixture(scope="module")
def ame():
    return features.ame_table().set_index(["Z", "A"])


def test_table_shape_and_identity(ame):
    assert len(ame) > 3000 and ame.index.is_unique
    assert (ame.N + ame.index.get_level_values("Z") == ame.index.get_level_values("A")).all()
    assert ame.loc[(17, 35), "element"] == "Cl"


def test_mass_excess_and_binding(ame):
    assert ame.loc[(17, 35), "mass_excess"] == pytest.approx(CL35 / 1e3, abs=1e-9)
    assert ame.loc[(17, 35), "B_per_A"] == pytest.approx(8.5202790, abs=2e-6)   # BINDING ENERGY/A column of the file
    assert ame.loc[(8, 16), "B_per_A"] == pytest.approx(7.9762072, abs=2e-6)
    assert not ame.loc[(17, 35), "estimated"]
    assert ame.loc[(17, 35), "atomic_mass_u"] == pytest.approx(34.968852694, abs=1e-10)       # ATOMIC MASS column of the file
    assert ame.loc[(17, 35), "Q_beta_minus"] == pytest.approx(-5.9662429, abs=2e-6)           # BETA-DECAY ENERGY column of the file


def test_separation_energies(ame):
    assert ame.loc[(8, 17), "S_n"] == pytest.approx((O16 + N - O17) / 1e3, abs=1e-9)
    assert ame.loc[(8, 17), "S_n"] == pytest.approx(4.143080, abs=1e-6)
    assert ame.loc[(17, 35), "S_p"] == pytest.approx((S34 + H - CL35) / 1e3, abs=1e-9)
    assert ame.loc[(17, 35), "S_p"] == pytest.approx(6.370814, abs=1e-6)


def test_q_values(ame):
    assert ame.loc[(17, 35), "Q_np"] == pytest.approx((CL35 + N - S35 - H) / 1e3, abs=1e-9)
    assert ame.loc[(17, 35), "Q_np"] == pytest.approx(0.615025, abs=1e-6)
    assert ame.loc[(17, 35), "Q_na"] == pytest.approx((CL35 + N - P32 - HE4) / 1e3, abs=1e-9)
    assert ame.loc[(79, 197), "Q_ng"] == pytest.approx((AU197 + N - AU198) / 1e3, abs=1e-9)
    assert ame.loc[(79, 197), "Q_ng"] == pytest.approx(ame.loc[(79, 198), "S_n"], abs=1e-9)
    assert features.q_value(17, 35, "n", ["p"]) == pytest.approx(0.615025, abs=1e-6)
    assert np.isnan(features.q_value(17, 20, "n", ["p"]))        # no such nuclide in the table


def test_generated_properties():
    f = features.nuclide_features().set_index(["Z", "A"])
    cl = f.loc[(17, 35)]
    assert (cl.N, cl.N_minus_Z, cl.Z_odd, cl.N_odd, cl.A_odd, cl.pairing) == (18, 1, 1, 0, 1, 0)
    assert (cl.Z_magic_dist, cl.N_magic_dist) == (3, 2)
    assert cl.P_factor == pytest.approx(3 * 2 / 5)
    ca, n14 = f.loc[(20, 40)], f.loc[(7, 14)]
    assert (ca.Z_magic, ca.N_magic, ca.pairing, ca.P_factor) == (1, 1, 1, 0)
    assert n14.pairing == -1
    assert cl.asymmetry == pytest.approx(1 / 35)


def test_append_keeps_rows_and_adds_reaction_q(toy):
    out = features.append(toy)
    assert len(out) == len(toy) and out.index.equals(toy.index)
    cl_np = out[(out.Z == 17) & (out.A == 35) & (out.MT == 103)]
    cl_ng = out[(out.Z == 17) & (out.A == 35) & (out.MT == 102)]
    assert cl_np.Q_reaction.iloc[0] == pytest.approx(0.615025, abs=1e-6)
    assert cl_ng.Q_reaction.iloc[0] == pytest.approx(cl_ng.Q_ng.iloc[0])
    assert np.allclose(out.log10_sigma, np.log10(toy.sigma_b))
    assert out.S_n.notna().all()


def test_reaction_q_other_projectiles_and_unknown_mt():
    df = pd.DataFrame({"projectile": ["p", "n", "n", "n"], "Z": [16, 17, 17, 17], "A": [35, 35, 35, 35], "MT": [4, 1, 2, 4]})
    q = features.reaction_q(df)
    assert q.iloc[0] == pytest.approx(-0.615025, abs=1e-6)       # S-35(p,n)Cl-35 is the inverse of Cl-35(n,p)S-35
    assert np.isnan(q.iloc[1]) and q.iloc[2] == 0 and q.iloc[3] == 0
