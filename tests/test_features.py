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
    df = pd.DataFrame({"projectile": ["p", "n", "n", "n"], "Z": [16, 17, 17, 17], "A": [35, 35, 35, 35], "MT": [4, 1, 2, 4]})   # no reaction column: MT route
    q = features.reaction_q(df)
    assert q.iloc[0] == pytest.approx(-0.615025, abs=1e-6)       # S-35(p,n)Cl-35 is the inverse of Cl-35(n,p)S-35
    assert np.isnan(q.iloc[1]) and q.iloc[2] == 0 and q.iloc[3] == 0


# mass excesses (keV) copied from data/mass_1.mas20.txt for the charged-particle checks
D2 = 13135.722895
NI63, NI64, NI65, CU64, CU65, CO61, ZN67 = -65512.891, -67099.034, -65125.796, -65424.418, -67263.677, -62898.179, -67880.376
ZN68, GA68, MO100, TC99 = -70007.154, -67086.054, -86193.029, -87327.869


def _frame(rows):
    return pd.DataFrame(rows, columns=["projectile", "Z", "A", "MT", "reaction"])


def test_reaction_q_charged_particles_and_photons_by_hand():
    df = _frame([
        ("p", 28, 64, 4, "28-NI-64(P,N)29-CU-64,,SIG"),
        ("p", 30, 68, 4, "30-ZN-68(P,N)31-GA-68,,SIG"),
        ("p", 42, 100, 16, "42-MO-100(P,2N)43-TC-99,,SIG"),
        ("p", 28, 64, 102, "28-NI-64(P,G)29-CU-65,,SIG"),
        ("p", 28, 64, 107, "28-NI-64(P,A)27-CO-61,,SIG"),
        ("d", 28, 64, 103, "28-NI-64(D,P)28-NI-65,,SIG"),
        ("a", 28, 64, 4, "28-NI-64(A,N)30-ZN-67,,SIG"),
        ("g", 28, 64, 4, "28-NI-64(G,N)28-NI-63,,SIG"),
    ])
    by_hand = [(NI64 + H - CU64 - N), (ZN68 + H - GA68 - N), (MO100 + H - TC99 - 2 * N), (NI64 + H - CU65), (NI64 + H - CO61 - HE4),
               (NI64 + D2 - NI65 - H), (NI64 + HE4 - ZN67 - N), (NI64 - NI63 - N)]
    rounded = [-2.456963, -3.703447, -7.718825, 7.453614, 0.663200, 3.873514, -4.865060, -9.657461]
    from_string = features.reaction_q(df)
    from_mt = features.reaction_q(df.drop(columns="reaction"))
    for i in range(len(df)):
        assert from_string.iloc[i] == pytest.approx(by_hand[i] / 1e3, abs=1e-9)
        assert from_string.iloc[i] == pytest.approx(rounded[i], abs=1e-6)
        assert from_mt.iloc[i] == pytest.approx(from_string.iloc[i], abs=1e-12)


def test_reaction_q_is_nan_where_the_outgoing_particles_are_not_fixed():
    df = _frame([
        ("n", 17, 35, 1, "17-CL-35(N,TOT),,SIG"),
        ("p", 28, 64, 9000, "28-NI-64(P,X)27-CO-58,,SIG"),
        ("n", 92, 235, 18, "92-U-235(N,F),,SIG"),
        ("p", 28, 64, 3, "28-NI-64(P,NON),,SIG"),
        ("n", 28, 64, 101, "28-NI-64(N,ABS),,SIG"),
        ("n", 28, 64, 9000, "28-NI-64(N,SCT),,SIG"),
    ])
    assert features.reaction_q(df).isna().all()
    assert features.reaction_q(df.drop(columns="reaction")).isna().all()


def test_reaction_string_overrides_a_mismatched_mt():
    # (D,INL) is filed under MT 103 in a few datasets: the string gives 0, the MT label alone would give the (d,p) value
    df = _frame([("d", 28, 64, 103, "28-NI-64(D,INL)28-NI-64,,SIG"), ("p", 28, 64, 103, "28-NI-64(P,INL)28-NI-64,,SIG"), ("n", 28, 64, 2, "28-NI-64(N,EL)28-NI-64,,SIG")])
    q = features.reaction_q(df)
    assert q.tolist() == [0.0, 0.0, 0.0]
    assert features.reaction_q(df.drop(columns="reaction")).iloc[0] == pytest.approx(3.873514, abs=1e-6)


def test_ejectiles_from_reaction():
    f = features.ejectiles_from_reaction
    assert f("17-CL-35(N,P)16-S-35,,SIG") == ["p"] and f("42-MO-100(P,2N)43-TC-99-M,,SIG") == ["n", "n"]
    assert f("1-H-2(N,G)1-H-3,,SIG") == [] and f("3-LI-7(P,N+A)2-HE-3,,SIG") == ["n", "a"]
    assert f("2-HE-4(HE3,G)4-BE-7,,SIG") == [] and f("3-LI-6(D,HE3)2-HE-5,,SIG") == ["he3"]
    assert f("26-FE-56(N,INL)26-FE-56,,SIG") == ["n"]
    assert f("92-U-238(N,F),,SIG") is None and f("82-PB-208(P,X)0-NN-1,,SIG") is None and f("toy") is None
