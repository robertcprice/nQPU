"""Comprehensive tests for the nqpu.bio package.

Tests cover: Photosynthesis (FMOComplex, SpectralDensity, DecoherenceModel,
QuantumTransportEfficiency), Enzyme Tunneling (EnzymeTunneling,
TunnelingBarrier, TunnelingSensitivity, ENZYMES), Olfaction (QuantumNose,
OlfactoryReceptor, MolecularVibration, Odorant, OdorDiscrimination, ODORANTS),
Avian Navigation (RadicalPair, CryptochromeModel, CompassSensitivity,
DecoherenceEffects), and DNA Mutation (BasePair, BasePairType,
DoubleWellPotential, TautomerTunneling, MutationRate).
"""

import math

import numpy as np
import pytest

from nqpu.bio import (
    # Photosynthesis
    FMOComplex,
    FMOEvolution,
    PhotosyntheticSystem,
    QuantumTransportEfficiency,
    DecoherenceModel,
    SpectralDensity,
    SpectralDensityType,
    # Enzyme tunneling
    EnzymeTunneling,
    TunnelingBarrier,
    BarrierShape,
    TunnelingSensitivity,
    ENZYMES,
    # Olfaction
    QuantumNose,
    OlfactoryReceptor,
    MolecularVibration,
    Odorant,
    OdorDiscrimination,
    ODORANTS,
    # Avian navigation
    RadicalPair,
    CryptochromeModel,
    CompassSensitivity,
    DecoherenceEffects,
    # DNA mutation
    BasePair,
    BasePairType,
    DoubleWellPotential,
    TautomerTunneling,
    MutationRate,
)


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def fmo():
    return FMOComplex.standard()


@pytest.fixture
def adh():
    return ENZYMES["alcohol_dehydrogenase"]


@pytest.fixture
def nose():
    return QuantumNose.default()


@pytest.fixture
def rp():
    return RadicalPair.cryptochrome()


@pytest.fixture
def at_pair():
    return BasePair.from_type(BasePairType.AT)


@pytest.fixture
def gc_pair():
    return BasePair.from_type(BasePairType.GC)


# ====================================================================
# Photosynthesis tests
# ====================================================================


class TestSpectralDensity:
    def test_drude_lorentz_positive_at_positive_freq(self):
        sd = SpectralDensity()
        val = sd.evaluate(100.0)
        assert val > 0

    def test_zero_at_nonpositive_frequency(self):
        sd = SpectralDensity()
        assert sd.evaluate(0.0) == 0.0
        assert sd.evaluate(-10.0) == 0.0

    @pytest.mark.parametrize("sd_type", list(SpectralDensityType))
    def test_all_density_types_nonnegative(self, sd_type):
        sd = SpectralDensity(density_type=sd_type)
        val = sd.evaluate(200.0)
        assert val >= 0

    def test_dephasing_rate_positive_at_room_temp(self):
        sd = SpectralDensity()
        rate = sd.dephasing_rate_at_temperature(300.0)
        assert rate > 0

    def test_dephasing_rate_raises_on_nonpositive_temp(self):
        sd = SpectralDensity()
        with pytest.raises(ValueError, match="positive"):
            sd.dephasing_rate_at_temperature(0.0)


class TestDecoherenceModel:
    def test_dephasing_rate_positive(self):
        dm = DecoherenceModel(temperature_k=300.0)
        assert dm.dephasing_rate_cm_inv() > 0

    def test_relaxation_rate_positive_for_positive_gap(self):
        dm = DecoherenceModel(temperature_k=300.0)
        rate = dm.relaxation_rate_cm_inv(100.0)
        assert rate > 0

    def test_relaxation_rate_zero_for_nonpositive_gap(self):
        dm = DecoherenceModel(temperature_k=300.0)
        assert dm.relaxation_rate_cm_inv(0.0) == 0.0
        assert dm.relaxation_rate_cm_inv(-50.0) == 0.0


class TestFMOComplex:
    def test_standard_has_seven_sites(self, fmo):
        assert fmo.sites == 7

    def test_hamiltonian_hermitian(self, fmo):
        h = fmo.hamiltonian
        assert np.allclose(h, h.T)

    def test_evolve_returns_fmo_evolution(self, fmo):
        result = fmo.evolve(duration_fs=200.0, steps=100)
        assert isinstance(result, FMOEvolution)

    def test_evolution_times_shape(self, fmo):
        steps = 100
        result = fmo.evolve(duration_fs=200.0, steps=steps)
        assert result.times_fs.shape == (steps + 1,)
        assert result.times_fs[0] == pytest.approx(0.0)

    def test_site_populations_shape_and_initial(self, fmo):
        result = fmo.evolve(duration_fs=200.0, steps=100)
        assert result.site_populations.shape == (101, 7)
        # Initial site 0 should have population 1
        assert result.site_populations[0, 0] == pytest.approx(1.0)

    def test_total_population_initially_one(self, fmo):
        result = fmo.evolve(duration_fs=200.0, steps=100)
        assert result.total_population(0) == pytest.approx(1.0)

    def test_peak_efficiency_nonnegative(self, fmo):
        result = fmo.evolve(duration_fs=200.0, steps=100)
        assert result.peak_efficiency() >= 0.0

    def test_average_coherence_nonnegative(self, fmo):
        result = fmo.evolve(duration_fs=200.0, steps=100)
        assert result.average_coherence(50) >= 0.0

    def test_coherence_lifetime_nonnegative(self, fmo):
        result = fmo.evolve(duration_fs=500.0, steps=200)
        lifetime = result.coherence_lifetime_fs()
        assert lifetime >= 0.0

    def test_evolve_raises_on_nonpositive_duration(self, fmo):
        with pytest.raises(ValueError, match="positive"):
            fmo.evolve(duration_fs=0.0, steps=100)

    def test_evolve_raises_on_nonpositive_steps(self, fmo):
        with pytest.raises(ValueError, match="positive"):
            fmo.evolve(duration_fs=100.0, steps=0)

    @pytest.mark.parametrize("system", list(PhotosyntheticSystem))
    def test_from_system_creates_valid_complex(self, system):
        c = FMOComplex.from_system(system)
        assert c.sites > 0
        assert c.hamiltonian.shape == (c.sites, c.sites)


class TestQuantumTransportEfficiency:
    def test_quantum_efficiency_positive(self, fmo):
        qte = QuantumTransportEfficiency(fmo.hamiltonian)
        eff = qte.quantum_efficiency(duration_fs=200.0, steps=100)
        assert eff >= 0.0

    def test_classical_efficiency_positive(self, fmo):
        qte = QuantumTransportEfficiency(fmo.hamiltonian)
        eff = qte.classical_efficiency(duration_fs=200.0, steps=100)
        assert eff >= 0.0


# ====================================================================
# Enzyme tunneling tests
# ====================================================================


class TestTunnelingBarrier:
    def test_validate_passes_for_valid_barrier(self):
        tb = TunnelingBarrier(height_ev=0.3, width_nm=0.05)
        tb.validate()  # Should not raise

    def test_validate_raises_on_negative_height(self):
        tb = TunnelingBarrier(height_ev=-0.1, width_nm=0.05)
        with pytest.raises(ValueError, match="height"):
            tb.validate()

    def test_validate_raises_on_negative_width(self):
        tb = TunnelingBarrier(height_ev=0.3, width_nm=-0.01)
        with pytest.raises(ValueError, match="width"):
            tb.validate()


class TestEnzymeTunneling:
    def test_from_barrier_creates_model(self):
        et = EnzymeTunneling.from_barrier(0.3, 0.05)
        assert et.barrier.height_ev == pytest.approx(0.3)
        assert et.barrier.width_nm == pytest.approx(0.05)

    def test_tunneling_probability_between_zero_and_one(self, adh):
        prob = adh.tunneling_probability()
        assert 0.0 <= prob <= 1.0

    def test_tunneling_rate_positive(self, adh):
        rate = adh.tunneling_rate()
        assert rate > 0

    def test_classical_rate_positive_at_room_temp(self, adh):
        rate = adh.classical_rate(300.0)
        assert rate >= 0

    def test_classical_rate_raises_on_nonpositive_temp(self, adh):
        with pytest.raises(ValueError, match="positive"):
            adh.classical_rate(0.0)

    def test_total_rate_exceeds_individual_rates(self, adh):
        total = adh.total_rate(300.0)
        tunnel = adh.tunneling_rate()
        assert total >= tunnel

    def test_kie_ratio_greater_than_one(self, adh):
        kie = adh.kie_ratio()
        # Proton tunnels faster than deuterium
        assert kie > 1.0

    def test_swain_schaad_exponent_finite(self, adh):
        ss = adh.swain_schaad_exponent()
        assert math.isfinite(ss)
        # Tunneling gives exponent > classical limit of 1.44
        assert ss > 1.0

    def test_tunnel_splitting_positive(self, adh):
        delta = adh.tunnel_splitting_ev()
        assert delta > 0

    def test_crossover_temperature_positive(self, adh):
        t_cross = adh.crossover_temperature()
        assert t_cross > 0

    def test_zero_barrier_gives_unit_probability(self):
        et = EnzymeTunneling.from_barrier(0.0, 0.05)
        assert et.tunneling_probability() == pytest.approx(1.0)

    @pytest.mark.parametrize("shape", list(BarrierShape))
    def test_all_barrier_shapes_give_valid_probability(self, shape):
        et = EnzymeTunneling(
            barrier=TunnelingBarrier(height_ev=0.3, width_nm=0.05, shape=shape)
        )
        prob = et.tunneling_probability()
        assert 0.0 <= prob <= 1.0

    def test_enzymes_dict_has_three_entries(self):
        assert len(ENZYMES) == 3
        assert "alcohol_dehydrogenase" in ENZYMES
        assert "soybean_lipoxygenase" in ENZYMES
        assert "aromatic_amine_dehydrogenase" in ENZYMES

    def test_rate_vs_temperature_shapes(self, adh):
        temps, total, classical, tunnel = adh.rate_vs_temperature(
            t_min=100.0, t_max=400.0, steps=10
        )
        assert temps.shape == (10,)
        assert total.shape == (10,)
        assert classical.shape == (10,)
        assert tunnel.shape == (10,)


class TestTunnelingSensitivity:
    def test_height_sensitivity_shape(self, adh):
        ts = TunnelingSensitivity(adh)
        heights, probs = ts.height_sensitivity(n_points=5)
        assert heights.shape == (5,)
        assert probs.shape == (5,)

    def test_width_sensitivity_shape(self, adh):
        ts = TunnelingSensitivity(adh)
        widths, probs = ts.width_sensitivity(n_points=5)
        assert widths.shape == (5,)
        assert probs.shape == (5,)

    def test_mass_sensitivity_shape(self, adh):
        ts = TunnelingSensitivity(adh)
        masses, probs = ts.mass_sensitivity()
        assert masses.shape == (3,)
        assert probs.shape == (3,)
        # Lighter particle tunnels more
        assert probs[0] >= probs[1] >= probs[2]


# ====================================================================
# Olfaction tests
# ====================================================================


class TestMolecularVibration:
    def test_energy_ev_positive(self):
        mv = MolecularVibration(frequency_cm_inv=1680.0)
        assert mv.energy_ev() > 0

    def test_thermal_occupation_nonnegative(self):
        mv = MolecularVibration(frequency_cm_inv=1680.0)
        n = mv.thermal_occupation(300.0)
        assert n >= 0

    def test_deuterated_lowers_frequency(self):
        mv = MolecularVibration(frequency_cm_inv=1680.0, reduced_mass_amu=1.0)
        d_mv = mv.deuterated()
        assert d_mv.frequency_cm_inv < mv.frequency_cm_inv


class TestOlfactoryReceptor:
    def test_energy_gap_positive(self):
        receptor = OlfactoryReceptor()
        assert receptor.energy_gap_ev() > 0

    def test_resonant_frequency_positive(self):
        receptor = OlfactoryReceptor()
        assert receptor.resonant_frequency_cm_inv() > 0


class TestQuantumNose:
    def test_default_creates_valid_nose(self, nose):
        assert nose.energy_gap_ev() > 0
        assert nose.resonant_frequency_cm_inv() > 0

    def test_tunneling_rate_positive_at_resonance(self, nose):
        freq = nose.resonant_frequency_cm_inv()
        rate = nose.tunneling_rate(freq)
        assert rate > 0

    def test_tunneling_rate_raises_on_negative_freq(self, nose):
        with pytest.raises(ValueError, match="non-negative"):
            nose.tunneling_rate(-100.0)

    def test_on_resonance_rate_exceeds_off_resonance(self, nose):
        freq_on = nose.resonant_frequency_cm_inv()
        freq_off = freq_on + 1000.0
        assert nose.tunneling_rate(freq_on) > nose.tunneling_rate(freq_off)

    def test_selectivity_greater_than_one(self, nose):
        sel = nose.selectivity(detuning_cm_inv=500.0)
        assert sel > 1.0

    def test_spectrum_shapes(self, nose):
        freqs, rates = nose.spectrum(steps=20)
        assert freqs.shape == (20,)
        assert rates.shape == (20,)

    def test_is_detected_returns_bool(self, nose):
        freq = nose.resonant_frequency_cm_inv()
        result = nose.is_detected(freq)
        assert isinstance(result, bool)


class TestOdorants:
    def test_odorants_dict_has_four_entries(self):
        assert len(ODORANTS) == 4
        assert "acetophenone" in ODORANTS
        assert "benzaldehyde" in ODORANTS
        assert "muscone" in ODORANTS
        assert "hydrogen_sulfide" in ODORANTS

    def test_odorant_deuterated_shifts_frequency(self):
        aceto = ODORANTS["acetophenone"]
        d_aceto = aceto.deuterated()
        assert d_aceto.primary_frequency_cm_inv < aceto.primary_frequency_cm_inv
        assert d_aceto.name.startswith("d-")


class TestOdorDiscrimination:
    def test_isotope_test_returns_dict(self, nose):
        disc = OdorDiscrimination(nose)
        aceto = ODORANTS["acetophenone"]
        result = disc.isotope_test(aceto)
        assert "quantum_discriminates" in result
        assert "classical_discriminates" in result
        assert isinstance(result["quantum_discriminates"], bool)

    def test_classical_cannot_discriminate_isotopologues(self, nose):
        disc = OdorDiscrimination(nose)
        aceto = ODORANTS["acetophenone"]
        d_aceto = aceto.deuterated()
        assert disc.classical_discriminates(aceto, d_aceto) is False


# ====================================================================
# Avian navigation tests
# ====================================================================


class TestRadicalPair:
    def test_cryptochrome_creates_valid_pair(self, rp):
        assert rp.hyperfine_coupling_mhz == pytest.approx(28.0)
        assert rp.field_strength_ut == pytest.approx(50.0)

    def test_singlet_yield_between_zero_and_one(self, rp):
        sy = rp.singlet_yield(0.0)
        assert 0.0 <= sy <= 1.0

    def test_singlet_yield_varies_with_angle(self, rp):
        y0 = rp.singlet_yield(0.0)
        y90 = rp.singlet_yield(math.pi / 2)
        # Anisotropic hyperfine should produce different yields
        assert y0 != pytest.approx(y90, abs=1e-4)

    def test_angular_response_shapes(self, rp):
        angles, yields = rp.angular_response(n_angles=6)
        assert angles.shape == (6,)
        assert yields.shape == (6,)

    def test_compass_anisotropy_positive(self, rp):
        aniso = rp.compass_anisotropy(n_angles=6)
        assert aniso > 0


class TestCryptochromeModel:
    def test_european_robin_has_three_trp(self):
        robin = CryptochromeModel.european_robin()
        assert len(robin.trp_hyperfine_mhz) == 3
        assert robin.species == "European robin"

    def test_homing_pigeon_has_three_trp(self):
        pigeon = CryptochromeModel.homing_pigeon()
        assert len(pigeon.trp_hyperfine_mhz) == 3
        assert pigeon.species == "homing pigeon"

    def test_effective_radical_pair_returns_radical_pair(self):
        robin = CryptochromeModel.european_robin()
        rp = robin.effective_radical_pair(0)
        assert isinstance(rp, RadicalPair)
        assert rp.hyperfine_coupling_mhz == pytest.approx(30.0)

    def test_effective_radical_pair_raises_on_bad_index(self):
        robin = CryptochromeModel.european_robin()
        with pytest.raises(IndexError):
            robin.effective_radical_pair(5)

    def test_total_compass_anisotropy_positive(self):
        robin = CryptochromeModel.european_robin()
        aniso = robin.total_compass_anisotropy(n_angles=6)
        assert aniso > 0


class TestCompassSensitivity:
    def test_angular_resolution_positive(self, rp):
        cs = CompassSensitivity(rp)
        res = cs.angular_resolution(n_angles=6)
        assert res > 0

    def test_field_strength_sensitivity_shapes(self, rp):
        cs = CompassSensitivity(rp)
        fields, aniso = cs.field_strength_sensitivity(n_points=3)
        assert fields.shape == (3,)
        assert aniso.shape == (3,)


class TestDecoherenceEffectsAvian:
    def test_anisotropy_vs_lifetime_shapes(self, rp):
        de = DecoherenceEffects(rp)
        lifetimes, aniso = de.anisotropy_vs_lifetime(n_points=3)
        assert lifetimes.shape == (3,)
        assert aniso.shape == (3,)


# ====================================================================
# DNA mutation tests
# ====================================================================


class TestDoubleWellPotential:
    def test_evaluate_at_origin_gives_barrier_height(self):
        dwp = DoubleWellPotential(
            barrier_height_ev=0.4, barrier_width_nm=0.07, asymmetry_ev=0.0
        )
        # For symmetric potential, V(0) should equal barrier height
        assert dwp.evaluate(0.0) == pytest.approx(0.4)

    def test_potential_curve_shape(self):
        dwp = DoubleWellPotential(
            barrier_height_ev=0.4, barrier_width_nm=0.07, asymmetry_ev=0.05
        )
        x, v = dwp.potential_curve(n_points=50)
        assert x.shape == (50,)
        assert v.shape == (50,)


class TestBasePair:
    def test_at_pair_has_two_bonds(self, at_pair):
        assert at_pair.n_bonds == 2
        assert at_pair.pair_type == BasePairType.AT

    def test_gc_pair_has_three_bonds(self, gc_pair):
        assert gc_pair.n_bonds == 3
        assert gc_pair.pair_type == BasePairType.GC

    def test_tautomer_probability_between_zero_and_one(self, at_pair):
        prob = at_pair.tautomer_probability(310.0)
        assert 0.0 <= prob <= 1.0

    def test_tautomer_probability_raises_on_nonpositive_temp(self, at_pair):
        with pytest.raises(ValueError, match="positive"):
            at_pair.tautomer_probability(0.0)

    def test_concerted_probability_less_than_single(self, at_pair):
        p_single = at_pair.tautomer_probability(310.0)
        p_concerted = at_pair.concerted_tautomer_probability(310.0)
        # concerted = single^n_bonds, so it should be <= single
        assert p_concerted <= p_single

    def test_mutation_rate_positive(self, at_pair):
        rate = at_pair.mutation_rate(310.0)
        assert rate >= 0

    def test_classical_mutation_rate_positive(self, at_pair):
        rate = at_pair.classical_mutation_rate(310.0)
        assert rate >= 0

    def test_quantum_classical_ratio_positive(self, at_pair):
        ratio = at_pair.quantum_classical_ratio(310.0)
        assert ratio > 0

    def test_tunnel_splitting_positive(self, at_pair):
        delta = at_pair.tunnel_splitting_ev()
        assert delta > 0

    @pytest.mark.parametrize("bp_type", list(BasePairType))
    def test_from_type_creates_valid_pair(self, bp_type):
        bp = BasePair.from_type(bp_type)
        assert bp.pair_type == bp_type
        assert bp.potential.barrier_height_ev > 0


class TestTautomerTunneling:
    def test_wkb_action_positive(self, at_pair):
        tt = TautomerTunneling(at_pair)
        action = tt.wkb_action()
        assert action > 0

    def test_instanton_rate_positive(self, at_pair):
        tt = TautomerTunneling(at_pair)
        rate = tt.instanton_rate(310.0)
        assert rate >= 0

    def test_tunneling_time_positive(self, at_pair):
        tt = TautomerTunneling(at_pair)
        t_fs = tt.tunneling_time_fs()
        assert t_fs > 0


class TestMutationRate:
    def test_compare_rates_returns_dict(self, at_pair):
        mr = MutationRate(at_pair)
        result = mr.compare_rates(310.0)
        assert "quantum_rate" in result
        assert "classical_rate" in result
        assert "ratio" in result
        assert "base_pair" in result

    def test_mutation_rate_per_cell_division_nonnegative(self, at_pair):
        mr = MutationRate(at_pair)
        count = mr.mutation_rate_per_cell_division(310.0)
        assert count >= 0

    def test_quantum_dominance_temperature_positive(self, at_pair):
        mr = MutationRate(at_pair)
        t_dom = mr.quantum_dominance_temperature()
        assert t_dom > 0
