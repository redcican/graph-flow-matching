"""Tests for geometric operations on the product manifold (Section 3, Eqs 5-12)."""

import math

import pytest
import torch

from graph_flow_matching.models.manifold_ops import (
    EPS,
    SLERP_THRESHOLD,
    angle_to_ordinal,
    circular_distance,
    clamp_probabilities,
    conditional_velocity_categorical,
    conditional_velocity_continuous,
    conditional_velocity_ordinal,
    fisher_rao_distance,
    geodesic_interpolation_categorical,
    geodesic_interpolation_continuous,
    geodesic_interpolation_ordinal,
    ordinal_to_angle,
    pairwise_product_distance_matrix,
    pairwise_sample_distance_l1,
    product_manifold_distance,
    product_manifold_distance_squared,
    sphere_map,
    sphere_map_inverse,
)

B = 8


# ===================================================================
# Probability simplex utilities
# ===================================================================

class TestClampProbabilities:
    def test_output_sums_to_one(self):
        p = torch.tensor([[0.0, 0.5, 0.5], [0.3, 0.3, 0.4]])
        out = clamp_probabilities(p)
        assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_no_zeros(self):
        p = torch.tensor([[0.0, 0.0, 1.0]])
        out = clamp_probabilities(p)
        assert (out > 0).all()

    def test_already_valid(self):
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        out = clamp_probabilities(p)
        assert torch.allclose(out, p, atol=1e-6)


class TestSphereMap:
    def test_eq5_formula(self):
        """φ(p) = 2√p (Eq 5)."""
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        s = sphere_map(p)
        expected = 2.0 * torch.sqrt(p)
        assert torch.allclose(s, expected, atol=1e-6)

    def test_roundtrip(self):
        p = torch.softmax(torch.randn(B, 5), dim=-1)
        recovered = sphere_map_inverse(sphere_map(p))
        assert torch.allclose(recovered, p, atol=1e-5)

    def test_output_on_sphere(self):
        """Sphere-mapped points should lie on the unit sphere (||φ||=2)."""
        p = torch.softmax(torch.randn(B, 4), dim=-1)
        s = sphere_map(p)
        norms = s.norm(dim=-1)
        assert torch.allclose(norms, 2.0 * torch.ones(B), atol=1e-5)


# ===================================================================
# Distance functions
# ===================================================================

class TestFisherRaoDistance:
    def test_eq6_formula(self):
        """d_FR(p, q) = 2·arccos(Σ√(p_k·q_k)) (Eq 6)."""
        p = torch.tensor([[0.5, 0.3, 0.2]])
        q = torch.tensor([[0.3, 0.4, 0.3]])
        d = fisher_rao_distance(p, q)
        cos_val = (torch.sqrt(p) * torch.sqrt(q)).sum()
        expected = 2.0 * torch.acos(cos_val.clamp(-1 + EPS, 1 - EPS))
        assert torch.allclose(d, expected, atol=1e-5)

    def test_zero_distance_for_identical(self):
        p = torch.tensor([[0.5, 0.3, 0.2]])
        # Clamping introduces small numerical offset; tolerance accounts for this
        assert fisher_rao_distance(p, p).item() < 1e-2

    def test_non_negative(self):
        p = torch.softmax(torch.randn(B, 5), dim=-1)
        q = torch.softmax(torch.randn(B, 5), dim=-1)
        assert (fisher_rao_distance(p, q) >= 0).all()

    def test_symmetry(self):
        p = torch.softmax(torch.randn(B, 3), dim=-1)
        q = torch.softmax(torch.randn(B, 3), dim=-1)
        assert torch.allclose(
            fisher_rao_distance(p, q), fisher_rao_distance(q, p), atol=1e-6
        )

    def test_maximum_distance(self):
        """Max FR distance is π (between orthogonal unit vectors via sphere map)."""
        p = torch.tensor([[1.0, 0.0, 0.0]])
        q = torch.tensor([[0.0, 1.0, 0.0]])
        d = fisher_rao_distance(p, q)
        # With clamping, slightly less than π
        assert d.item() < math.pi + 0.1
        assert d.item() > math.pi - 0.5


class TestCircularDistance:
    def test_eq7_formula(self):
        """d_circ(θ₁, θ₂) = min(|θ₁-θ₂|, 2π-|θ₁-θ₂|) (Eq 7)."""
        t1 = torch.tensor([0.5])
        t2 = torch.tensor([1.0])
        assert circular_distance(t1, t2).item() == pytest.approx(0.5, abs=1e-6)

    def test_wraparound(self):
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([2 * math.pi - 0.1])
        assert circular_distance(t1, t2).item() == pytest.approx(0.2, abs=1e-6)

    def test_zero_distance(self):
        t = torch.tensor([1.5])
        assert circular_distance(t, t).item() < 1e-7

    def test_max_distance_is_pi(self):
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([math.pi])
        assert circular_distance(t1, t2).item() == pytest.approx(math.pi, abs=1e-6)


# ===================================================================
# Product manifold distance (Eq 8)
# ===================================================================

class TestProductManifoldDistance:
    def test_zero_for_identical(self):
        x_c = torch.randn(B, 3)
        x_d = [torch.softmax(torch.randn(B, 4), dim=-1)]
        x_o = [torch.rand(B) * 2 * math.pi]
        d = product_manifold_distance(x_c, x_c, x_d, x_d, x_o, x_o)
        assert d.max().item() < 1e-3

    def test_non_negative(self):
        x_c1 = torch.randn(B, 2)
        x_c2 = torch.randn(B, 2)
        d = product_manifold_distance(x_c1, x_c2, [], [], [], [])
        assert (d >= 0).all()

    def test_squared_is_square_of_unsquared(self):
        x_c1 = torch.randn(B, 2)
        x_c2 = torch.randn(B, 2)
        x_d1 = [torch.softmax(torch.randn(B, 3), dim=-1)]
        x_d2 = [torch.softmax(torch.randn(B, 3), dim=-1)]
        d = product_manifold_distance(x_c1, x_c2, x_d1, x_d2, [], [])
        d_sq = product_manifold_distance_squared(x_c1, x_c2, x_d1, x_d2, [], [])
        assert torch.allclose(d.pow(2), d_sq, atol=1e-4)


class TestPairwiseDistanceMatrices:
    def test_squared_shape(self):
        x_c = torch.randn(5, 2)
        y_c = torch.randn(7, 2)
        D = pairwise_product_distance_matrix(x_c, y_c, [], [], [], [])
        assert D.shape == (5, 7)

    def test_l1_shape(self):
        x_c = torch.randn(5, 2)
        y_c = torch.randn(7, 2)
        D = pairwise_sample_distance_l1(x_c, y_c, [], [], [], [])
        assert D.shape == (5, 7)

    def test_squared_diagonal_zero(self):
        x_c = torch.randn(B, 3)
        x_d = [torch.softmax(torch.randn(B, 4), dim=-1)]
        D = pairwise_product_distance_matrix(x_c, x_c, x_d, x_d, [], [])
        assert D.diag().max().item() < 1e-4

    def test_l1_non_negative(self):
        x_c = torch.randn(B, 2)
        D = pairwise_sample_distance_l1(x_c, x_c, [], [], [], [])
        assert (D >= -1e-6).all()

    def test_l1_vs_squared_ordering_same_for_nearest(self):
        """k-NN topology should be similar for both metrics."""
        x_c = torch.randn(20, 3)
        x_d = [torch.softmax(torch.randn(20, 4), dim=-1)]
        D_sq = pairwise_product_distance_matrix(x_c, x_c, x_d, x_d, [], [])
        D_l1 = pairwise_sample_distance_l1(x_c, x_c, x_d, x_d, [], [])
        D_sq.fill_diagonal_(float("inf"))
        D_l1.fill_diagonal_(float("inf"))
        # Nearest neighbor should often agree
        nn_sq = D_sq.argmin(dim=1)
        nn_l1 = D_l1.argmin(dim=1)
        agreement = (nn_sq == nn_l1).float().mean()
        assert agreement > 0.5  # at least half agree


# ===================================================================
# Geodesic interpolation (Eq 9)
# ===================================================================

class TestGeodesicInterpolation:
    def test_continuous_at_endpoints(self):
        x0 = torch.randn(B, 3)
        x1 = torch.randn(B, 3)
        assert torch.allclose(
            geodesic_interpolation_continuous(x0, x1, torch.tensor(0.0)), x0, atol=1e-6
        )
        assert torch.allclose(
            geodesic_interpolation_continuous(x0, x1, torch.tensor(1.0)), x1, atol=1e-6
        )

    def test_continuous_midpoint(self):
        x0 = torch.zeros(B, 2)
        x1 = torch.ones(B, 2)
        mid = geodesic_interpolation_continuous(x0, x1, torch.tensor(0.5))
        assert torch.allclose(mid, 0.5 * torch.ones(B, 2), atol=1e-6)

    def test_categorical_at_endpoints(self):
        x0 = torch.softmax(torch.randn(B, 5), dim=-1)
        x1 = torch.softmax(torch.randn(B, 5), dim=-1)
        t0 = torch.tensor(0.0)
        t1 = torch.tensor(1.0)
        assert torch.allclose(
            geodesic_interpolation_categorical(x0, x1, t0), x0, atol=1e-4
        )
        assert torch.allclose(
            geodesic_interpolation_categorical(x0, x1, t1), x1, atol=1e-4
        )

    def test_categorical_stays_on_simplex(self):
        x0 = torch.softmax(torch.randn(B, 4), dim=-1)
        x1 = torch.softmax(torch.randn(B, 4), dim=-1)
        t = torch.rand(B)
        xt = geodesic_interpolation_categorical(x0, x1, t)
        assert (xt >= 0).all(), "Negative probabilities"
        assert torch.allclose(xt.sum(dim=-1), torch.ones(B), atol=1e-4)

    def test_categorical_slerp_fallback_for_identical(self):
        """When x0 ≈ x1, should fallback to linear interpolation."""
        x0 = torch.softmax(torch.randn(B, 3), dim=-1)
        x1 = x0.clone()
        t = torch.rand(B)
        xt = geodesic_interpolation_categorical(x0, x1, t)
        assert torch.allclose(xt.sum(dim=-1), torch.ones(B), atol=1e-4)

    def test_ordinal_at_endpoints(self):
        t0 = torch.rand(B) * 2 * math.pi
        t1 = torch.rand(B) * 2 * math.pi
        assert torch.allclose(
            geodesic_interpolation_ordinal(t0, t1, torch.tensor(0.0)),
            t0 % (2 * math.pi),
            atol=1e-5,
        )

    def test_ordinal_wraparound(self):
        """Interpolation should take the shortest arc."""
        t0 = torch.tensor([0.1])
        t1 = torch.tensor([2 * math.pi - 0.1])
        mid = geodesic_interpolation_ordinal(t0, t1, torch.tensor(0.5))
        # Shortest path crosses 0/2π boundary
        expected_angle = 0.0  # midpoint of the short arc
        assert circular_distance(mid, torch.tensor([expected_angle])).item() < 0.2


# ===================================================================
# Conditional velocities (Eq 12)
# ===================================================================

class TestConditionalVelocities:
    def test_continuous_is_displacement(self):
        x0 = torch.randn(B, 3)
        x1 = torch.randn(B, 3)
        u = conditional_velocity_continuous(x0, x1)
        assert torch.allclose(u, x1 - x0, atol=1e-6)

    def test_categorical_finite(self):
        x0 = torch.softmax(torch.randn(B, 5), dim=-1)
        x1 = torch.softmax(torch.randn(B, 5), dim=-1)
        t = torch.rand(B, 1)
        u = conditional_velocity_categorical(x0, x1, t)
        assert u.isfinite().all()

    def test_categorical_fallback_for_identical(self):
        x0 = torch.softmax(torch.randn(B, 4), dim=-1)
        t = torch.rand(B, 1)
        u = conditional_velocity_categorical(x0, x0, t)
        assert u.isfinite().all()
        assert u.abs().max() < 1e-3

    def test_ordinal_is_signed_displacement(self):
        t0 = torch.tensor([0.5])
        t1 = torch.tensor([1.5])
        u = conditional_velocity_ordinal(t0, t1)
        assert u.item() == pytest.approx(1.0, abs=1e-5)

    def test_ordinal_shortest_arc(self):
        t0 = torch.tensor([0.1])
        t1 = torch.tensor([2 * math.pi - 0.1])
        u = conditional_velocity_ordinal(t0, t1)
        # Shortest arc is -0.2 (going backwards)
        assert u.item() == pytest.approx(-0.2, abs=1e-5)


# ===================================================================
# Ordinal encoding
# ===================================================================

class TestOrdinalEncoding:
    def test_angle_range(self):
        """θ_k = πk/(K+1) should be in (0, π)."""
        levels = torch.arange(5)
        angles = ordinal_to_angle(levels, 5)
        assert (angles >= 0).all()
        assert (angles < math.pi).all()

    def test_roundtrip(self):
        K = 5
        levels = torch.arange(K)
        angles = ordinal_to_angle(levels, K)
        recovered = angle_to_ordinal(angles, K)
        assert torch.equal(levels, recovered)

    def test_monotonic(self):
        K = 10
        angles = ordinal_to_angle(torch.arange(K), K)
        diffs = angles[1:] - angles[:-1]
        assert (diffs > 0).all(), "Angles should be strictly increasing"
