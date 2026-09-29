"""Slab intersections against an independent scalar oracle and autograd checks."""

import math

import numpy as np
import pytest
import torch


def oracle(origin, direction, box, bound=1e10, invalid=1e10):
    lower, upper = -math.inf, math.inf
    for axis in range(3):
        o, d = origin[axis], direction[axis]
        low, high = box[axis], box[axis + 3]
        if d == 0:
            if o < low or o > high:
                return invalid, invalid
            continue
        a, b = (low - o) / d, (high - o) / d
        lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
    lower = min(bound, max(0.0, lower))
    upper = min(bound, max(0.0, upper))
    return (invalid, invalid) if upper <= lower else (lower, upper)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ray_along_box_face(nerfstudio_aabb, axis, side, dtype):
    origin = torch.zeros(1, 3, dtype=dtype)
    direction = torch.zeros_like(origin)
    origin[0, axis] = side
    moving = (axis + 1) % 3
    origin[0, moving] = -2.0
    direction[0, moving] = 1.0
    box = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=dtype)
    result = nerfstudio_aabb.intersect_aabb(origin, direction, box)
    torch.testing.assert_close(result[0], torch.tensor([1.0], dtype=dtype))
    torch.testing.assert_close(result[1], torch.tensor([3.0], dtype=dtype))


@pytest.mark.parametrize("negative_zero", [False, True])
def test_mixed_hit_miss_inside_edge_and_behind(nerfstudio_aabb, negative_zero):
    dtype = torch.float64
    origins = torch.tensor(
        [[1.0, -2.0, 1.0], [2.0, -2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [0.0, 2.0, 0.0]], dtype=dtype
    )
    z = -0.0 if negative_zero else 0.0
    directions = torch.tensor([[z, 1.0, z], [z, 1.0, z], [z, 1.0, z], [z, -1.0, z], [z, 1.0, z]], dtype=dtype)
    box = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=dtype)
    expected = torch.tensor(
        [oracle(o, d, box.tolist(), 8.0, -77.0) for o, d in zip(origins.tolist(), directions.tolist())], dtype=dtype
    )
    actual = torch.stack(
        nerfstudio_aabb.intersect_aabb(origins, directions, box, max_bound=8.0, invalid_value=-77.0), dim=-1
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_randomized_discrete_geometry_against_oracle(nerfstudio_aabb):
    rng = np.random.default_rng(321)
    origins = rng.choice([-3.0, -1.0, 0.0, 1.0, 3.0], size=(500, 3))
    directions = rng.choice([-1.0, 0.0, 1.0], size=(500, 3))
    keep = np.any(directions != 0, axis=1)
    origins, directions = origins[keep], directions[keep]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    box = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    expected = torch.tensor([oracle(o, d, box) for o, d in zip(origins, directions)], dtype=torch.float64)
    o, d, b = (torch.tensor(v, dtype=torch.float64) for v in [origins, directions, box])
    actual = torch.stack(nerfstudio_aabb.intersect_aabb(o, d, b), dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


def test_backward_on_face_hit_is_finite(nerfstudio_aabb):
    o = torch.tensor([[1.0, -2.0, 0.0]], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    box = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    low, high = nerfstudio_aabb.intersect_aabb(o, d, box)
    go, gd = torch.autograd.grad((low + high).sum(), (o, d))
    assert torch.isfinite(go).all() and torch.isfinite(gd).all()
    # Differentiate along the face; the perpendicular boundary is not smooth.
    torch.testing.assert_close(go[:, 1], torch.tensor([-2.0], dtype=o.dtype))
    torch.testing.assert_close(gd[:, 1], torch.tensor([-4.0], dtype=o.dtype))


def test_gradcheck_away_from_boundaries(nerfstudio_aabb):
    o = torch.tensor([[-3.0, 0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([[1.0, 0.02, 0.03]], dtype=torch.float64, requires_grad=True)
    box = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x, v: nerfstudio_aabb.intersect_aabb(x, v, box), (o, d))


def test_gradcheck_parallel_interior(nerfstudio_aabb):
    o = torch.tensor([[-3.0, 0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    box = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x, v: nerfstudio_aabb.intersect_aabb(x, v, box), (o, d))


def test_tiny_nonzero_components_are_not_clamped(nerfstudio_aabb):
    o = torch.tensor([[-1.1, -2e12, 0.0]], dtype=torch.float64)
    d = torch.tensor([[1e-12, 1.0, 0.0]], dtype=torch.float64)
    box = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    actual = torch.stack(nerfstudio_aabb.intersect_aabb(o, d, box, max_bound=1e14), dim=-1)
    expected = torch.tensor([oracle(o[0].tolist(), d[0].tolist(), box.tolist(), 1e14)], dtype=o.dtype)
    torch.testing.assert_close(actual, expected)


def test_empty_noncontiguous_and_nonmutation(nerfstudio_aabb):
    box = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    o = torch.tensor([[-2.0, 8.0, 0.0, 8.0, 0.0, 8.0]], dtype=torch.float64)[:, ::2]
    d = torch.tensor([[1.0, 9.0, 0.0, 9.0, 0.0, 9.0]], dtype=torch.float64)[:, ::2]
    original_o, original_d = o.clone(), d.clone()
    result = nerfstudio_aabb.intersect_aabb(o, d, box)
    torch.testing.assert_close(result[0], torch.tensor([1.0], dtype=o.dtype))
    torch.testing.assert_close(result[1], torch.tensor([3.0], dtype=o.dtype))
    torch.testing.assert_close(o, original_o, atol=0, rtol=0)
    torch.testing.assert_close(d, original_d, atol=0, rtol=0)
    result = nerfstudio_aabb.intersect_aabb(o[:0], d[:0], box)
    assert result[0].shape == result[1].shape == (0,)


@pytest.fixture(scope="module")
def nerfstudio_aabb():
    from nerfstudio.utils import math as module

    return module
