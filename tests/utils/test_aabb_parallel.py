# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slab intersections for rays parallel to faces, including origins on faces."""

import pytest
import torch

from nerfstudio.utils.math import intersect_aabb


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coordinate", [0.0, 1.0])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_face_parallel_ray_has_finite_entry_exit(dtype, coordinate, axis):
    origin = torch.tensor([[coordinate, 0.5, -1.0]], dtype=dtype).roll(axis, dims=-1)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype).roll(axis, dims=-1)
    aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=dtype)
    near, far = intersect_aabb(origin, direction, aabb)
    torch.testing.assert_close(near, torch.tensor([1.0], dtype=dtype))
    torch.testing.assert_close(far, torch.tensor([2.0], dtype=dtype))


@pytest.mark.parametrize("coordinate", [-0.1, 1.1])
def test_outside_parallel_slab_remains_a_miss(coordinate):
    aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    near, far = intersect_aabb(torch.tensor([[coordinate, 0.5, -1.0]]), torch.tensor([[0.0, 0.0, 1.0]]), aabb)
    assert near.item() == far.item() == 1e10


def test_parallel_inside_slab_gradient_is_finite():
    aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    origin = torch.tensor([[0.0, 0.5, -1.0]], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    near, far = intersect_aabb(origin, direction, aabb)
    gradients = torch.autograd.grad((near + far).sum(), (origin, direction))
    torch.testing.assert_close(gradients[0], torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64))
    torch.testing.assert_close(gradients[1], torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float64))
