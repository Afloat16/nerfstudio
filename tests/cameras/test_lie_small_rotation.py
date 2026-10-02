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

"""Direct-product rotation exponential compared with a matrix exponential."""

import pytest
import torch

from nerfstudio.cameras.lie_groups import exp_map_SO3xR3


def reference(tangent):
    w = tangent[:, 3:]
    z = torch.zeros_like(w[:, 0])
    skew = torch.stack([z, -w[:, 2], w[:, 1], w[:, 2], z, -w[:, 0], -w[:, 1], w[:, 0], z], -1).reshape(-1, 3, 3)
    return torch.cat([torch.matrix_exp(skew), tangent[:, :3, None]], dim=-1)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 1e-5, 0.001, 0.009, 0.01, 0.1, 1.0])
def test_small_rotation_exponential_and_gradient(angle):
    tangent = torch.tensor([[0.2, -0.1, 0.3, angle, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    actual = exp_map_SO3xR3(tangent)
    expected = reference(tangent)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weights = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
    actual_grad = torch.autograd.grad((actual * weights).sum(), tangent)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), tangent)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-11, rtol=1e-11)
    R = actual[:, :3, :3]
    torch.testing.assert_close(R.transpose(-1, -2) @ R, torch.eye(3, dtype=R.dtype)[None], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(1, dtype=R.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.001, 0.1])
def test_small_rotation_hessian_matches_matrix_exponential(angle):
    tangent = torch.tensor([0.2, -0.1, 0.3, angle, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    weights = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
    actual = torch.autograd.functional.hessian(lambda x: (exp_map_SO3xR3(x[None]) * weights).sum(), tangent)
    expected = torch.autograd.functional.hessian(lambda x: (reference(x[None]) * weights).sum(), tangent)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.autograd.gradgradcheck(exp_map_SO3xR3, (tangent[None],), atol=1e-6, rtol=1e-5)
