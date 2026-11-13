"""Surface-FEM approximation to the Brownian cone's angular eigenpair.

Budd--Castro (2022), (22), (26), (41). This computes geometry, not an
excursion sampler. Linear elements live on planar facets inscribed in the
sphere. Mesh refinement is necessary; eigenvalues are not certified bounds.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class ConeEigenpair:
    vertices: np.ndarray
    faces: np.ndarray
    boundary: np.ndarray
    values: np.ndarray
    eigenvalue: float
    refinement: int

    @property
    def alpha1(self):
        return float(np.sqrt(self.eigenvalue + 0.25))

    @property
    def string_susceptibility(self):
        return 1.0 - self.alpha1

    def angular_mode(self, directions):
        """Radially project onto facets; interpolate the positive eigenmode.

        O(number of faces) per query. Intended for diagnostics, not bulk
        h-transform sampling. Values outside the cone are zero.
        """
        directions = np.asarray(directions, dtype=float)
        scalar = directions.ndim == 1
        directions = np.atleast_2d(directions)
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms == 0):
            raise ValueError("Directions must be nonzero")
        inverse = np.linalg.inv(self.vertices[self.faces].transpose(0, 2, 1))
        output = []
        for direction in directions / norms[:, None]:
            weights = np.einsum("fij,j->fi", inverse, direction)
            candidates = np.flatnonzero(np.all(weights >= -1e-12, axis=1))
            if not len(candidates):
                output.append(0.0)
                continue
            index = candidates[0]
            bary = np.maximum(weights[index], 0)
            bary /= bary.sum()
            output.append(float(bary @ self.values[self.faces[index]]))
        return output[0] if scalar else np.asarray(output)

    def log_h(self, points):
        """Approximate log h(x) = (-alpha1-1/2)log|x|+log m1(x/|x|)."""
        points = np.asarray(points, dtype=float)
        radius = np.linalg.norm(points, axis=-1)
        if np.any(radius <= 0):
            raise ValueError("h is singular at the cone vertex")
        with np.errstate(divide="ignore"):
            return (-self.alpha1 - 0.5) * np.log(radius) + np.log(self.angular_mode(points))


def spherical_cone_mesh(covariance, refinement=4):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.allclose(covariance, covariance.T):
        raise ValueError("Expected symmetric 3x3 covariance")
    if not isinstance(refinement, int) or refinement < 0:
        raise ValueError("refinement must be a nonnegative integer")
    transform = np.linalg.cholesky(covariance)
    corners = np.linalg.solve(transform, np.eye(3)).T
    corners /= np.linalg.norm(corners, axis=1)[:, None]
    vertices = list(corners)
    faces = [(0, 1, 2)]
    boundary_edges = {(0, 1), (0, 2), (1, 2)}
    for _ in range(refinement):
        midpoints = {}
        next_boundary = set()

        def midpoint(a, b):
            edge = tuple(sorted((a, b)))
            if edge not in midpoints:
                point = vertices[a] + vertices[b]
                point /= np.linalg.norm(point)
                midpoints[edge] = len(vertices)
                vertices.append(point)
                if edge in boundary_edges:
                    next_boundary.update((tuple(sorted((a, midpoints[edge]))),
                                          tuple(sorted((b, midpoints[edge])))))
            return midpoints[edge]

        next_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            next_faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        faces = next_faces
        boundary_edges = next_boundary
    boundary = np.zeros(len(vertices), dtype=bool)
    for a, b in boundary_edges:
        boundary[[a, b]] = True
    return np.asarray(vertices), np.asarray(faces), boundary


def cone_eigenpair(covariance, refinement=4):
    """Compute the first Dirichlet eigenpair on R^-1 R_+^3 intersect S^2."""
    vertices, faces, boundary = spherical_cone_mesh(covariance, refinement)
    triangles = vertices[faces]
    e01, e02 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    twice_area = np.linalg.norm(np.cross(e01, e02), axis=1)
    if np.any(twice_area < 1e-20):
        raise ValueError("Degenerate mesh; covariance is too ill conditioned")
    cot0 = np.einsum("ij,ij->i", e01, e02) / twice_area
    cot1 = np.einsum("ij,ij->i", -e01, e02-e01) / twice_area
    cot2 = np.einsum("ij,ij->i", -e02, e01-e02) / twice_area
    stiffness = np.zeros((len(faces), 3, 3))
    for a, b, cot in ((0, 1, cot2), (0, 2, cot1), (1, 2, cot0)):
        stiffness[:, a, a] += cot / 2
        stiffness[:, b, b] += cot / 2
        stiffness[:, a, b] -= cot / 2
        stiffness[:, b, a] -= cot / 2
    mass = twice_area[:, None, None] / 24 * (np.ones((3, 3)) + np.eye(3))
    rows = np.broadcast_to(faces[:, :, None], stiffness.shape).ravel()
    cols = np.broadcast_to(faces[:, None, :], stiffness.shape).ravel()
    shape = (len(vertices), len(vertices))
    K = coo_matrix((stiffness.ravel(), (rows, cols)), shape=shape).tocsr()
    M = coo_matrix((mass.ravel(), (rows, cols)), shape=shape).tocsr()
    interior = np.flatnonzero(~boundary)
    if len(interior) < 2:
        raise ValueError("Use refinement >= 3 to resolve the eigenmode")
    eigenvalues, modes = eigsh(K[interior][:, interior], M=M[interior][:, interior],
                              k=1, sigma=0.0, which="LM", tol=1e-10)
    mode = modes[:, 0]
    if mode.sum() < 0:
        mode = -mode
    values = np.zeros(len(vertices))
    values[interior] = mode
    return ConeEigenpair(vertices, faces, boundary, values, float(eigenvalues[0]), refinement)
