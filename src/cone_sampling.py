"""Cached angular FEM evaluation for cone-sampler diagnostics.

This accelerates the *same* radial facet interpolant as ConeEigenpair. It does
not make that interpolant harmonic, certify its error, or validate a sampler.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lqg-matplotlib")
import numpy as np
from matplotlib.tri import Triangulation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from .cone_geometry import ConeEigenpair, spherical_cone_mesh


def enriched_cone_eigenpair(covariance, refinement=4, strategy="edge"):
    """Optional conforming enrichment of every all-boundary corner facet.

    Split each corner facet's opposite *interior* edge, along with the adjacent
    facet, to connect it to interior degrees of freedom. The failed alternative
    `strategy="centroid"` is retained for audit: its nodes couple only to zero
    boundary nodes and cannot fix the principal eigenmode's corner patch.
    Neither variant certifies correct corner exponents or harmonicity.
    """
    vertices, faces, boundary = spherical_cone_mesh(covariance, refinement)
    vertices = list(vertices)
    next_faces = []
    inserted = 0
    if strategy == "centroid":
        for face in faces:
            if np.all(boundary[face]):
                point = np.asarray([vertices[i] for i in face]).sum(axis=0)
                point /= np.linalg.norm(point)
                center = len(vertices)
                vertices.append(point)
                a,b,c = face
                next_faces.extend(((a,b,center),(b,c,center),(c,a,center)))
                inserted += 1
            else:
                next_faces.append(face)
    elif strategy == "edge":
        counts={}
        for a,b,c in faces:
            for edge in [(a,b),(a,c),(b,c)]:
                edge=tuple(sorted(edge));counts[edge]=counts.get(edge,0)+1
        split={}
        for face in faces:
            if np.all(boundary[face]):
                a,b,c=face
                edge=next(tuple(sorted(edge)) for edge in [(a,b),(a,c),(b,c)]
                          if counts[tuple(sorted(edge))]==2)
                if edge not in split:
                    point=vertices[edge[0]]+vertices[edge[1]]
                    point/=np.linalg.norm(point)
                    split[edge]=len(vertices);vertices.append(point);inserted+=1
        for face in faces:
            a,b,c=face
            edges=[tuple(sorted(edge)) for edge in [(a,b),(a,c),(b,c)]]
            chosen=[edge for edge in edges if edge in split]
            if len(chosen)>1:
                raise ValueError("Refine the base mesh before corner-edge enrichment")
            if chosen:
                edge=chosen[0];third=next(i for i in face if i not in edge)
                middle=split[edge]
                next_faces.extend(((edge[0],middle,third),(middle,edge[1],third)))
            else:
                next_faces.append(face)
    else:
        raise ValueError("strategy must be edge or centroid")
    vertices, faces = np.asarray(vertices), np.asarray(next_faces)
    boundary = np.r_[boundary,np.zeros(inserted,dtype=bool)]
    triangles = vertices[faces]
    e01, e02 = triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]
    twice_area = np.linalg.norm(np.cross(e01,e02),axis=1)
    cot0 = np.einsum("ij,ij->i",e01,e02)/twice_area
    cot1 = np.einsum("ij,ij->i",-e01,e02-e01)/twice_area
    cot2 = np.einsum("ij,ij->i",-e02,e01-e02)/twice_area
    stiffness = np.zeros((len(faces),3,3))
    for a,b,cot in ((0,1,cot2),(0,2,cot1),(1,2,cot0)):
        stiffness[:,a,a] += cot/2
        stiffness[:,b,b] += cot/2
        stiffness[:,a,b] -= cot/2
        stiffness[:,b,a] -= cot/2
    mass = twice_area[:,None,None]/24*(np.ones((3,3))+np.eye(3))
    rows=np.broadcast_to(faces[:,:,None],stiffness.shape).ravel()
    cols=np.broadcast_to(faces[:,None,:],stiffness.shape).ravel()
    shape=(len(vertices),len(vertices))
    K=coo_matrix((stiffness.ravel(),(rows,cols)),shape=shape).tocsr()
    M=coo_matrix((mass.ravel(),(rows,cols)),shape=shape).tocsr()
    interior=np.flatnonzero(~boundary)
    if len(interior)<2:
        raise ValueError("Use a finer mesh")
    eigenvalues,modes=eigsh(K[interior][:,interior],M=M[interior][:,interior],
                           k=1,sigma=0.,which="LM",tol=1e-10)
    mode=modes[:,0]
    if mode.sum()<0:
        mode=-mode
    values=np.zeros(len(vertices)); values[interior]=mode
    output=ConeEigenpair(vertices,faces,boundary,values,float(eigenvalues[0]),refinement)
    output.corner_enriched=True
    output.inserted_corner_nodes=inserted
    output.enrichment_strategy=strategy
    return output


class FastFEMMode:
    """Locate rays in a projected spherical mesh, then interpolate on facets.

    Cone rays project to a planar simplex in the basis of the three corner
    rays. Great-circle mesh edges become straight edges there. An explicit
    triangulation finder therefore locates exactly the original facet, not a
    new Delaunay interpolation of its vertices.
    """
    def __init__(self, eigenpair, covariance):
        self.vertices = eigenpair.vertices
        self.faces = eigenpair.faces
        self.boundary = eigenpair.boundary
        self.values = eigenpair.values
        self.eigenvalue = eigenpair.eigenvalue
        self.refinement = eigenpair.refinement
        self._original = eigenpair
        C = np.asarray(covariance, dtype=float)
        if C.shape != (3, 3):
            raise ValueError("FEM interpolation requires a 3D cone")
        self._R = np.linalg.cholesky(C)
        if np.any(self.values < 0) or not np.all(np.isfinite(self.values)):
            raise ValueError("FEM eigenmode is not nonnegative; do not clip a sampling density")
        corners = np.linalg.solve(self._R, np.eye(3)).T
        corners /= np.linalg.norm(corners, axis=1)[:, None]
        self._corner_inverse = np.linalg.inv(corners.T)
        weights = self.vertices @ self._corner_inverse.T
        bary = weights / weights.sum(axis=1)[:, None]
        triangulation = Triangulation(bary[:, 1], bary[:, 2], triangles=self.faces)
        self._find = triangulation.get_trifinder()
        self._inverse = np.linalg.inv(self.vertices[self.faces].transpose(0, 2, 1))

    @property
    def alpha1(self):
        return float(np.sqrt(self.eigenvalue + .25))

    @property
    def maximum(self):
        return float(self.values.max())

    def angular_mode(self, directions):
        directions = np.asarray(directions, dtype=float)
        scalar = directions.ndim == 1
        points = np.atleast_2d(directions)
        if points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Expected finite 3D directions")
        radius = np.linalg.norm(points, axis=1)
        if np.any(radius == 0):
            raise ValueError("Directions must be nonzero")
        unit = points / radius[:, None]
        weights = unit @ self._corner_inverse.T
        valid = np.all(unit @ self._R.T >= -1e-13, axis=1) & (weights.sum(axis=1) > 0)
        output = np.zeros(len(points))
        if np.any(valid):
            indices = np.flatnonzero(valid)
            bary = weights[valid] / weights[valid].sum(axis=1)[:, None]
            faces = self._find(bary[:, 1], bary[:, 2])
            located = faces >= 0
            indices = indices[located]
            faces = faces[located]
            local = np.einsum("nij,nj->ni", self._inverse[faces], unit[indices])
            local = np.maximum(local, 0)  # ray location tolerance only, not eigenmode clipping
            local /= local.sum(axis=1)[:, None]
            output[indices] = np.einsum("ni,ni->n", local, self.values[self.faces[faces]])
        return float(output[0]) if scalar else output

    def log_h(self, points):
        points = np.asarray(points, dtype=float)
        radius = np.linalg.norm(points, axis=-1)
        if np.any(radius <= 0):
            raise ValueError("h is singular at the vertex")
        with np.errstate(divide="ignore"):
            return (-self.alpha1-.5)*np.log(radius) + np.log(self.angular_mode(points))
