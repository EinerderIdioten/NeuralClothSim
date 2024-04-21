import openmesh
import numpy as np
import tensorflow as tf
import open3d 


class MeshTopo:
    def __init__(self, mesh):
        self.mesh = mesh

        self.vN = mesh.n_vertices()
        self.fN = mesh.n_faces()
        self.eN = mesh.n_edges()
        self.hN = mesh.n_halfedges()

        self.vb, self.h2f, self.h2v, self.f2v, self.f2f = self.assign()


    def assign(self):
        mesh = self.mesh
        vb = np.zeros(self.vN, dtype=bool)
        h2f = np.full(self.hN, -1, dtype=int)
        h2v = np.full(self.hN, -1, dtype=int)

        f2v = np.full((self.fN, 3), -1, dtype=int)  
        f2f = np.full((self.fN, 3), -1, dtype=int)  
        f2h = np.full((self.fN, 3), -1, dtype=int)
        
        for hh in mesh.halfedges():
            h2f[hh.idx()] = mesh.face_handle(hh).idx()
            h2v[hh.idx()] = mesh.to_vertex_handle(hh).idx()
            vb[h2v[hh.idx()]] = (vb[h2v[hh.idx()]] > 0) or (h2f[hh.idx()] < 0)
        
        for fh0 in mesh.faces():
            for i, heh in enumerate(mesh.fh(fh0)):
                f2h[fh0.idx()][i] = heh.idx()
                f2v[fh0.idx()][i] = mesh.to_vertex_handle(heh).idx()
                f2f[fh0.idx()][i] = h2f[f2h[fh0.idx()][0] ^ 1]

        return vb, h2f, h2v, f2v, f2f