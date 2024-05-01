from typing import Any
import tensorflow as tf
from utils.mesh import vertex_normals, face_normals, compute_vertex_face_adjacency, face_normals_one_frame, compute_sparse_adjacency, nonzero_slice_row
from dev_utils import compute_edge_normals, compute_midpoint, compute_projection
from model.cloth import Garment
import numpy as np
import openmesh as om
from mesh_topo import MeshTopo
import math
# Mass-spring model
class EdgeLoss:
    def __init__(self, garment):
        self.edges = garment.edges
        self.edge_lengths_true = garment.edge_lengths

    @tf.function
    def __call__(self, vertices):
        edges = tf.gather(vertices, self.edges[:, 0], axis=1) - tf.gather(
            vertices, self.edges[:, 1], axis=1
        )
        edge_lengths = tf.norm(edges, axis=-1)
        edge_difference = edge_lengths - self.edge_lengths_true
        loss = edge_difference**2
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.abs(edge_difference)
        error = tf.reduce_mean(error)
        return loss, error


# Baraff '98 cloth model (squared)
class ClothLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices

    @tf.function
    def __call__(self, vertices):
        dX = tf.stack(
            [
                tf.gather(vertices, self.faces[:, 1], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
                tf.gather(vertices, self.faces[:, 2], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
            ],
            axis=2,
        )
        
        w = tf.einsum("abcd,bce->abed", dX, self.uv_matrices)

        stretch = tf.norm(w, axis=-1) - 1
        print(f'shape of stretch is {stretch.shape}')
        stretch_loss = self.face_areas[:, None] * stretch**2
        stretch_loss = tf.reduce_sum(stretch_loss, axis=[1, 2])
        stretch_loss = tf.reduce_mean(stretch_loss)

        

        stretch_error = (
            self.face_areas[:, None] * tf.abs(stretch) * (0.5 / self.total_area)
        )
        stretch_error = tf.reduce_mean(tf.reduce_sum(stretch_error, axis=-1))

        shear = tf.reduce_sum(tf.multiply(w[:, :, 0], w[:, :, 1]), axis=-1)
        shear_loss = shear**2
        shear_loss *= self.face_areas
        shear_loss = tf.reduce_sum(shear_loss, axis=1)
        shear_loss = tf.reduce_mean(shear_loss)
        shear_error = self.face_areas * tf.abs(shear) * (1 / self.total_area)
        shear_error = tf.reduce_mean(tf.reduce_sum(shear_error, axis=-1))

        # print(f'the shape of stretch loss is {stretch_loss.shape}')
        return stretch_loss, stretch_error, shear_loss, shear_error


# Saint-Venant Kirchhoff
class StVKLoss:
    def __init__(self, garment, l, m):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices
        self.l = l
        self.m = m

    @tf.function
    def __call__(self, vertices):
        dX = tf.stack(
            [
                tf.gather(vertices, self.faces[:, 1], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
                tf.gather(vertices, self.faces[:, 2], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
            ],
            axis=-1,
        )
        F = dX @ self.uv_matrices
        Ft = tf.linalg.matrix_transpose(F)
        G = 0.5 * (Ft @ F - tf.eye(2))
        S = self.m * G + (0.5 * self.l * tf.einsum("...ii", G))[
            ..., None, None
        ] * tf.eye(2, batch_shape=tf.shape(G)[:2])
        loss = tf.einsum("...ii", tf.linalg.matrix_transpose(S) @ G)
        loss *= self.face_areas
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_sum(loss)
        error = loss / (self.total_area)

        return loss, error


class BendingLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_adjacency = garment.face_adjacency
        face_areas = garment.face_areas[garment.face_adjacency].sum(-1)
        edge_lengths = garment.face_adjacency_edge_lengths
        self.stiffness_scaling = edge_lengths**2 / (8 * face_areas)
        self.angle_true = garment.face_dihedral

    @tf.function
    def __call__(self, vertices):
        mesh_face_normals = face_normals(vertices, self.faces)
        normals0 = tf.gather(mesh_face_normals, self.face_adjacency[:, 0], axis=1)
        normals1 = tf.gather(mesh_face_normals, self.face_adjacency[:, 1], axis=1)
        cos = tf.einsum("abc,abc->ab", normals0, normals1)
        sin = tf.norm(tf.linalg.cross(normals0, normals1), axis=-1)
        angle = tf.math.atan2(sin, cos) - self.angle_true
        loss = angle**2
        error = tf.abs(angle)
        loss *= self.stiffness_scaling
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.reduce_mean(error)
        return loss, error


# Fast estimation of SDF
class CollisionLoss:
    def __init__(self, body, collision_threshold=0.004):
        self.body_faces = body.faces
        self.collision_vertices = tf.constant(body.collision_vertices)
        self.collision_threshold = collision_threshold

    @tf.function
    def __call__(self, vertices, body_vertices, indices):
        # Compute body vertex normals
        body_vertex_normals = vertex_normals(body_vertices, self.body_faces)
        # Gather collision vertices
        body_vertices = tf.gather(body_vertices, self.collision_vertices, axis=1)
        body_vertex_normals = tf.gather(
            body_vertex_normals, self.collision_vertices, axis=1
        )
        # Compute loss
        cloth_to_body = vertices - tf.gather_nd(body_vertices, indices)
        body_vertex_normals = tf.gather_nd(body_vertex_normals, indices)
        normal_dist = tf.einsum("abc,abc->ab", cloth_to_body, body_vertex_normals)
        loss = tf.minimum(normal_dist - self.collision_threshold, 0.0) ** 2
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.math.less(normal_dist, 0.0)
        error = tf.cast(error, tf.float32)
        error = tf.reduce_mean(error)
        return loss, error


class GravityLoss:
    def __init__(self, vertex_area, density=0.15, gravity=[0, 0, -9.81]):
        self.vertex_mass = density * vertex_area[:, None]
        self.gravity = tf.constant(gravity, tf.float32)

    @tf.function
    def __call__(self, vertices):
        loss = -self.vertex_mass * vertices * self.gravity
        loss = tf.reduce_sum(loss, axis=[1, 2])
        loss = tf.reduce_mean(loss)
        return loss


class InertiaLoss:
    def __init__(self, dt, vertex_area, density=0.15):
        self.dt = dt
        self.vertex_mass = density * vertex_area
        self.total_mass = tf.reduce_sum(self.vertex_mass)

    @tf.function
    def __call__(self, vertices):
        x0, x1, x2 = tf.unstack(vertices, axis=1)
        x_proj = 2 * x1 - x0
        x_proj = tf.stop_gradient(x_proj)
        dx = x2 - x_proj
        loss = (0.5 / self.dt**2) * self.vertex_mass[:, None] * dx**2
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_sum(loss)
        error = self.vertex_mass * tf.norm(dx, axis=-1)
        error = tf.reduce_sum(error, axis=-1) / self.total_mass
        error = tf.reduce_mean(error)
        return loss, error


class PinningLoss:
    def __init__(self, garment, pin_blend_weights=False):
        self.indices = garment.pinning_vertices
        self.vertices = garment.vertices[self.indices]
        self.pin_blend_weights = pin_blend_weights
        if pin_blend_weights:
            self.blend_weights = garment.blend_weights[self.indices]

    @tf.function
    def __call__(self, unskinned, blend_weights):
        loss = tf.gather(unskinned, self.indices, axis=-2) - self.vertices
        loss = loss**2
        loss = tf.reduce_sum(loss, axis=[1, 2])
        loss = tf.reduce_mean(loss)
        if self.pin_blend_weights:
            _loss = tf.gather(blend_weights, self.indices, axis=-2) - self.blend_weights
            _loss = _loss**2
            _loss = tf.reduce_mean(_loss, 0)
            loss += 1e2 * tf.reduce_sum(_loss)
        return loss

class HingePairLoss:
    def __init__(self, garment:Garment, weighted_by_tipangles:bool, normalize_by_pairnum:bool):
        self.edges = garment.edges
        self.faces = garment.faces
        
        self.edge_lengths_true = garment.edge_lengths
        self.n_pairs = 0

        self.weighted_by_tipangles = weighted_by_tipangles
        self.normalized_by_pairnum = normalize_by_pairnum

    @tf.function
    def handleNormalPair(self, face_normals, n1, n2):
        if self.normalized_by_pairnum:
            self.n_pairs+=1
        
        normal1 = tf.gather(face_normals, n1)
        normal2 = tf.gather(face_normals, n2)

        squared_norm = tf.reduce_sum(tf.square(normal1 - normal2))
        #if self.weighted_by_tipangles:
        #    angle1 = tf.gather_nd(self.angles, [n1])
        #    angle2 = tf.gather_nd(self.angles, [n2])
        #    return squared_norm * angle1 * angle2
        #else:
        #    return squared_norm
        return squared_norm

    @tf.function
    def __call__(self, vertices_with_time_steps):

        vshape = vertices_with_time_steps.shape
        time_steps, num_vertices = vshape[1], vshape[2]

        #energy_per_vertex= tf.zeros([num_vertices], dtype=tf.float32)
        #partition_indices = tf.fill([num_vertices, 2], -1)
        
        energy_per_vertex = [0.0] * num_vertices
        mesh_face_normals_alltime = face_normals(vertices_with_time_steps, self.faces)

        for t in tf.range(time_steps):
            mesh_face_normals = mesh_face_normals_alltime[:,t,...]
            
            VF, VFi = compute_vertex_face_adjacency(self.faces)
            #VF, VFi = tf.ragged.constant(VF), tf.ragged.constant(VFi)

            for vert in range(num_vertices):
                
                current_energy = float('inf')
                adjacent_faces = VF[vert]
                adjacent_facesi = VFi[vert]
                adjacent_faces_count = len(adjacent_faces)


                for edge1 in range(adjacent_faces_count-2):
                    for edge2 in range(edge1 + 2, adjacent_faces_count - 1 \
                                    if edge1 == 0 else adjacent_faces_count):
                        print(f'handling edge1 {edge1} and edge2 {edge2}')

                        # Handle normal pairs from edge1 to edge2
                        edge1sum = 0.0
                        for n1 in range(edge1, edge2):
                            for n2 in range(n1 + 1, edge2):
                                edge1sum += self.handleNormalPair(mesh_face_normals, n1, n2)

                        # Handle normal pairs from edge2 to the end, and then from the start to edge1
                        edge2sum = 0.0
                        for n1 in range(edge2, adjacent_faces_count):
                            for n2 in range(n1 + 1, adjacent_faces_count):
                                edge2sum += self.handleNormalPair(mesh_face_normals, n1, n2)
                            for n2 in range(edge1):
                                edge2sum += self.handleNormalPair(mesh_face_normals, n1, n2)

                        # Additional loop for handling pairs from the start to edge1
                        for n1 in range(edge1):
                            for n2 in range(n1 + 1, edge1):
                                edge2sum += self.handleNormalPair(mesh_face_normals, n1, n2)

                        local_energy = 0.5*(edge1sum + edge2sum)

                        if self.normalized_by_pairnum:
                            local_energy/self.n_pairs
                        if local_energy < current_energy:
                            current_energy = local_energy
                            #partition_indices[vert] = [edge1, edge2]
                
                # add the min energy per time step 
                
                #energy_per_vertex = tf.tensor_scatter_nd_update(
                #    energy_per_vertex,
                #    [[vert]],
                #    current_energy+energy_per_vertex[vert]
                #)
                energy_per_vertex[vert] = current_energy
        #mean_energy = tf.reduce_mean(energy_per_vertex / 3.0)    
        #mean_energy = tf.convert_to_tensor(energy_per_vertex)
        #mean_energy = tf.reduce_mean(energy_per_vertex / 3.0) 
        mean_energy =  0
        return mean_energy

class curvatureLoss:
    def __init__(self, garment:Garment):
        self.faces = garment.faces
        self.vertices = garment.vertices
        self.normals = garment.normals
        self.dihedrals = garment.face_dihedral
        self.boundary = garment.boundary
        self.pi = tf.constant(math.pi, dtype=tf.float32)

    

    @tf.function
    def planar_angle(self, corner, pt1, pt2):
        vector1 = pt1-corner
        vector2 = pt2-corner
        dot_product = tf.reduce_sum(vector1 * vector2, axis=-1)
        norm_product = tf.norm(vector1, axis=-1) * tf.norm(vector2, axis=-1)
        cosine_angle = dot_product / norm_product
        angles = tf.acos(tf.clip_by_value(cosine_angle, -1.0, 1.0))
        return angles
        
    
    def __call__(self, vertices):
        vshape = vertices.shape

        #pre-computing
        f_normals = face_normals(self.vertices, self.faces)

        sparse_adjacency = compute_sparse_adjacency(self.vertices,self.faces)
        
        # _, face_to_face_adjacency, face_to_face_edges = faces_to_edges_and_adjacency(faces)
        energy = tf.expand_dims(tf.zeros((vshape[0]), dtype=tf.float32),1)
        for vert_idx in range(vshape[1]):
            if self.boundary[vert_idx]:
                print(f'the skipped boundary vert idx is: {vert_idx}')
                continue

            
            else:
                angle_sum_per_vertex = 0
                # first col stands for face idx
                # second col stands for vert idx in the traingular face 1 to 3
                vertex_adjacency = nonzero_slice_row(sparse_adjacency, vert_idx)
                for f_idx in range(vertex_adjacency.shape[0]):
                    face = self.faces[vertex_adjacency[f_idx, 0]]
                    i = vert_idx
                    local_index_i = vertex_adjacency[f_idx, 1]-1 # the idx are from 1 to 3 in sparse matrix
                    j = face[(local_index_i+1)%3]
                    k = face[(local_index_i+2)%3]
                    #compute the planar angle
                    angle = self.planar_angle(vertices[:,i], vertices[:,j], vertices[:,k])
                    angle_sum_per_vertex += angle
                # now go back to vertex
                K_vertex = 2.0*self.pi - angle_sum_per_vertex
                energy_vertex = tf.expand_dims(K_vertex*K_vertex, axis=1)
                energy = tf.concat([energy, energy_vertex], axis=-1)
        energy = tf.reduce_sum(energy, axis=1) # sum over vertices
        energy = tf.reduce_sum(energy, axis=0) # sum over batches
        return energy
        
'''      
class projDevLoss():
    def __init__(self, mesh, topo):
        self.delta_N = tf.zeros(topo.eN)
        mid_points = tf.zeros(shape=(topo.eN, 3), dtype=tf.float64)
        diags = tf.zeros(shape=(topo.eN, 2, 3))
        edge_normals = compute_edge_normals(mesh, topo)

        # edge_ids = tf.fill(dims=(topo.eN, 2, 3), value=-1)
        
        for eh in mesh.edges(): 
            edge_idx = eh.idx()
            heh0 = mesh.halfedge_handle(tf.bitwise.left_shift(edge_idx,1))
            if topo.vb[mesh.from_vertex_handle(heh0).idx()] and topo.vb[mesh.to_vertex_handle(heh0).idx()]:
                continue
            # edge_ids[edge_idx,0,0] = tf.bitwise.right_shift(heh0.idx(), 1)
            edge_idx_00 = tf.bitwise.right_shift(heh0.idx(), 1)
            mid_points[edge_idx] = compute_midpoint(mesh, heh0)
            
            heh1 = mesh.halfedge_handle(heh0.idx() ^ 1)
            # edge_ids[edge_idx,1,0] = tf.bitwise.right_shift(heh1.idx(),1)
            edge_idx_00_10 = tf.bitwise.right_shift(heh1.idx(),1)
            
            heh01 = mesh.next_halfedge_handle(heh0)
            edge_ids[edge_idx,0,1] = tf.bitwise.right_shift(heh01.idx(),1)
            
            heh02 = mesh.next_halfedge_handle(heh01)
            edge_ids[edge_idx,0,2] = tf.bitwise.right_shift(heh02.idx(),1)
            
            heh11 = mesh.next_halfedge_handle(heh1)
            edge_ids[edge_idx,1,1] = tf.bitwise.right_shift(heh11.idx(),1)
            
            heh12 = mesh.next_halfedge_handle(heh11)
            edge_ids[edge_idx,1,2] = tf.bitwise.right_shift(heh12.idx(),1)
        
        half_edges = []
        valid_edges = []
        mid_points = []

        for eh in mesh.edges(): 
            edge_idx = eh.idx()
            
            heh0 = mesh.halfedge_handle(tf.bitwise.left_shift(edge_idx, 1))
            mid_point_cur_edge = compute_midpoint(mesh, heh0)
            mid_points.append(mid_point_cur_edge)
            
            # Check if both vertices are boundary vertices
            if topo.vb[mesh.from_vertex_handle(heh0).idx()] and topo.vb[mesh.to_vertex_handle(heh0).idx()]:
                continue

            valid_edges.append(edge_idx)
            
            # First set of three half-edges related to heh0
            half_edges.append(heh0.idx())
            half_edges.append(mesh.next_halfedge_handle(heh0).idx())
            half_edges.append(mesh.next_halfedge_handle(mesh.next_halfedge_handle(heh0)).idx())
            
            # Second set of three half-edges related to heh1 (opposite half-edge of heh0)
            heh1 = mesh.halfedge_handle(heh0.idx() ^ 1)
            half_edges.append(heh1.idx())
            half_edges.append(mesh.next_halfedge_handle(heh1).idx())
            half_edges.append(mesh.next_halfedge_handle(mesh.next_halfedge_handle(heh1)).idx())

        # Convert list to TensorFlow tensor
        half_edge_tensor = tf.constant(half_edges, dtype=tf.int32)
        mid_points_tensor = tf.constant(mid_points, dtype=tf.float64)
        
        shifted_indices = tf.bitwise.right_shift(half_edge_tensor, 1)
        edge_ids = tf.reshape(shifted_indices, [-1, 2, 3])
        
        #============================================================#
        
        adj_edge_ids = tf.reshape(edge_ids[:, :, 1:], (-1, 4))
        quad_vertices = tf.gather(mid_points_tensor, adj_edge_ids)
        # quad vertices has shape (num_edges, 4, 3)
        edge_ids = tf.reshape(edge_ids, (-1, 6))
        N = tf.gather(edge_normals, edge_ids)
        #diags are associated to each edge
        diags = tf.stack([quad_vertices[:,2] - quad_vertices[:,0], quad_vertices[:,3] - quad_vertices[:,1]], axis=1)
        proj_plane_normals = tf.linalg.cross(diags[:,0], diags[:,1])
        
        areas = tf.linalg.norm(proj_plane_normals, axis=-1)
        areas_power = tf.math.pow(areas, 2/3)
        areas_power = tf.reshape(areas_power, (areas.shape[0], 1, 1))
        
        dN = tf.stack([N[:,1]-N[:,4], N[:,2]-N[:,5]], axis=1)
        Pr_dN = tf.stack([compute_projection(dN[:,0], proj_plane_normals),\
            compute_projection(dN[:,1], proj_plane_normals),\
            tf.zeros(shape=(len(valid_edges),3), dtype=tf.float64)])
        
        A = tf.stack([diags[:,0], diags[:,1], proj_plane_normals], axis=-1)
        B = Pr_dN
        
        try: 
            phi_tilde = tf.linalg.solve(A, B)
            eigenvalues, _ = tf.linalg.eigh(phi_tilde/areas_power)
            eigenvalues = eigenvalues[tf.abs(eigenvalues) > 0.000001]
            assert(eigenvalues.shape[1]==2),'less eigenvalues!'
            
            delta_N = eigenvalues[:,0] * eigenvalues[:1]
            dev_loss = tf.reduce_sum(delta_N)
            return dev_loss
        
        except tf.errors.InvalidArgumentError as e:
            print(e)
'''


class projDevLoss():
    def __init__(self, mesh, topo, garment):
        self.mesh = mesh
        self.topo = topo
        self.garment = garment
        self.edge_ids, self.valid_edges, self.adj_faces = self.get_edge_tensor(self.mesh, self.topo)
        self.init_curvature = self.get_init_curvature(self.garment.vertices)
    
    @tf.function
    def get_edge_tensor(self, mesh, topo):
        half_edges = []
        valid_edges = []
        adj_faces = []

        for eh in mesh.edges():
            edge_idx = eh.idx()
            
            heh0 = mesh.halfedge_handle(edge_idx<<1)
            topoint, frompoint = mesh.from_vertex_handle(heh0).idx(), mesh.to_vertex_handle(heh0).idx()
            # Check if both vertices are boundary vertices
            if topo.vb[topoint] and topo.vb[frompoint]:
                continue

            valid_edges.append([topoint, frompoint])
            
            # First set of three half-edges related to heh0
            half_edges.append(heh0.idx())
            half_edges.append(mesh.next_halfedge_handle(heh0).idx())
            half_edges.append(mesh.next_halfedge_handle(mesh.next_halfedge_handle(heh0)).idx())
            
            # Second set of three half-edges related to heh1 (opposite half-edge of heh0)
            heh1 = mesh.halfedge_handle(heh0.idx() ^ 1)
            half_edges.append(heh1.idx())
            half_edges.append(mesh.next_halfedge_handle(heh1).idx())
            half_edges.append(mesh.next_halfedge_handle(mesh.next_halfedge_handle(heh1)).idx())

            adj_faces.append([topo.h2f[heh0.idx()], topo.h2f[heh1.idx()]])
            
        # Convert list to TensorFlow tensor
        half_edge_tensor = tf.constant(half_edges, dtype=tf.int32)
        #mid_points_tensor = tf.constant(mid_points,tf.float64,(len(mid_points),3))
        shifted_indices = tf.bitwise.right_shift(half_edge_tensor, 1)
        edge_ids = tf.reshape(shifted_indices, [-1, 2, 3])
        valid_edges = np.array(valid_edges)
        adj_faces = np.array(adj_faces)
        # formulate midpoints as (num_edges, 3)
        # formulate edge_ids as (num_edges, 2, 3)
        return edge_ids, valid_edges, adj_faces

    @tf.function
    def get_midpoints(self, edges, vertices):
        # edges (num_valid_edges, 2)
        topoints, frompoints = tf.gather(vertices, edges[:,1], axis=0), tf.gather(vertices, edges[:,0], axis=0)
        return (topoints + frompoints)/2

    @tf.function
    def get_edge_normals(self, vertices):
        f_normals = face_normals_one_frame(vertices, self.garment.faces)
        # f_normals of (19, 6956, 3)
        # adj faces of (10332, 2)
        edge_normals = tf.gather(f_normals, self.adj_faces[:, 0], axis=0) + tf.gather(f_normals, self.adj_faces[:, 0], axis=0)
        print(f'edge normals shape {edge_normals.shape}')
        return edge_normals
    
    @tf.function
    def get_eigen_one_frame(self, vertices): 
        edge_ids, valid_edges = self.edge_ids, self.valid_edges
        # mid_points_tensor = self.mid_points
        mid_points_tensor = self.get_midpoints(valid_edges, vertices)
        edge_normals = self.get_edge_normals(vertices)
        
        #print(f'mid points shape {mid_points_tensor.shape}')
        #print(f'edge_ids shape {edge_ids.shape}')
        #print(f"edge normals {edge_normals.shape}")
        
        adj_edge_ids = tf.reshape(edge_ids[:, :, 1:], (-1, 4))
        quad_vertices = tf.gather(mid_points_tensor, adj_edge_ids, axis=0)
        # quad vertices has shape (num_edges, 4, 3)
        edge_ids = tf.reshape(edge_ids, (-1, 6))
        # edge normals (10332, 3)
        # gather_normals = [tf.gather(edge_normals[b], edge_ids, axis=0) for b in range(edge_normals.shape[0])]
        N = tf.gather(edge_normals, edge_ids, axis=0)
        # N = tf.stack(gather_normals)
        # diags are associated to each edge
        # quad (10332, 4, 3)
        diags = tf.stack([quad_vertices[:,2] - quad_vertices[:,0], quad_vertices[:,3] - quad_vertices[:,1]], axis=1)
        # diags (10332, 2, 3)
        proj_plane_normals = tf.linalg.cross(diags[:,0], diags[:,1])
        # proj_plane_normals (10332, 3)
        
        # areas = tf.linalg.norm(proj_plane_normals, axis=-1)
        # areas_power = tf.math.pow(areas, 2/3)
        
        # N (10332, 6, 3)
        dN = tf.stack([N[:,1]-N[:,4], N[:,2]-N[:,5]], axis=1)
        # dN (10332, 2, 3)
        Pr_dN = tf.stack([compute_projection(dN[:,0], proj_plane_normals),\
            compute_projection(dN[:,1], proj_plane_normals),\
            tf.zeros(shape=dN[:,0,:].shape, dtype=tf.float32)], axis=-1)
        
        A = tf.stack([diags[:,0], diags[:,1], proj_plane_normals], axis=-1)
        B = Pr_dN
        l = 1e-5 * tf.eye(3, batch_shape=A[:,0,0].shape)
        A = A+l
        #print(f'shape of A is {A.shape}')

        # phi_tilde = tf.linalg.solve(A, B) / tf.reshape(areas_power, [-1, 1, 1])
        phi_tilde = tf.linalg.solve(A, B)
        singulars = tf.linalg.svd(phi_tilde, compute_uv=False)
        print(singulars[..., :-1])
        print(f'shape of singulars is {singulars.shape}')
        singulars = singulars[..., :-1]

        #print(f'shape of phi is {phi_tilde.shape}')
        #eigenvalues, _ = tf.linalg.eig(phi_tilde)
        return singulars
    
    @tf.function
    def get_curvature_one_frame(self, eigenvalues):
        #threshold = 1e-4
        #non_zero_eigenvalues = tf.where(tf.abs(eigenvalues) < threshold, 1.0, eigenvalues)
        return tf.reduce_prod(eigenvalues, axis=1)

    def get_init_curvature(self, init_vertices):
        eigenvalues = self.get_eigen_one_frame(init_vertices)
        init_curvature = self.get_curvature_one_frame(eigenvalues)
        return init_curvature

    def __call__(self, vertices):
        eigenvalues_start = self.get_eigen_one_frame(vertices[0])
        eigenvalues_end = self.get_eigen_one_frame(vertices[-1])
        start_curvature = self.get_curvature_one_frame(eigenvalues_start)
        end_curvature = self.get_curvature_one_frame(eigenvalues_end)

        diff_per_edge = tf.abs(start_curvature - end_curvature)
        # there are a few (less than 50) NAN values on the boundaries
        diff_per_edge = tf.where(tf.math.is_nan(diff_per_edge), tf.zeros_like(diff_per_edge, dtype=tf.float32), diff_per_edge)
        diff_per_edge = tf.where(diff_per_edge > 100.0, tf.fill(diff_per_edge.shape, 100.0), diff_per_edge)
        loss = tf.reduce_sum(diff_per_edge)
        error = loss
        return loss, error


        
            



    
    
    
    
    
