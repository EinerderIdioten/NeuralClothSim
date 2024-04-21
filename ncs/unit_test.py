# from ncs.utils.mesh import faces_to_edges_and_adjacency
import numpy as np

import tensorflow as tf
from model.cloth import Garment
from utils.mesh import vertex_normals, face_normals, compute_vertex_face_adjacency, is_boundary, nonzero_slice_row
from model.cloth import Garment
import math

def is_boundary(faces):
    '''
    # A vertex is on the boundary if it is part of an edge that is shared by only one face.
    edges, _, faces_to_edges_and_adjacency_edge = faces_to_edges_and_adjacency(faces)
    
    # Create a mapping from edges to their number of faces sharing the edge
    edge_count = {}
    for edge in edges:
        edge = tuple(edge)
        if edge in edge_count:
            edge_count[edge]+=1
        else:
            edge_count[edge]=1
    num_vertices = np.max(faces) + 1
    boundary = [False] * num_vertices
    for edge, count in edge_count.items():
        if count == 1:  # Edge is part of only one face, hence a boundary edge
            boundary[edge[0]] = True
            boundary[edge[1]] = True
    '''
    # return boundary

def faces_to_edges_and_adjacency(faces):
    edges = dict()
    boundary_vertices = [False]*(np.max(faces)+1)
    for fidx, face in enumerate(faces):
        # i goes from 0 to 2
        # v is the current vertex
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
            edges[edge] += [fidx]
    face_adjacency = []
    face_adjacency_edges = []
    for edge, face_list in edges.items():
        if len(face_list) == 1:
            boundary_vertices[edge[0]], boundary_vertices[edge[1]] = True, True
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                face_adjacency += [[face_list[i], face_list[j]]]
                face_adjacency_edges += [edge]
    
    edges = np.array([list(edge) for edge in edges.keys()], np.int32)
    face_adjacency = np.array(face_adjacency, np.int32)
    face_adjacency_edges = np.array(face_adjacency_edges, np.int32)
    return edges, face_adjacency, face_adjacency_edges, boundary_vertices




class HingePairLoss:
    def __init__(self, faces):
        self.faces = faces
        

    def handleNormalPair(self, face_normals, n1, n2):
        normal1 = face_normals[n1]
        normal2 = face_normals[n2]

        squared_norm = np.sum(np.square(normal1 - normal2))
        print(squared_norm)
        return squared_norm
    
    def __call__(self, vertices_with_time_steps):

        vshape = vertices_with_time_steps.shape
        time_steps, num_vertices = vshape[1], vshape[2]

        #energy_per_vertex= tf.zeros([num_vertices], dtype=tf.float32)
        #partition_indices = tf.fill([num_vertices, 2], -1)
        
        energy_per_vertex = np.zeros(vshape)
        mesh_face_normals_alltime = face_normals(vertices_with_time_steps, self.faces)

        for t in tf.range(time_steps):
            mesh_face_normals = mesh_face_normals_alltime[:,t,...]
            
            VF, VFi = compute_vertex_face_adjacency(self.faces)
            #VF, VFi = tf.ragged.constant(VF), tf.ragged.constant(VFi)

            for vert in range(num_vertices):
                #if self.is_boundary[vert] and len(VF[vert])==3:
                #    # there can be no hinge if the neighborhood has only size 3
                #    energy_per_vertex[vert] = 0
                #    continue
                
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

                        if local_energy < current_energy:
                            current_energy = local_energy
                        print(current_energy)
          
        return current_energy
    
def compute_sparse_adjacency_boolean(vertices:tf.Tensor, faces:tf.Tensor)->tf.SparseTensor:
    num_vertices = vertices.shape[0]  
    num_faces = faces.shape[0]

    # Prepare indices for SparseTensor
    face_indices = tf.cast(tf.repeat(tf.range(num_faces), repeats=3), dtype=tf.int64)  # Repeat each face index 3 times
    vertex_indices = tf.cast(tf.reshape(faces, [-1]), dtype=tf.int64)  # Flatten face vertex indices
    sparse_indices = tf.cast(tf.stack([vertex_indices, face_indices], axis=1), dtype=tf.int64) # Combine as sparse tensor indices

    # Values are 1s, indicating adjacency
    values = tf.ones([tf.shape(sparse_indices)[0]], dtype=tf.int32)

    # Create the SparseTensor
    vertex_face_adjacency = tf.SparseTensor(indices=sparse_indices,
                                            values=values,
                                            dense_shape=[num_vertices, num_faces])

    return vertex_face_adjacency

def compute_sparse_adjacency(vertices:tf.Tensor, faces:tf.Tensor)->tf.SparseTensor:
    num_vertices = vertices.shape[0]  
    num_faces = faces.shape[0]

    # Prepare indices for SparseTensor
    face_indices = tf.cast(tf.repeat(tf.range(num_faces), repeats=3), dtype=tf.int64)  # Repeat each face index 3 times
    vertex_indices = tf.cast(tf.reshape(faces, [-1]), dtype=tf.int64)  # Flatten face vertex indices
    sparse_indices = tf.cast(tf.stack([vertex_indices, face_indices], axis=1), dtype=tf.int64) # Combine as sparse tensor indices

    values_pattern = tf.constant([1,2,3], dtype=tf.int64)
    values = tf.tile(values_pattern, [num_faces])
    # Create the SparseTensor
    vertex_face_adjacency = tf.SparseTensor(indices=sparse_indices,
                                            values=values,
                                            dense_shape=[num_vertices, num_faces])

    return vertex_face_adjacency

def nonzero_slice_row_idx(sparse:tf.SparseTensor, row):
    indices = sparse.indices
    values = sparse.values
    userindices = tf.where(tf.equal(indices[:,0], row))
    found_idx = tf.gather(indices, userindices)[:, 0, 1]
    res = tf.cast(found_idx, tf.int64)
    return res

def nonzero_slice_row(sparse:tf.SparseTensor, row):
    indices = sparse.indices
    values = sparse.values
    userindices = tf.where(tf.equal(indices[:,0], row))
    found_idx = tf.gather(indices, userindices)[:, 0, 1]
    found_vals = tf.gather(values, userindices)[:,0:1]

    res = tf.concat([tf.expand_dims(found_idx, -1), found_vals], axis = 1)
    return res

class curvatureLoss:
    def __init__(self, garment:Garment):
        self.faces = garment.faces
        self.vertices = garment.vertices
        self.normals = garment.normals
        self.dihedrals = garment.face_dihedral
        self.pi = tf.constant(math.pi, dtype=tf.float32)

    # @tf.function
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
        energy = tf.zeros((vshape[0], vshape[1], vshape[2]*vshape[3]), dtype=tf.float32)

        #pre-computing
        f_normals = face_normals(self.vertices, self.faces)
        boundary = is_boundary(self.faces)

        sparse_adjacency = compute_sparse_adjacency(self.vertices,self.faces)
        
        # _, face_to_face_adjacency, face_to_face_edges = faces_to_edges_and_adjacency(faces)
        
        for timestep in range(vshape[1]):
            for vert_idx in range(vshape[2]):
                angle_sum_per_time_vertex = 0
                #if is_boundary[vert_idx]:
                #    energy[vert_idx] = tf.constant(0.0, dtype=tf.float32)

                #    continue

                # first col stands for face idx
                # second col stands for vert idx in the traingular face 1 to 3
                vertex_adjacency = nonzero_slice_row(sparse_adjacency, vert_idx)
                for f in range(vertex_adjacency.shape[0]):
                    i = vert_idx
                    # get two other indices of the current triangle
                    j = faces[vertex_adjacency[f,0], (vertex_adjacency[f,1]+1)%3]
                    k = faces[vertex_adjacency[f,0], (vertex_adjacency[f,1]+2)%3]
                    #compute the planar angle
                    angle = self.planar_angle(vertices[:,timestep,i], vertices[:,timestep,j], vertices[:,timestep,k])
                    angle_sum_per_time_vertex += angle
                # now go back to vertex
                K_vertex = 2.0*self.pi - angle_sum_per_time_vertex
                energy_vertex = K_vertex*K_vertex
            print()
        
            
                


                    


                



'''
vertex = np.array([
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [0,0,1]
])


faces = np.array([
    [0,1,2],
    [0,2,3]
])

'''
garment_file_path = '/mnt/c/Users/JingchaoKong/NeuralClothSim/body_models/mannequin/tshirt.obj'

garment = Garment(garment_file_path)

vertices = garment.vertices
faces = garment.faces

adj_matrix = compute_sparse_adjacency(vertices, faces)
nonzero_row = nonzero_slice_row(adj_matrix, 3000)

curvature_loss = curvatureLoss(garment)
expanded_tensor = tf.expand_dims(tf.expand_dims(garment.vertices, 0),0)
batch_vertices = tf.broadcast_to(expanded_tensor, (4,3,3553,3))


loss = curvature_loss(batch_vertices)
boundary = is_boundary(faces)

print()