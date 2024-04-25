import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

from utils.tensor import tf_shape


def triangulate(faces):
    triangles = np.int32(
        [triangle for polygon in faces for triangle in _triangulate_recursive(polygon)]
    )
    return triangles


def _triangulate_recursive(face):
    if len(face) == 3:
        return [face]
    else:
        return [face[:3]] + _triangulate_recursive([face[0], *face[2:]])


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

def compute_vertex_face_adjacency(faces):
    # Initialize lists of VF and VFi
    vertex_face_adj = [[] for _ in range(np.max(faces) + 1)]
    vertex_local_indices = [[] for _ in range(np.max(faces) + 1)]

    # Iterate through each face and its vertices
    for fidx, face in enumerate(faces):
        for local_idx, vertex in enumerate(face):
            vertex_face_adj[vertex].append(fidx) 
            vertex_local_indices[vertex].append(local_idx)
    return vertex_face_adj, vertex_local_indices

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

def is_boundary(faces):
    # A vertex is on the boundary if it is part of an edge that is shared by only one face.
    edges, _, _,_ = faces_to_edges_and_adjacency(faces)
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
    
    return boundary

def laplacian_matrix(faces):
    G = {}
    for face in faces:
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            if v not in G:
                G[v] = {}
            if nv not in G:
                G[nv] = {}
            G[v][nv] = 1
            G[nv][v] = 1
    return graph_laplacian(G)


def graph_laplacian(graph):
    row, col, data = [], [], []
    for v in graph:
        n = len(graph[v])
        row += [v] * n
        col += [u for u in graph[v]]
        data += [1.0 / n] * n
    return coo_matrix((data, (row, col)), shape=[len(graph)] * 2)


@tf.function
def face_normals(vertices, faces, normalized=False):
    # (19, 3553, 3)
    input_shape = vertices.get_shape()
    vertices = tf.reshape(vertices, (-1, *input_shape[-2:]))
    v01 = tf.gather(vertices, faces[:, 1], axis=1) - tf.gather(
        vertices, faces[:, 0], axis=1
    )
    v12 = tf.gather(vertices, faces[:, 2], axis=1) - tf.gather(
        vertices, faces[:, 1], axis=1
    )
    normals = tf.linalg.cross(v01, v12)
    if normalized:
        normals /= tf.norm(normals, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    normals = tf.reshape(normals, (*input_shape[:-2], -1, 3))
    return normals

@tf.function
def face_normals_one_frame(vertices, faces, normalized=False):
    # (3553, 3)
    input_shape = vertices.get_shape()
    # vertices = tf.reshape(vertices, (-1, *input_shape[-2:]))
    v01 = tf.gather(vertices, faces[:, 1], axis=0) - tf.gather(
        vertices, faces[:, 0], axis=0
    )
    v12 = tf.gather(vertices, faces[:, 2], axis=0) - tf.gather(
        vertices, faces[:, 1], axis=0
    )
    normals = tf.linalg.cross(v01, v12)
    if normalized:
        normals /= tf.norm(normals, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    #normals = tf.reshape(normals, (*input_shape[:-2], -1, 3))
    print(f'face normals shape {normals.shape}')
    return normals

@tf.function
def vertex_normals(vertices, faces):
    input_shape = vertices.get_shape()
    batch_size = tf.reduce_prod(input_shape[:-2] or [1])
    vertices = tf.reshape(vertices, (-1, *input_shape[-2:]))
    # Compute face normals
    mesh_normals = face_normals(vertices, faces, normalized=False)
    # Scatter face normals
    faces_batched = tf.stack(
        (
            tf.tile(tf.range(batch_size)[:, None, None], [1, *faces.get_shape()]),
            tf.tile(faces[None], [batch_size, 1, 1]),
        ),
        axis=-1,
    )
    mesh_normals = tf.tile(mesh_normals[:, :, None], [1, 1, 3, 1])
    vertex_normals = tf.zeros((batch_size, *input_shape[-2:]), tf.float32)
    vertex_normals = tf.tensor_scatter_nd_add(
        vertex_normals, faces_batched, mesh_normals
    )
    vertex_normals /= (
        tf.norm(vertex_normals, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    )
    # Reshape back to input shape
    vertex_normals = tf.reshape(vertex_normals, input_shape)
    return vertex_normals


def edge_lengths(vertices, edges):
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)


def dihedral_angle_adjacent_faces(normals, adjacency):
    normals0 = normals[adjacency[:, 0]]
    normals1 = normals[adjacency[:, 1]]
    cos = np.einsum("ab,ab->a", normals0, normals1)
    sin = np.linalg.norm(np.cross(normals0, normals1), axis=-1)
    return np.arctan2(sin, cos)


def vertex_area(vertices, faces):
    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v12 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
    face_areas = np.linalg.norm(np.cross(v01, v12), axis=-1)
    vertex_areas = np.zeros((vertices.shape[0],), np.float32)
    for i, face in enumerate(faces):
        vertex_areas[face] += face_areas[i]
    vertex_areas *= 1 / 6
    total_area = vertex_areas.sum()
    return vertex_areas, face_areas, total_area


@tf.function
def lbs(vertices, matrices, blend_weights=None):
    matrices = tf.reshape(matrices, (*tf_shape(matrices)[:-2], 3 * 4))
    if blend_weights is not None:
        matrices = blend_weights @ matrices
    matrices = tf.reshape(matrices, (*tf_shape(matrices)[:-1], 3, 4))
    rotations, translations = tf.split(matrices, [3, 1], axis=-1)
    vertices = rotations @ vertices[..., None]
    vertices += translations
    return vertices[..., 0]
