from typing import Any
import tensorflow as tf
import numpy as np
from utils.mesh import vertex_normals, face_normals, compute_vertex_face_adjacency
from model.cloth import Garment

class HingePairLoss:
    def __init__(self, garment:Garment, weighted_by_tipangles:bool, normalize_by_pairnum:bool):
        self.edges = garment.edges
        self.faces = garment.faces
        self.face_normals = face_normals(self.faces)
        self.edge_lengths_true = garment.edge_lengths
        self.is_boundary = garment.is_boundary
        self.n_pairs = 0

        self.weighted_by_tipangles = weighted_by_tipangles
        self.normalized_by_pairnum = normalize_by_pairnum
    
    def handleNormalPair(self, n1, n2):
        if self.normalized_by_pairnum:
            self.n_pairs+=1
        
        normal1 = tf.gather(self.faceNormals, n1)
        normal2 = tf.gather(self.faceNormals, n2)

        squared_norm = tf.reduce_sum(tf.square(normal1 - normal2))

        if self.weighted_by_tipangles:
            angle1 = tf.gather_nd(self.angles, [n1])
            angle2 = tf.gather_nd(self.angles, [n2])
            return squared_norm * angle1 * angle2
        else:
            return squared_norm


    @tf.function
    def __call__(self, vertices):

        VF, VFi = compute_vertex_face_adjacency(self.faces)
        energy_per_vertex = np.zeros(vertices.shape[0]) 
        partition_indices = np.full((vertices.shape[0], 2), fill_value=-1)

        shape = vertices.shape
        # first try: flatten along the time axis
        vertices = tf.reshape(vertices, [shape[0], shape[1]*shape[2], shape[3]])

        for vert in vertices:
            if self.is_boundary[vert] and len(VF[vert])==3:
                 # there can be no hinge if the neighborhood has only size 3
                 energy_per_vertex[vert] = 0
                 continue
            
            current_energy = np.inf
            adjacent_faces = VF[vert]
            adjacent_facesi = VFi[vert]
            adjacent_faces_count = len(adjacent_faces)

            for edge1 in range(len(adjacent_faces)-2):
                for edge2 in range(edge1 + 2, len(adjacent_faces) - 1 \
                                   if edge1 == 0 else len(adjacent_faces)):


                    # Handle normal pairs from edge1 to edge2
                    edge1sum = 0
                    for n1 in range(edge1, edge2):
                        for n2 in range(n1 + 1, edge2):
                            edge1sum += self.handleNormalPair(n1, n2)

                    # Handle normal pairs from edge2 to the end, and then from the start to edge1
                    edge2sum = 0
                    for n1 in range(edge2, adjacent_faces_count):
                        for n2 in range(n1 + 1, adjacent_faces_count):
                            edge2sum += self.handleNormalPair(n1, n2)
                        for n2 in range(edge1):
                            edge2sum += self.handleNormalPair(n1, n2)

                    # Additional loop for handling pairs from the start to edge1
                    for n1 in range(edge1):
                        for n2 in range(n1 + 1, edge1):
                            edge2sum += self.handleNormalPair(n1, n2)
                    local_energy = 0.5*(edge1sum + edge2sum)

                    if self.normalized_by_pairnum:
                        local_energy/self.n_pairs
                    if local_energy < currentEnergy:
                        currentEnergy = local_energy
                        partition_indices[vert] = [edge1, edge2]
        
        return current_energy


                            


    