from mesh_topo import MeshTopo
import openmesh as om
import tensorflow as tf
from model.cloth import Garment
import numpy as np
import math
import matplotlib.pyplot as plt

def compute_edge_normals(mesh, topo: MeshTopo):
    ea, eb = np.zeros((topo.fN, 3), dtype=np.float32), np.zeros((topo.fN, 3), dtype=np.float32)
    for i in range(topo.fN):
        ea[i] = mesh.point(mesh.vertex_handle(topo.f2v[i][1])) - mesh.point(mesh.vertex_handle(topo.f2v[i][0]))
        eb[i] = mesh.point(mesh.vertex_handle(topo.f2v[i][2])) - mesh.point(mesh.vertex_handle(topo.f2v[i][0]))
    fn = np.cross(ea, eb, axis=-1).astype(np.float32)
    en = np.zeros((topo.eN,3), dtype=np.float32)
    for i in range(topo.hN):
        if topo.h2f[i] < 0:
            continue
        en[i>>1] += fn[topo.h2f[i]]

    #for j in range(mesh_topo.eN):
    #    norm = np.linalg.norm(en, axis=1, keepdims=True)
    #    en = en/norm
    return en

#def compute_midpoint(mesh, heh):
#    topointh, frompointh = mesh.to_vertex_handle(heh), mesh.from_vertex_handle(heh)
#    topoint, frompoint = np.array(mesh.point(topointh)), np.array(mesh.point(frompointh))
#    return (topoint + frompoint) / 2.0

def compute_midpoint(mesh, heh, vertices:tf.Tensor):
    topointh, frompointh = mesh.to_vertex_handle(heh), mesh.from_vertex_handle(heh)
    topoint_idx, frompoint_idx = topointh.idx(), frompointh.idx()
    topoint, frompoint = tf.gather(vertices, topoint_idx, axis=1), tf.gather(vertices, frompoint_idx, axis=1)
    return (topoint + frompoint)/2 # (19, 3)

"""
def compute_projection(vec, normals):
    # vec has shape (num_edges, 3)
    # normal has shape (num_edges, 3)
    proj_onto_normal = np.dot(vec, normals) / np.dot(normals, normals) * normals
    proj_onto_plane = vec - proj_onto_normal
    return proj_onto_plane
"""

def compute_projection(vecs, normals):
    proj_onto_normal = tf.reduce_sum(vecs*normals, axis=-1, keepdims=True)
    proj_onto_plane = vecs - proj_onto_normal*normals
    return proj_onto_plane

'''
def compute_Gauss_curvature_static(mesh: om.TriMesh, mesh_topo:MeshTopo):
    delta_N = np.zeros(topo.eN, dtype = float)
    #for fh0 in mesh.faces():
        #for heh in mesh.fh(fh0):
            #print(f'heh has from vertex {mesh.from_vertex_handle(heh).idx()} and to vertex {mesh.to_vertex_handle(heh).idx()}')
            #print(f'position: {mesh.point(mesh.from_vertex_handle(heh))} and {mesh.point(mesh.to_vertex_handle(heh))}')
    edge_normals = compute_edge_normals(mesh, mesh_topo)
    for eh in mesh.edges():
        #print(f"edge index is {eh.idx()}")
        heh0 = mesh.halfedge_handle(eh.idx() << 1)

        #print(f'from {mesh.from_vertex_handle(heh0).idx()} to {mesh.to_vertex_handle(heh0).idx()}') 
        if topo.vb[mesh.from_vertex_handle(heh0).idx()] and topo.vb[mesh.to_vertex_handle(heh0).idx()]:
            continue

        heh1 = mesh.halfedge_handle(heh0.idx() ^ 1)
        # midpoint = compute_midpoint(heh0)
        # visit the midpoints of each edge in the direction of these half edges

        heh01 = mesh.next_halfedge_handle(heh0)
        N01 = edge_normals[heh01.idx() >> 1]
        heh02 = mesh.next_halfedge_handle(heh01)
        N02 = edge_normals[heh02.idx() >> 1]
        heh11 = mesh.next_halfedge_handle(heh1)
        N11 = edge_normals[heh11.idx() >> 1]
        heh12 = mesh.next_halfedge_handle(heh11)
        N12 = edge_normals[heh12.idx() >> 1]
        assert(mesh.next_halfedge_handle(heh02)==heh0),'goes outside the inner triangle!'
        assert(mesh.next_halfedge_handle(heh12)==heh1),'goes outside the inner triangle!'
        midpoints = np.vstack((compute_midpoint(heh01), compute_midpoint(heh02),
                                compute_midpoint(heh11), compute_midpoint(heh12)))

        diag0111 = midpoints[0] - midpoints[2]
        diag0212 = midpoints[1] - midpoints[3]
        area = np.linalg.norm(np.cross(diag0111, diag0212)) * 0.5
        # print(f'Area is {area}')
        diags_normed = np.vstack((diag0111, diag0212)) 
        # midpoints_normed = midpoints/area
        proj_plane_normal = np.cross(diags_normed[0], diags_normed[1])

        dN0111 = (N01 - N11)
        dN0212 = (N02 - N12)

        Pr_dN0111 = compute_projection(dN0111, proj_plane_normal)
        Pr_dN0212 = compute_projection(dN0212, proj_plane_normal)

        a = np.vstack((diags_normed[0], diags_normed[1], proj_plane_normal))
        b = np.vstack((Pr_dN0111, Pr_dN0212, np.zeros(3)))

        phi_tilde = np.linalg.solve(a, b)
        
        # _,S,_ = np.linalg.svd(phi_tilde)
        eigenvalues, eigenvectors = np.linalg.eig(phi_tilde/area**(2/3))
        #eigenvalues, eigenvectors = np.linalg.eig(phi_tilde)
        eigenvalues = eigenvalues[np.abs(eigenvalues) > 0.0000001]
        assert(eigenvalues.shape[0] == 2),'less eigenvalues!'
        #eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-10]
        #assert(eigenvalues.shape[0] == 2), 'shape less than 2'

        delta_N[eh.idx()] = (eigenvalues[0] * eigenvalues[1])
        
        
    return delta_N


def curvature_to_color(curvature, min_curvature, max_curvature):
    # Normalize the curvature value
    normalized_curvature = (curvature - min_curvature) / (max_curvature - min_curvature)
    normalized_curvature = np.clip(normalized_curvature, 0, 1)
    
    red = normalized_curvature
    blue = 1-normalized_curvature
    alpha = 1

    # Create the RGBA color array
    color = np.array([red, 0, blue, alpha]) * 255
    # Round to nearest integer and convert to integer type
    color = np.rint(color).astype(int)
    return color
'''
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def linear_rescale(values, new_min=0, new_max=1):
    """Rescale an array linearly to the new_min and new_max."""
    old_min, old_max = np.min(values), np.max(values)
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def curvature_to_color(vertex_curvature):
    # Normalize curvature without standard deviation to avoid clipping issues
    normalized_curvature = linear_rescale(vertex_curvature)
    
    # Mapping colors: red for high, blue for low, green for mid-values
    red = np.clip(2 * (normalized_curvature - 0.5), 0, 1)
    blue = np.clip(2 * (0.5 - normalized_curvature), 0, 1)
    
    # Adjust green to be high for values near the middle of the range
    # This is a simple approach; adjust the factor as needed for your data's range
    green = 1 - np.abs(normalized_curvature - 0.5) * 2
    
    alpha = np.ones_like(vertex_curvature)  # Full opacity

    # Stack and transpose to align colors along the last axis
    color = np.stack((red, green, blue, alpha), axis=-1)
    
    return color
'''
    
'''
def curvature_to_color(vertex_curvature):
    normalized_curvature = (vertex_curvature - np.min(vertex_curvature)) / (np.max(vertex_curvature)-np.min(vertex_curvature))
    
    plt.hist(normalized_curvature, bins=100)
    plt.title("Distribution of normalized Curvature")
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.show()   
    
    red = np.clip(normalized_curvature*2.0, 0,1)
    blue = np.clip(-normalized_curvature*2.0, 0, 1)
    green = np.clip((1-np.abs(normalized_curvature))*2.0, 0,1)
    alpha = np.ones_like(vertex_curvature)


    color = np.stack((red, green, blue, alpha))
    return np.transpose(color)
         
def write_ply_with_colors(filename, vertices, faces, colors):
    """
    Write a PLY file from vertex positions, faces, and vertex colors.

    :param filename: Output filename
    :param vertices: List of vertex positions [(x, y, z), ...]
    :param faces: List of faces [(v1, v2, v3), ...], assuming triangles
    :param colors: List of vertex colors [(r, g, b, a), ...]
    """
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        
        # Vertex list
        for vertexh, color in zip(vertices, colors):
            vertex=mesh.point(vertexh)
            f.write(f"{' '.join(map(str, vertex))} {' '.join(map(str, color))}\n")
        
        # Face list
        for faceh in faces:
            face = []
            for vh in mesh.fv(faceh):
                face.append(vh.idx())
            # Assuming faces are tuples of vertex indices
            f.write(f"3 {' '.join(map(str, face))}\n")
    

# Test
#CLOTH_MODEL = '/mnt/c/Users/JingchaoKong/NeuralClothSim/body_models/smpl_female_neutral/pants.obj'
CLOTH_MODEL = '/mnt/c/Users/JingchaoKong/NeuralClothSim/rest.obj'
#CLOTH_MODEL = '/mnt/c/Users/JingchaoKong/NeuralClothSim/simple.obj'
garment = Garment(CLOTH_MODEL)

mesh = om.read_trimesh(CLOTH_MODEL)
topo = MeshTopo(mesh)




# edge_normals = compute_edge_normals(topo, garment)
Gauss_curvature = compute_Gauss_curvature_static(mesh, topo)
Gauss_curvature = np.clip(Gauss_curvature, -500, 500)

# Get min and max curvature for normalization
min_curvature = np.min(Gauss_curvature)
max_curvature = np.max(Gauss_curvature)

v_Gaussian_curvature = np.zeros(topo.vN, dtype=float)
for vh in mesh.vertices():
    v_color = np.array([0.0, 0.0, 0.0, 0.0]) # Assuming RGBA color, initialize to black or zero
    num_adj = 0
    
    #for heh in mesh.vih(vh):
    #    num_adj += 1
    #    curv = Gauss_curvature[heh.idx() >> 1] # Getting the corresponding edge's curvature
    #    color = curvature_to_color(curv, min_curvature, max_curvature)
    #    v_color += np.array(color) # Ensure color is added as an array
    #v_color /= num_adj # Average the color
    for heh in mesh.vih(vh):
        num_adj+=1
        curv = Gauss_curvature[heh.idx() >> 1]
        v_Gaussian_curvature[mesh.to_vertex_handle(heh).idx()] += curv


plt.hist(v_Gaussian_curvature, bins=100)
plt.title("Distribution of Gauss Curvature")
plt.xlabel('value')
plt.ylabel('frequency')
plt.grid(True)
plt.show()    
    
    #mesh.vertex_colors()[vh.idx()] = v_color
vertex_colors = curvature_to_color(v_Gaussian_curvature)
for vh, color in zip(mesh.vertices(), vertex_colors):
    color_uint8 = (color*255).astype(np.uint8)
    mesh.set_color(vh, color_uint8)


# Save the mesh with vertex colors as PLY
# om.write_mesh("colored_mesh.ply", mesh)

write_ply_with_colors('dress.ply', mesh.vertices(), mesh.faces(), mesh.vertex_colors())
print()
'''
