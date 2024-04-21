import bpy
import os
import sys
import traceback
import math
ROOT_DIR = '/mnt/c/Users/JingchaoKong/NeuralClothSim'
BODY_DIR = os.path.join(ROOT_DIR, "body_models")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
DATA_DIR = os.path.join(ROOT_DIR, "data")
TXT_DIR = os.path.join(ROOT_DIR, "ncs", "dataset", "txt")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SMPL_DIR = os.path.join(ROOT_DIR, "smpl")
TMP_DIR = os.path.join(ROOT_DIR, "tmp")

def clean_scene():
    for object in bpy.data.objects:
        object.select_set(True)
    bpy.ops.object.delete()


def load_object(obj, name, loc, rot=(0, 0, 0)):
    bpy.ops.import_scene.obj(
        filepath=obj, split_mode="OFF", axis_forward="-Y", axis_up="Z"
    )
    bpy.ops.object.shade_smooth()
    assert len(bpy.context.selected_objects) == 1, "Multiple objects in one OBJ? " + obj
    object = bpy.context.selected_objects[0]
    object.name = name
    object.location = loc
    object.rotation_euler = rot
    return object


def select(object):
    if type(object) is str:
        object = bpy.data.objects[object]
    deselect()
    object.select_set(True)
    bpy.context.view_layer.objects.active = object
    return object


def deselect():
    for object in bpy.data.objects.values():
        object.select_set(False)


def mesh_cache_modifier(object, pc2_file):
    object = select(object)
    bpy.ops.object.modifier_add(type="MESH_CACHE")
    object.modifiers["MeshCache"].cache_format = "PC2"
    object.modifiers["MeshCache"].filepath = pc2_file


def createVertexGroups(ob, W):
    for j in range(W.shape[1]):
        vg_name = "bone" + str(j)
        createVertexGroup(ob, W[:, j], vg_name)


def createVertexGroup(ob, W, name):
    vg = ob.vertex_groups.new(name=name)
    for i, w in enumerate(W):
        vg.add([i], w, "ADD")



def main(loc=(0, 0, 0), rot=(0, 0, 0)):

    camera_data = bpy.data.cameras.new(name='MyCamera')
    camera_object = bpy.data.objects.new('MyCameraObject', camera_data)
    bpy.context.collection.objects.link(camera_object)
    camera_object.location = (0, 0, 7)
    camera_object.rotation_euler = (0, 0, math.radians(90))
    bpy.context.scene.camera = camera_object

    light_data = bpy.data.lights.new(name="MyPointLight", type='POINT')

    light_object = bpy.data.objects.new(name="MyPointLightObject", object_data=light_data)

    bpy.context.collection.objects.link(light_object)

    light_object.location = (5, -5, 9) 

    light_data.energy = 800  
    light_data.color = (1, 1, 1)  

    # Load OBJs
    body_obj = os.path.join(BODY_DIR, 'mannequin', "body.obj")
    body_pc2 = os.path.join(RESULTS_DIR, 'mixamo', "body.pc2")
    body = load_object(body_obj, name="body", loc=loc, rot=rot)
    mesh_cache_modifier(body, body_pc2)

    garment_obj = os.path.join(
        BODY_DIR, 'mannequin', 'tshirt' + ".obj"
    )
    garment_pc2 = os.path.join(RESULTS_DIR, 'mixamo', 'tshirt' + ".pc2")
    garment_unskinned_pc2 = os.path.join(
        RESULTS_DIR, 'mixamo', 'tshirt' + "_unskinned.pc2"
    )
    garment = load_object(garment_obj, 'tshirt', loc=loc, rot=rot)
    mesh_cache_modifier(garment, garment_pc2)
    garment.active_material.diffuse_color = (0.5, 0.5, 1.0, 1.0)

    loc = (loc[0] + 1, *loc[1:])
    garment_unskinned = load_object(
        garment_obj, 'tshirt' + "_unskinned", loc=loc, rot=rot
    )
    mesh_cache_modifier(garment_unskinned, garment_unskinned_pc2)
    garment_unskinned.active_material.diffuse_color = (0.5, 0.5, 1.0, 1.0)


if __name__ == "__main__":
    clean_scene()
    results_folder = os.path.join(ROOT_DIR, "results")
    main(loc=(0, 0, 0), rot=(0, 0, 0))