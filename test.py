import os
import xml.etree.ElementTree as ET

def parse_xml_file(file_path,file_name):
    tree = ET.parse(file_path+"/"+file_name)
    root = tree.getroot()
    
    scene_dict = set()  # Use a set to prevent duplicates
    object_dict = {}
    
    for obj in root.findall('obj'):
        obj_id_elem = obj.find('obj_id')
        if obj_id_elem is not None and obj_id_elem.text is not None:
            obj_id = int(obj_id_elem.text)
        else:
            continue
        
        obj_name_elem = obj.find('obj_name')
        obj_name = obj_name_elem.text if obj_name_elem is not None else 'Unknown'
        
        obj_path_elem = obj.find('obj_path')
        obj_path = obj_path_elem.text if obj_path_elem is not None else 'Unknown'
        
        # Add to scene dictionary
        scene_dict.add(obj_id)  # Use add method for a set
        
        # Add to object dictionary
        object_dict[obj_id] = {
            'obj_name': obj_name,
            'obj_path': obj_path
        }
    
    return list(scene_dict), object_dict

# def process_directory(directory_path):
#     all_scenes = set()  # Use a set directly to avoid duplicates
#     all_objects = {}
    
#     for scene_id_loop in range(10):
#         scenes_list = os.listdir(directory_path)
#         print(scenes_list)
#         scene_id=f'scene_{scene_id_loop:04d}'
#         print(scene_id)
#         for scene_id in scenes_list:
#             file_path = os.path.join(directory_path, scene_id, "kinect", "annotations")
#             for file in os.listdir(file_path):
#                 if file.endswith('.xml'):
#                     scene_dict, object_dict = parse_xml_file(file_path, file)
#                     all_scenes.add(scene_id)  # Add scene_id directly to the set
#                     all_objects.update(object_dict)
    
#     return list(all_scenes), all_objects  # Convert set back to list if necessary for further processing

def process_directory(directory_path):
    all_scenes = set()  # Use a set directly to avoid duplicates
    all_objects = {}
    
    scenes_list = os.listdir(directory_path)
    # Filter scenes_list to include only the first ten scenes
    filtered_scenes = [scene for scene in scenes_list if scene.startswith('scene_') and int(scene.split('_')[1]) < 10]
    
    for scene_id in filtered_scenes:
        file_path = os.path.join(directory_path, scene_id, "kinect", "annotations")
        for file in os.listdir(file_path):
            if file.endswith('.xml'):
                scene_dict, object_dict = parse_xml_file(file_path, file)
                all_scenes.update(scene_dict)  # Correct: update set with elements from the list
                all_objects.update(object_dict)
    
    return list(all_scenes), all_objects  # Convert set back to list if necessary for further processing

# Example usage
directory_path = '../../../ssd_scratch/mohit.g/GraspNet/scenes'
scenes, objects = process_directory(directory_path)

print("Scenes with Object IDs:", scenes)
print("Objects with Names and Paths:", objects)