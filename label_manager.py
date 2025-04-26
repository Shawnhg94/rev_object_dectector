import yaml
from object_entity import ObjectEntity

class LabelManager:
    def __init__(self):
        with open("object.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        self.obj_entity_map = {}
        self.class_names = []
        for ele in config['DRIVING_objects']:
            name = ele['Entity']
            id = ele['ID']
            colour = ele['Colour']
            entity = ObjectEntity(name, id=id, colour=colour)
            self.obj_entity_map.update({id: entity})
            self.class_names.append(name)
        
    
    def get_colour(self, id:int):
        return self.obj_entity_map[id].colour
    

    def get_num(self):
        return len(list(self.obj_entity_map.keys()))
    
    def get_ids(self):
        return list(self.obj_entity_map.keys())
    
    def get_names(self):
        return self.class_names
    
    def get_name(self, id:int):
        return self.class_names[id]
    
    def get_label(self, name:str):
        index = self.class_names.index(name)
        return index + 1
    

def parse_label(filepath:str):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)

    # return list(config.values())
    return config