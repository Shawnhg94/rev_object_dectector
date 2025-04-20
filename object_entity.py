
class ObjectEntity:

    def __init__(self, name: str, id: int, colour: list):
        self.name = name
        self.id = id
        self.colour = colour

    def __str__(self):
        return "Name: {}, ID: {}, Colour: {}".format(self.name, self.id, self.colour)


    