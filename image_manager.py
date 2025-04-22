import os
import shutil
from PIL import ImageTk, Image


def numerical_sort(filename):
    """Extracts the numerical part of the filename for sorting."""
    return int(filename.split('.')[0])

class ImageManager:
    def __init__(self, path: str):
        print('Create ImageManger {}'.format(path))
        self.img_path = path
        self.imgs = list(sorted(os.listdir(path), key= numerical_sort))
        # print(self.imgs)
        img_path = os.path.join(path, self.imgs[0])
        img = Image.open(img_path)
        self.height = img.height
        self.width = img.width

        default_img = Image.new('RGB', (self.width, self.height), color='black')
        self.default_photo = ImageTk.PhotoImage(default_img)

    def get_photo(self, idx: int):
        img_path = os.path.join(self.img_path, self.imgs[idx])
        img = Image.open(img_path)
        photo = ImageTk.PhotoImage(img)
        return photo
    
    def get_num(self):
        return len(self.imgs)

    
    def get_img(self, idx: int):
        img_path = os.path.join(self.img_path, self.imgs[idx])
        return Image.open(img_path)

    def convert_photo(self, img: Image):
        return ImageTk.PhotoImage(img)
    
    # delete all previous files
    def clear_capture(self):
        for filename in os.listdir('capture'):
            file_path = os.path.join('capture', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))