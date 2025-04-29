import os
import sys
from PIL import Image


old_path = './source_path'
new_path = './destination' 

for filename in os.listdir(old_path):
    #print(filename)
    old_image = Image.open(os.path.join(old_path,filename))
    print(old_image.size)
    # filename_, ext = os.path.splitext(filename)
    
    # if ext == '.jpg':
    #     pass
    # else:
    #     filename = filename_ + '.jpg'
        
    new_image = old_image.resize((224, 224))
    print(new_image.size)
    path =os.path.join(new_path,filename)
    #print(path)
    new_image.save(path)
    