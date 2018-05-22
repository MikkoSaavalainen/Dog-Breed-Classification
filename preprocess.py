from skimage.transform import resize
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from lxml import etree
import os
import numpy as np

def load_images(input_dir, anno_dir, output_dir):
    
    images = []
    
    class_num = 0
    
    for class_folder_im, class_folder_anno in zip(os.listdir(input_dir), os.listdir(anno_dir)):
        
        path_im = input_dir + "/" + class_folder_im
        path_anno = anno_dir + "/" + class_folder_anno
        
        image_list = os.listdir(input_dir + "/" + class_folder_im)
        anno_list = os.listdir(anno_dir + "/" + class_folder_anno)
        for image, annotation in zip( image_list, anno_list ):
            
            image_dir = path_im + "/" + image
            im = io.imread(image_dir)
            
            annotation_dir = path_anno + "/" + annotation
            bndbox = read_annotations(annotation_dir)
            
            processed_image = process_image(im, [64, 64], bndbox, True)
            
            image_data = [class_num processed_image]
            images.append(image_data)
            
        class_num += 1
            
            
                    
def read_annotations(path):
    
    tree = etree.parse(path)
            
    ranges = []
    for box in tree.xpath("//bndbox"):
        for child in box.getchildren():
            ranges.append(child.text)
    
    return ranges
    
def process_image(image, size, bndbox, BW: bool):
    
    xMin = int( bndbox[0] )
    yMin = int( bndbox[1] )
    xMax = int( bndbox[2] )
    yMax = int( bndbox[3] )
    
    io.imshow(image)
    plt.show()
    
    im = image[xMin:xMax, yMin:yMax, :]
    im = resize(im, (size[0], size[1]))
    im = color.rgb2gray(im)
    
    io.imshow(im)
    plt.show()
    
    return im

def main():
    load_images("Images", "Annotation", "ProcessedData")

if __name__ == '__main__':
    main()
