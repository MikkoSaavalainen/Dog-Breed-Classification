from skimage.transform import resize
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from lxml import etree
import os
import numpy as np

def load_images(input_dir, anno_dir):
    
    images = []
    labels = []
    
    class_num = 0
    
    for class_folder_im, class_folder_anno in zip(os.listdir(input_dir), os.listdir(anno_dir)):
        
        path_im = input_dir + "/" + class_folder_im
        path_anno = anno_dir + "/" + class_folder_anno
        
        image_list = os.listdir(input_dir + "/" + class_folder_im)
        anno_list = os.listdir(anno_dir + "/" + class_folder_anno)
        print("Reading folder: ", class_num)
        for image, annotation in zip( image_list, anno_list ):
            
            image_dir = path_im + "/" + image
            im = io.imread(image_dir)
            
            annotation_dir = path_anno + "/" + annotation
            bndbox = read_annotations(annotation_dir)
            
            processed_image = process_image(im, [64, 64], bndbox, True)

            labels.append(class_num)
            images.append(processed_image)
            
#        if class_num > 80:
#            break
        class_num += 1
            
    return images, labels
                    
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
    
    im = image[yMin:yMax, xMin:xMax, :]
    im = resize(im, (size[0], size[1]))
    
    if BW:
        im = color.rgb2gray(im)
    
    return im

def save_data(target_dir, data, labels):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    label_name = f"{target_dir}/labels"
    data_name = f"{target_dir}/data"

    np.save(data_name, data)
    np.save(label_name, labels)

def main():
    data, labels = load_images("Images", "Annotation")
    save_data("Processed", data, labels)

if __name__ == '__main__':
    main()
