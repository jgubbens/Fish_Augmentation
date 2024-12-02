from ObjectPlacement.sam2_generator import ImageSegmenter
from ObjectPlacement.brightness_modifier import BrightnessModifier
from ObjectPlacement.object_placement import ObjectPlacer
from DiffuseMix.diffusemix import Diffuse
from ElasticDeformation.elastic_deformation import ElasticDeformer
import os

# ARGS - WILL BE GOTTEN FROM CONFIG FILE IN THE FUTURE
source_coords = (0.8, 0.2)
adj_intensity = 0.3
num_generated = 5000
training_path = 'processed_data'
background_image = os.path.join('ObjectPlacement', 'equalized_background.png')

def main():
    # Brightness Calculator
    modifier = BrightnessModifier(source_coords, adj_intensity)
    brightness = modifier.find_brightness((0.8, 0.2), (0.6, 0.6))
    print(brightness)

    # Elastic Deformation
    deformer = ElasticDeformer(30, 5)
    deformer.run_transform('ElasticDeformation/original.png')
    deformer.transform_dir(training_path)
    
    # Run SAM2
    model_checkpoint = 'checkpoints/sam2.1_hiera_large.pt'
    model_config = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    segmenter = ImageSegmenter(model_checkpoint, model_config)
    images_folder = os.path.join(training_path, 'images')
    labels_folder = os.path.join(training_path, 'labels')
    #segmenter.process_all_files(images_folder, labels_folder)
    #segmenter.copy_labels_with_full_image_bbox(labels_folder, 'segmented/bbox_labels', images_folder)
    
    # Run DiffuseMix
    diffuse_output = 'result'
    #diffuse_train_data = os.path.join('DiffuseMix', 'training_sets', 'color_equalized')
    diffuse_train_data = 'segmented'
    prompts = 'Ukiyo-e,Snowy,Watercolor'
    diffuse_runner = Diffuse(diffuse_output, diffuse_train_data, prompts)
    #diffuse_runner.run_diffusemix()
    #diffuse_runner.resize_output('diffuse-output')

    # Run Object Placement
    segmented_dir = os.path.join('diffuse-output', 'images')
    label_dir = os.path.join('diffuse-output', 'labels')
    #random_placer = ObjectPlacer(background_image, num_generated, modifier, segmented_dir, label_dir)
    #random_placer.generate_augmented_images()
if __name__ == "__main__":
    main()
