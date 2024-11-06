"""
DiffuseMix Paper: https://arxiv.org/pdf/2405.14881

DiffuseMix Github: https://github.com/khawar-islam/diffuseMix/tree/main

DiffuseMix main.py: https://github.com/khawar-islam/diffuseMix/blob/main/main.py
"""

#from augment.handler import ModelHandler
#from augment.utils import Utils
#from augment.diffuseMix import DiffuseMix
import sys
import os
from torchvision import datasets
import shutil

class Diffuse:
    def __init__(self, output_dir = 'DiffuseMix/result', training_folder = 'color_equalized', prompts = 'Ukiyo-e,Snowy,Watercolor'):
        # Define paths to directories
        self.prompts = prompts
        self.augment_dir = f'augment'
        self.train_dir = f'training_sets/{training_folder}'
        self.fractal_dir = f'fractal/deviantart'
        self.output_dir = output_dir

        # Add the directory to the Python path
        sys.path.append(self.augment_dir)

        # Path to the parent directory of 'augment'
        parent_dir = f'DiffuseMix'
        # Add the parent directory to the Python path
        sys.path.append(parent_dir)
        
    
    def duplicate_text_files(folder_path, save_image_dir, arr):
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.txt'):
                    original_file = os.path.join(root, filename)
                    for suffix in arr:
                        new_filename = f"{os.path.splitext(filename)[0]}_{suffix}_0.txt"
                        new_file = os.path.join(save_image_dir, new_filename)
                        shutil.copy2(original_file, new_file)

    def run_diffusemix(self):
        prompts_list = self.prompts.split(',')

        # Import the modules
        try:
            from augment.handler import ModelHandler
            from augment.utils import Utils
            from augment.diffuseMix import DiffuseMix
            print("Modules imported successfully.")
        except ImportError as e:
            print(f"ImportError: {e}")

        # Initialize the model
        model_id = "timbrooks/instruct-pix2pix"
        model_initialization = ModelHandler(model_id=model_id, device='cuda')

        # Load the original dataset
        train_dataset = datasets.ImageFolder(root=self.train_dir)
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

        # Load fractal images
        fractal_imgs = Utils.load_fractal_images(self.fractal_dir)

        # Create the augmented dataset
        augmented_train_dataset = DiffuseMix(
            original_dataset=train_dataset,
            fractal_imgs=fractal_imgs,
            num_images=1,
            guidance_scale=4,
            idx_to_class=idx_to_class,
            prompts=prompts_list,
            model_handler=model_initialization
        )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Save augmented images
        for idx, (image, label) in enumerate(augmented_train_dataset):
            image.save(f'{self.output_dir}/{idx}.png')

        print(f'Augmented images saved to {self.output_dir}')

        arr = self.prompts.split(',')
        print('Duplicating image labels, using suffixes ' + str(arr))
        folder_path = f'{self.train_dir}'
        save_image_dir = f'{self.output_dir}/labels'
        os.makedirs(save_image_dir, exist_ok=True)
        self.duplicate_text_files(f'{folder_path}/train/labels', save_image_dir, arr)
        self.duplicate_text_files(f'{folder_path}/valid/labels', save_image_dir, arr)
        self.duplicate_text_files(f'{folder_path}/test/labels', save_image_dir, arr)