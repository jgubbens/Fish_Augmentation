'''import cv2
import numpy as np
from PIL import Image
import os
import random

class ObjectPlacer:
    def __init__(self, background_path, num_augmented_images, brightness_augment):
        self.num_augmented_images = num_augmented_images
        self.background_path = background_path
        self.brightness_augment = brightness_augment
        self.segmented_fish_dir = os.path.join('segmented', 'images')
        self.labels_dir = os.path.join('segmented', 'labels')
        self.output_directory = 'augmented_seg'

        # Load background image
        self.background_bgr = cv2.imread(self.background_path)
        if self.background_bgr is None:
            raise FileNotFoundError(f"Background image not found at {self.background_path}")

        # Create output directory
        os.makedirs(self.output_directory, exist_ok=True)

        # Load fish images
        self.fish_images = os.listdir(self.segmented_fish_dir)
        if not self.fish_images:
            raise FileNotFoundError(f"No fish images found in directory {self.segmented_fish_dir}")

    def load_fish_label(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        fish_data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            fish_data.append((class_id, x_center, y_center, width, height))
        
        return fish_data

    def does_overlap(self, existing_boxes, new_box):
        new_x1, new_y1, new_x2, new_y2 = new_box
        for (x1, y1, x2, y2) in existing_boxes:
            if (new_x1 < x2 and new_x2 > x1 and
                    new_y1 < y2 and new_y2 > y1):
                return True
        return False

    def average_brightness(self, image):
        return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def adjust_brightness(self, image, target_brightness, max_adjustment=0.2):
        current_brightness = self.average_brightness(image)
        if current_brightness == 0:
            return image  # Avoid division by zero
        
        # Calculate the adjustment
        brightness_difference = target_brightness - current_brightness
        adjustment = max_adjustment * brightness_difference  # Limit adjustment to a percentage

        # Create a new adjusted image
        adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=adjustment)
        return adjusted_image

    def overlay_fish(self, background_img, fish_img_path, fish_labels, occupied_boxes):
        fish_img = Image.open(fish_img_path).convert("RGBA")
        bbox = fish_img.getbbox()

        if bbox is None:
            return background_img, []

        left, upper, right, lower = bbox
        fish_width = right - left
        fish_height = lower - upper

        if fish_width > background_img.shape[1] or fish_height > background_img.shape[0]:
            print(f"Fish image {fish_img_path} is too large for the background.")
            return background_img, []

        # Try to find a position
        for _ in range(100):
            x_offset = random.randint(0, background_img.shape[1] - fish_width)
            y_offset = random.randint(0, background_img.shape[0] - fish_height)

            new_box = (x_offset, y_offset, x_offset + fish_width, y_offset + fish_height)
            if not self.does_overlap(occupied_boxes, new_box):
                fish_img_cropped = fish_img.crop(bbox)

                fish_cv = cv2.cvtColor(np.array(fish_img_cropped), cv2.COLOR_RGBA2BGRA)
                alpha_fish = fish_cv[:, :, 3] / 255.0

                source_x = (left + fish_width/2)/background_img.shape[1]
                source_y = (upper + fish_height/2)/background_img.shape[0]
                source_coords = (source_x, source_y)
                adj_coords = ((x_offset + fish_width/2)/background_img.shape[1],(y_offset + fish_height/2)/background_img.shape[0])
                brightness_adj = self.brightness_augment.find_brightness(source_coords, adj_coords)

                # Adjust brightness of the fish image
                fish_cv[:, :, :3] = np.clip(fish_cv[:, :, :3] * brightness_adj, 0, 255).astype(np.uint8)

                overlay_height, overlay_width, _ = fish_cv.shape

                y_end = min(y_offset + overlay_height, background_img.shape[0])
                x_end = min(x_offset + overlay_width, background_img.shape[1])

                # Ensure sizes are valid
                if y_offset >= y_end or x_offset >= x_end:
                    continue

                for c in range(0, 3):
                    background_img[y_offset:y_end, x_offset:x_end, c] = (
                        alpha_fish[:y_end - y_offset, :x_end - x_offset] * fish_cv[:y_end - y_offset, :x_end - x_offset, c] +
                        (1 - alpha_fish[:y_end - y_offset, :x_end - x_offset]) * background_img[y_offset:y_end, x_offset:x_end, c]
                    )

                # Track the occupied area
                occupied_boxes.append(new_box)

                # Adjust fish label coordinates to the new position
                fish_label_transformed = []
                for (class_id, x_center, y_center, width, height) in fish_labels:
                    x_center = (x_offset + fish_width / 2) / background_img.shape[1]
                    y_center = (y_offset + fish_height / 2) / background_img.shape[0]
                    fish_label_transformed.append((class_id, x_center, y_center, width, height))

                return background_img, fish_label_transformed

        print(f"Could not find a suitable position for {fish_img_path}.")
        return background_img, []

    def save_yolo_label_file(self, image_path, fish_data):
        label_path = image_path.replace('.jpg', '.txt')
        with open(label_path, 'w') as file:
            for (class_id, x_center, y_center, w, h) in fish_data:
                file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

    def generate_augmented_images(self):
        for i in range(self.num_augmented_images):
            background_copy = self.background_bgr.copy()
            all_fish_labels = []
            occupied_boxes = []

            num_fish_to_overlay = random.randint(3, 5)
            for _ in range(num_fish_to_overlay):
                fish_image_name = random.choice(self.fish_images)
                fish_image_path = os.path.join(self.segmented_fish_dir, fish_image_name)
                label_path = os.path.join(self.labels_dir, fish_image_name.replace('.png', '.txt'))

                if not os.path.exists(label_path):
                    continue
                
                fish_labels = self.load_fish_label(label_path)
                background_copy, fish_labels_transformed = self.overlay_fish(background_copy, fish_image_path, fish_labels, occupied_boxes)
                all_fish_labels.extend(fish_labels_transformed)
            
            # Save the augmented image
            output_image_path = os.path.join(self.output_directory, f'augmented_image_{i + 1}.jpg')
            cv2.imwrite(output_image_path, background_copy)

            # Save the label file
            self.save_yolo_label_file(output_image_path, all_fish_labels)'''

import cv2
import numpy as np
from PIL import Image
import os
import random

class ObjectPlacer:
    def __init__(self, background_path, num_augmented_images, brightness_augment, placed_obj_dir, label_dir):
        self.num_augmented_images = num_augmented_images
        self.background_path = background_path
        self.brightness_augment = brightness_augment
        self.segmented_fish_dir = placed_obj_dir
        self.labels_dir = label_dir
        self.output_directory = 'augmented_seg'

        # Load background image
        self.background_bgr = cv2.imread(self.background_path)
        if self.background_bgr is None:
            raise FileNotFoundError(f"Background image not found at {self.background_path}")

        # Create output directory
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.segmented_fish_dir, exist_ok=True)

        # Load segmented fish images
        self.fish_images = os.listdir(self.segmented_fish_dir)
        if not self.fish_images:
            raise FileNotFoundError(f"Fish images not found at {self.fish_images}")

    def load_fish_label(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        fish_data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            fish_data.append((class_id, x_center, y_center, width, height))
        
        return fish_data

    def does_overlap(self, existing_boxes, new_box):
        new_x1, new_y1, new_x2, new_y2 = new_box
        for (x1, y1, x2, y2) in existing_boxes:
            if (new_x1 < x2 and new_x2 > x1 and
                    new_y1 < y2 and new_y2 > y1):
                return True
        return False

    def average_brightness(self, image):
        return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def adjust_brightness(self, image, target_brightness, max_adjustment=0.2):
        current_brightness = self.average_brightness(image)
        if current_brightness == 0:
            return image  # Avoid division by zero
        
        # Calculate the adjustment
        brightness_difference = target_brightness - current_brightness
        adjustment = max_adjustment * brightness_difference  # Limit adjustment to a percentage

        # Create a new adjusted image
        adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=adjustment)
        return adjusted_image

    def overlay_fish(self, background_img, fish_img_path, fish_labels, occupied_boxes):
        fish_img = Image.open(fish_img_path).convert("RGBA")
        bbox = fish_img.getbbox()

        if bbox is None:
            return background_img, []

        left, upper, right, lower = bbox
        fish_width = right - left
        fish_height = lower - upper

        if fish_width > background_img.shape[1] or fish_height > background_img.shape[0]:
            print(f"Fish image {fish_img_path} is too large for the background.")
            return background_img, []

        # Try to find a position
        for _ in range(100):
            x_offset = random.randint(0, background_img.shape[1] - fish_width)
            y_offset = random.randint(0, background_img.shape[0] - fish_height)

            new_box = (x_offset, y_offset, x_offset + fish_width, y_offset + fish_height)
            if not self.does_overlap(occupied_boxes, new_box):
                fish_img_cropped = fish_img.crop(bbox)

                fish_cv = cv2.cvtColor(np.array(fish_img_cropped), cv2.COLOR_RGBA2BGRA)
                alpha_fish = fish_cv[:, :, 3] / 255.0

                # Adjust the fish brightness based on the new position
                source_x = (left + fish_width / 2) / background_img.shape[1]
                source_y = (upper + fish_height / 2) / background_img.shape[0]
                source_coords = (source_x, source_y)
                adj_coords = ((x_offset + fish_width / 2) / background_img.shape[1], (y_offset + fish_height / 2) / background_img.shape[0])
                brightness_adj = self.brightness_augment.find_brightness(source_coords, adj_coords)

                # Adjust brightness of the fish image
                fish_cv[:, :, :3] = np.clip(fish_cv[:, :, :3] * brightness_adj, 0, 255).astype(np.uint8)

                overlay_height, overlay_width, _ = fish_cv.shape

                y_end = min(y_offset + overlay_height, background_img.shape[0])
                x_end = min(x_offset + overlay_width, background_img.shape[1])

                # Ensure sizes are valid
                if y_offset >= y_end or x_offset >= x_end:
                    continue

                for c in range(0, 3):
                    background_img[y_offset:y_end, x_offset:x_end, c] = (
                        alpha_fish[:y_end - y_offset, :x_end - x_offset] * fish_cv[:y_end - y_offset, :x_end - x_offset, c] +
                        (1 - alpha_fish[:y_end - y_offset, :x_end - x_offset]) * background_img[y_offset:y_end, x_offset:x_end, c]
                    )

                # Track the occupied area
                occupied_boxes.append(new_box)

                # Adjust fish label coordinates to the new position
                fish_label_transformed = []
                for (class_id, x_center, y_center, width, height) in fish_labels:
                    x_center = (x_offset + fish_width / 2) / background_img.shape[1]
                    y_center = (y_offset + fish_height / 2) / background_img.shape[0]
                    fish_label_transformed.append((class_id, x_center, y_center, width, height))

                return background_img, fish_label_transformed

        print(f"Could not find a suitable position for {fish_img_path}.")
        return background_img, []

    def save_yolo_label_file(self, label_path, fish_data):
        with open(label_path, 'w') as file:
            for (class_id, x_center, y_center, w, h) in fish_data:
                file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

    def generate_augmented_images(self):
        print('Generating augmented images...')
        os.makedirs('augmented_seg/images', exist_ok=True)
        os.makedirs('augmented_seg/labels', exist_ok=True)

        for i in range(self.num_augmented_images):
            background_copy = self.background_bgr.copy()
            all_fish_labels = []
            occupied_boxes = []

            num_fish_to_overlay = random.randint(3, 5)
            for _ in range(num_fish_to_overlay):
                filename = random.choice(self.fish_images)
                base_name, ext = os.path.splitext(filename)
                fish_image_name = base_name.replace('.png', '', 1)
                fish_image_path = os.path.join(self.segmented_fish_dir, filename)
                label_path = os.path.join(self.labels_dir, fish_image_name + '.txt')

                if not os.path.exists(label_path):
                    print(f'Skipping. The path does not exist: {label_path}')
                    continue
                
                fish_labels = self.load_fish_label(label_path)
                background_copy, fish_labels_transformed = self.overlay_fish(background_copy, fish_image_path, fish_labels, occupied_boxes)
                all_fish_labels.extend(fish_labels_transformed)
            
            # Save the augmented image
            output_image_path = os.path.join(self.output_directory, 'images', f'augmented_image_{i + 1}.jpg')
            cv2.imwrite(output_image_path, background_copy)

            # Save the label file
            output_label_path = os.path.join(self.output_directory, 'labels', f'augmented_image_{i + 1}.txt')
            self.save_yolo_label_file(output_label_path, all_fish_labels)
    
    def convert_folder_jpg_to_png(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
        # Iterate through all files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                input_path = os.path.join(input_folder, filename)
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)
                
                # Open the JPG image and save it as PNG
                img = Image.open(input_path)
                img.save(output_path, "PNG")
                print(f"Converted {input_path} to {output_path}")