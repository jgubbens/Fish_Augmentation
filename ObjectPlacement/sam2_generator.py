import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2

class ImageSegmenter:
    def __init__(self, model_checkpoint, model_config, device=None):
        self.device = device or self._get_device()
        self.mask_generator = self._load_model(model_checkpoint, model_config)
    
    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_model(self, checkpoint, config):
        sam2 = build_sam2(config, checkpoint, device=self.device, apply_postprocessing=False)
        '''return SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )'''
        return SAM2AutomaticMaskGenerator(sam2)
    
    def _segment_image(self, image_path, label_path, output_name):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_width, image_height = image.shape[1], image.shape[0]
        masks = self.mask_generator.generate(image)
        print('Masks generated.')

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        self.show_anns(masks)
        plt.axis('off')
        #plt.show()

        masks_with_area = [(mask['area'], mask) for mask in masks]
        sorted_masks = sorted(masks_with_area, key=lambda x: x[0], reverse=True)
        
        bounding_boxes = self._extract_bounding_boxes(label_path, image_width, image_height)
        
        for counter, bounding_box in enumerate(bounding_boxes):
            print(f'Processing bounding box {counter}...')
            self._process_bounding_box(image, bounding_box, sorted_masks, output_name, counter)

    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    cv2.drawContours(img, [contour], -1, (0, 0, 1, 0.4), thickness=1)
        plt.imshow(img)

    def _extract_bounding_boxes(self, yolo_label_path, image_width, image_height):
        bounding_boxes = []
        try:
            with open(yolo_label_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    values = line.strip().split()
                    class_id = int(values[0])
                    x_center, y_center = float(values[1]), float(values[2])
                    width, height = float(values[3]), float(values[4])
                    x_min = (x_center - width / 2) * image_width
                    y_min = (y_center - height / 2) * image_height
                    x_max = (x_center + width / 2) * image_width
                    y_max = (y_center + height / 2) * image_height
                    bounding_boxes.append((x_min, y_min, x_max, y_max, class_id))
        except Exception as e:
            print(f"Error reading {yolo_label_path}: {e}")
        return bounding_boxes

    def _process_bounding_box(self, image, bounding_box, sorted_masks, output_name, counter):
        x_min, y_min, x_max, y_max, class_id = bounding_box
        
        if (x_min <= 5 or y_min <= 5 or x_max >= image.shape[1]-5 or y_max >= image.shape[0]-5):
            print(f'Skipping bounding box {bounding_box} as it touches the image edge.')
            return

        masks_within_bbox = [
            mask for mask in sorted_masks 
            if self._is_mask_within_bbox(mask[1]['segmentation'], bounding_box)
        ]

        if masks_within_bbox:
            largest_mask = max(masks_within_bbox, key=lambda x: x[0])
            self._save_largest_mask(image, largest_mask, bounding_box, class_id, output_name, counter)
            print('Saved masks.')
        else:
            print('No masks found within the bounding box.')

    def _is_mask_within_bbox(self, segmentation, bounding_box):
        coordinates = np.column_stack(np.where(segmentation > 0))
        x_min, y_min, x_max, y_max, _ = bounding_box
        return np.all((coordinates[:, 0] >= y_min) & (coordinates[:, 0] <= y_max) &
                       (coordinates[:, 1] >= x_min) & (coordinates[:, 1] <= x_max))

    '''
    # ORIGINAL VERSION
    def _save_largest_mask(self, image, largest_mask, bounding_box, class_id, output_name, counter):
        largest_segmentation = largest_mask[1]['segmentation']
        mask_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        mask_image[..., 3] = 0

        mask_image[largest_segmentation > 0, :3] = image[largest_segmentation > 0]
        mask_image[largest_segmentation > 0, 3] = 255

        output_path = os.path.join('segmented', 'images', f'{output_name}_seg{counter}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(mask_image).convert('RGBA').save(output_path)

        self._save_label_file(bounding_box, class_id, output_name, counter, image.shape)'''

    # SAVE BOUNDING BOX VERSION
    def _save_largest_mask(self, image, largest_mask, bounding_box, class_id, output_name, counter):
        largest_segmentation = largest_mask[1]['segmentation']
        mask_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        mask_image[..., 3] = 0

        # Create a mask for the largest segmentation
        mask_image[largest_segmentation > 0, :3] = image[largest_segmentation > 0]
        mask_image[largest_segmentation > 0, 3] = 255

        # Ensure bounding box coordinates are integers
        x_min, y_min, x_max, y_max, _ = map(int, bounding_box)

        # Crop the image and mask image to the bounding box
        cropped_mask_image = mask_image[y_min:y_max, x_min:x_max]
        
        # Adjust the output path to include the cropped version
        output_path = os.path.join('segmented', 'train', 'images', f'{output_name}_seg{counter}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the cropped mask image
        Image.fromarray(cropped_mask_image).convert('RGBA').save(output_path)

        # Save the label file
        self._save_label_file(bounding_box, class_id, output_name, counter, image.shape)


    def _save_label_file(self, bounding_box, class_id, output_name, counter, image_shape):
        x_min, y_min, x_max, y_max, _ = bounding_box
        h, w = image_shape[:2]

        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h

        output_label_path = os.path.join('segmented', 'train', 'labels', f'{output_name}_seg{counter}.txt')
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        
        with open(output_label_path, 'w') as file:
            file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

    def process_all_files(self, images_folder, labels_folder):
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"Images directory does not exist: {images_folder}")
        if not os.path.isdir(labels_folder):
            raise FileNotFoundError(f"Labels directory does not exist: {labels_folder}")

        for counter, file_name in enumerate(os.listdir(images_folder)):
            print(f'Processing image {counter}...')
            if file_name.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(images_folder, file_name)
                label_path = os.path.join(labels_folder, file_name.rsplit('.', 1)[0] + '.txt')

                if not os.path.isfile(label_path):
                    print(f"Label file not found for {file_name}")
                    continue
                
                self._segment_image(image_path, label_path, f'image{counter}')

    def copy_labels_with_full_image_bbox(self, labels_folder, output_folder, images_folder):
        if not os.path.isdir(labels_folder):
            raise FileNotFoundError(f"Labels directory does not exist: {labels_folder}")
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"Images directory does not exist: {images_folder}")

        for label_file in os.listdir(labels_folder):
            label_path = os.path.join(labels_folder, label_file)
            
            # Check if the file is a valid YOLO label file
            if label_file.lower().endswith('.txt') and os.path.isfile(label_path):
                image_name = label_file.rsplit('.', 1)[0]
                image_path = os.path.join(images_folder, f"{image_name}.jpg")  # Assuming the images are .jpg, adjust as necessary.
                
                if not os.path.isfile(image_path):
                    print(f"Image file not found for label: {label_file}")
                    continue

                # Get image width and height to normalize the bounding box
                image = cv2.imread(image_path)
                image_height, image_width = image.shape[:2]

                # Read the existing label file
                with open(label_path, 'r') as file:
                    lines = file.readlines()

                # Open the new label file in the destination folder
                new_label_path = os.path.join(output_folder, label_file)
                with open(new_label_path, 'w') as new_file:
                    for line in lines:
                        values = line.strip().split()
                        class_id = int(values[0])

                        # Set the new bounding box to cover the entire image
                        x_center = 0.5  # Center of the image in normalized coordinates
                        y_center = 0.5  # Center of the image in normalized coordinates
                        bbox_width = 1.0  # Full width of the image (normalized)
                        bbox_height = 1.0  # Full height of the image (normalized)

                        # Write the new label with full image bounding box
                        new_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                
                print(f"Processed label file: {label_file}")
