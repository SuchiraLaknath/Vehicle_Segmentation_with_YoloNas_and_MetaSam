import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from yolo_nas import YoloPrediction
import cv2
import random
from config_inference import vehicle_color_dict

class SamSegmentation:
    def __init__(self) -> None:
        self.device = self.select_device()
        self.model = self.select_model()

    def select_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select_model(self, model_name = "vit_l", model_weights ="checkpoints/sam_vit_l_0b3195.pth", device = None):
        if device == None:
            device = self.device
        sam = sam_model_registry[model_name](checkpoint=model_weights).to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    
    def preprocess_bbox_cordinaes(self, yolo_predictions):
        vehicle_list = vehicle_color_dict.keys()
        bboxs = [[prediction["bbox"], prediction["label_name"]]for prediction in yolo_predictions]
        bboxs = []
        for prediction in yolo_predictions:
            prediction_label_name = prediction["label_name"]
            if prediction_label_name in vehicle_list:
                prediction_bbox = prediction["bbox"]
                bboxs.append({
                    "label": prediction_label_name,
                    "bbox": prediction_bbox
                })
            
        
        return bboxs
    
    def draw_masks_fromDict(self, image, masks_generated) :
        masked_image = image.copy()
        for i in range(len(masks_generated)) :
            print(len(masks_generated[i]))
            for mask in masks_generated[i]:
                masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                                    np.random.choice(range(256), size=3),
                                    masked_image)

                masked_image = masked_image.astype(np.uint8)

        return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)
    
    def draw_bboxs(self, image, bboxs):
        clone_image = image.copy()
        for bbox in bboxs:
            x,y,w,h = bbox["bbox"]
            clone_image = cv2.rectangle(clone_image, (x, y), ( w, h ), (255,0,0), 4)
        return clone_image

    
    def predict_images(self, img, yolo_predictions , model= None):
        image = img.copy()
        if not model:
            model = self.model
        bboxs = self.preprocess_bbox_cordinaes(yolo_predictions=yolo_predictions)
        
        masks = []
        model.set_image(image)
        combined_mask = np.zeros((image.shape[0], image.shape[1], 3))
        for bbox in bboxs:
            box_cordinates = np.array(bbox["bbox"])
            box_label = bbox["label"]
            mask, _, _ = model.predict(
                point_coords=None,
                point_labels=None,
                box= box_cordinates[None, :],
                multimask_output=False)
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            mask_color = np.array(vehicle_color_dict[box_label])
            combined_mask += mask[0][..., None] * mask_color
        return combined_mask


def mian():
    yolo = YoloPrediction()
    image = cv2.imread("data/cab_cars_street_urban_city-1409810.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    list_of_boxes = yolo.predict_image(image= image)
    
    sam = SamSegmentation()
    bboxs = sam.preprocess_bbox_cordinaes(yolo_predictions=list_of_boxes)
    masked_image = sam.draw_bboxs(image=image,bboxs=bboxs)
    masks_genrated = sam.predict_images(img=image, yolo_predictions= list_of_boxes)
    masked_image = cv2.addWeighted(masked_image, 0.3, masks_genrated.astype(np.uint8), 0.7, 0)
    print(len(masks_genrated))
    # masked_image = sam.draw_masks_fromDict(image= image, masks_generated=masks_genrated)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Final Image output", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo = YoloPrediction()
    image = cv2.imread("data/cab_cars_street_urban_city-1409810.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    list_of_boxes = yolo.predict_image(image= image)
    
    sam = SamSegmentation()
    bboxs = sam.preprocess_bbox_cordinaes(yolo_predictions=list_of_boxes)
    masked_image = sam.draw_bboxs(image=image,bboxs=bboxs)
    masks_genrated = sam.predict_images(img=image, yolo_predictions= list_of_boxes)
    masked_image = cv2.addWeighted(masked_image, 0.3, masks_genrated.astype(np.uint8), 0.7, 0)
    print(len(masks_genrated))
    # masked_image = sam.draw_masks_fromDict(image= image, masks_generated=masks_genrated)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Final Image output", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
