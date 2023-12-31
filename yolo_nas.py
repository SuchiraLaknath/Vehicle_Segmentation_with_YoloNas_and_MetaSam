from super_gradients.training import models
import numpy as np
import torch
import cv2

class YoloPrediction:
    def __init__(self) -> None:
        self.device = self.select_device()
        self.model = self.load_model(device=self.device)

    def get_class_names(self, predictions):
        return predictions.class_names
    
    def bbox_xyxy_to_xywh(self, xyxys):
        xywhs = []
        for xyxy in xyxys:
            xywhs.append([xyxy[0], xyxy[1], (xyxy[2]- xyxy[0]), (xyxy[3]- xyxy[1])])
        return xywhs

    def get_bboxs_and_labels(self, prediction, class_names):
        bboxs = prediction.prediction.bboxes_xyxy.astype(int).tolist()
        # bboxs = self.bbox_xyxy_to_xywh(bboxs)
        labels = prediction.prediction.labels
        confidances = prediction.prediction.confidence
        labels_names = [class_names[int(label)] for label in labels]
        list_of_bboxes = []
        # print(prediction)
        for bbox, label_name, label, confidance  in zip(bboxs, labels_names, labels, confidances):
            box_dict = {"bbox" : bbox,
                        "label_name" : label_name,
                        "label index": int(label),
                        "confidance": confidance}
            list_of_bboxes.append(box_dict)
        return list_of_bboxes

    def post_process_results(self, yolo_out):
        predictions = yolo_out._images_prediction_lst[0]
        class_names = self.get_class_names(predictions=predictions)
        # print(f"class_names = {class_names}")
        list_of_bboxs = self.get_bboxs_and_labels(predictions, class_names= class_names)
        return list_of_bboxs


    def select_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_name = "yolo_nas_s", pretrained_weights = "coco", device = "cpu"):
        return models.get(model_name=model_name, pretrained_weights= pretrained_weights).to(device=device)
    
    def predict_image(self, image, model = None , conf = 0.5):
        image_copy = image.copy()
        if model == None:
            model = self.model
        yolo_out = self.model.predict(image_copy, conf= conf)
        list_of_bboxs = self.post_process_results(yolo_out=yolo_out)
        return list_of_bboxs
        

if __name__ == "__main__":
    yolo = YoloPrediction()
    image = cv2.imread("data/cd2e47a9c2e1a74998eb66f92711de0c.webp")
    list_of_boxes = yolo.predict_image(image= image)
    print(list_of_boxes)
