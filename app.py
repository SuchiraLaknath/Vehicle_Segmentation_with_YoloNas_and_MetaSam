from sam_inference import SamSegmentation
from yolo_nas import YoloPrediction
import cv2
import numpy as np
import config_inference


def main():
    yolo = YoloPrediction()
    image = cv2.imread(config_inference.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    list_of_boxes = yolo.predict_image(image= image)

    sam = SamSegmentation()
    bboxs = sam.preprocess_bbox_cordinaes(yolo_predictions=list_of_boxes)
    masked_image = sam.draw_bboxs(image=image,bboxs=bboxs)
    masks_genrated = sam.predict_images(img=image, yolo_predictions= list_of_boxes)
    masked_image = cv2.addWeighted(image, 0.3, masks_genrated.astype(np.uint8), 0.7, 0)
    print(len(masks_genrated))
    # masked_image = sam.draw_masks_fromDict(image= image, masks_generated=masks_genrated)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Final Image output", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()