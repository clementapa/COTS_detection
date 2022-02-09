'''
    Author: Clément APAVOU
'''
import wandb
from utils.constant import COTS_CLASSES

class WandbLogger():
    
    def __init__(self, name, entity, project, config):
        self.run = wandb.init(name=name, project=project, config=config, entity=entity)

    def log_metrics(self, metrics):
        """
        metrics dict e.g {str: float}
        """
        self.run.log(metrics)

    def log_images(self, batch, name_set, n, outputs = None):
        """
        """
        images, targets = batch

        images = [img.cpu().numpy().transpose((1, 2, 0)) for img in list(images)[:n]]
        targets = [{k: v.cpu().numpy() for k, v in t.items()} for t in targets[:n]]
        if outputs != None:
            outputs = [{k: v.cpu().detach().numpy() for k, v in out.items()} for out in outputs[:n]]
        
        pos_conv = ["minX", "minY", "maxX", "maxY"]

        samples = []
        for i in range(len(images)):
            image = images[i]
            target = targets[i]

            dict_boxes = {}

            if outputs != None:
                output = outputs[i]
                box_data_preds = []
                boxes = output["boxes"] # FIXME différent pour yolo 
                labels = output["labels"]
                scores = output["scores"]
                for box, label, score in zip(boxes, labels, scores):
                    dict_pred = {}
                    box_caption = COTS_CLASSES[label]
                    position = {pos_conv[i]:float(box[i]) for i in range(4)}
                    
                    dict_pred["position"] = position
                    dict_pred["class_id"] = int(label)
                    dict_pred["box_caption"] = box_caption
                    dict_pred["scores"] = {"confidence": float(score)}
                    dict_pred["domain"] = "pixel"

                    box_data_preds.append(dict_pred)

                dict_boxes["predictions"] = {'box_data': box_data_preds, "class_labels": COTS_CLASSES}

            box_data_gt = []
            boxes = target['boxes'] # FIXME différent pour yolo 
            labels = target['labels'] # FIXME différent pour yolo 
            for box, label in zip(boxes, labels):
                
                dict_gt = {}
                box_caption = COTS_CLASSES[label]
                position = {pos_conv[i]:float(box[i]) for i in range(4)}
                
                dict_gt["position"] = position
                dict_gt["class_id"] = int(label)
                dict_gt["box_caption"] = box_caption
                dict_gt["domain"] = "pixel" # FIXME différent pour yolo je crois

                box_data_gt.append(dict_gt)
            
            dict_boxes["ground_truth"] = {'box_data': box_data_gt, "class_labels": COTS_CLASSES}

            img = wandb.Image(image, boxes=dict_boxes)

            samples.append(img)

        self.run.log({name_set: samples})
    
    def log_videos(self, batch, name_set):
        pass 

    def log_hyperparams(self):
        pass

