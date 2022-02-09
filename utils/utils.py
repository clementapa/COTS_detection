'''
    Author: Clément APAVOU
'''
from importlib import import_module
import cv2


def import_class(name, instantiate=None):

    namesplit = name.split(".")
    module = import_module(".".join(namesplit[:-1]))
    imported_class = getattr(module, namesplit[-1])

    if imported_class:
        if instantiate is not None:
            return imported_class(**instantiate)
        else:
            return imported_class
    raise Exception("Class {} can be imported".format(import_class))


def draw_predictions_and_targets(img, pred_bboxes, target_bboxes=None):
    img_pred = img.copy()
    # predictions
    if len(pred_bboxes) > 0:
        if len(pred_bboxes[0]) == 5:
            pred_bboxes = pred_bboxes[pred_bboxes[:, 0].argsort()
                                    [::-1]]  # sort by conf
        for bbox in pred_bboxes:
            if len(bbox) == 5:
                conf = bbox[0]
                x0, y0, x1, y1 = bbox[1:].round().astype(int)
            else:
                x0, y0, x1, y1 = bbox[:].round().astype(int)
            cv2.rectangle(
                img_pred,
                (x0, y0),
                (x1, y1),
                (0, 255, 255),
                thickness=2,
            )
            if len(bbox) == 5:
                cv2.putText(
                    img_pred,
                    f"{conf:.2}",
                    (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    thickness=1,
                )
    if target_bboxes != None:
        img_targ = img.copy()
        img_targ_pred = img_pred.copy()
        # target_bboxes
        if len(target_bboxes) > 0:
            for bbox in target_bboxes:
                x0, y0, x1, y1 = bbox[:].round().astype(int)
                cv2.rectangle(
                    img_targ,
                    (x0, y0),
                    (x1, y1),
                    (255, 0, 0),
                    thickness=2,
                )
                cv2.rectangle(
                    img_targ_pred,
                    (x0, y0),
                    (x1, y1),
                    (255, 0, 0),
                    thickness=2,
                )
    else:
        return {"img": img, "img_pred": img_pred}

    return {
        "img": img,
        "img_pred": img_pred,
        "img_targ": img_targ,
        "img_targ_pred": img_targ_pred
    }


def resize(img, coeff):
    return cv2.resize(img,
                      dsize=(img.shape[1] // coeff, img.shape[0] // coeff),
                      interpolation=cv2.INTER_LINEAR)