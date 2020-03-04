from typing import Optional, Union
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from .image import ImageParser, ImageResizer
import numpy as np
import pandas as pd
from .utils_multi_labels import get_category_name
import cv2
import glob


def load_model_json(model_arch_path, model_weights_path):
    with open(model_arch_path, "r") as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights(model_weights_path)
    return model

def imgs_for_pred(
        df: pd.DataFrame,
        path_col: str,
        img_dims: tuple,
        batch_size: int,
    ):
    image_set = (
    tf.data.Dataset.from_tensor_slices(
        (
            df[path_col],
            tuple([]),
        )
    )
    .map(ImageParser.ImageParser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .map(ImageResizer.ImageResizer(img_dims, "stretch"), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
    return image_set


def smart_crop_pipleline(
        img_df: pd.DataFrame,
        path_col: str,
        pred_settings: dict,
        model,
        target_ratio: Optional[tuple] = (1, 1),
        target_size: Optional[tuple] = None,
        return_df_without_crop: Optional[bool] = False,
) -> list:
    result = []
    image_set = imgs_for_pred(
        img_df,
        path_col,
        pred_settings['img_dims'],
        pred_settings['batch_size'],
    )
    prob_cat, pred_bbox = model.predict(image_set)
    pred_code = prob_cat.argmax(axis=1)
    pred_bbox_tuple = [(x) for x in pred_bbox]
    df = pd.DataFrame.from_dict(
        {
            "pred": pred_code,
            "pred_bbox": pred_bbox_tuple,
            "filename": img_df[path_col]
        },
    )
    if return_df_without_crop:
        return df

    for row in df.itertuples():
        category = get_category_name(row.pred, pred_settings['category_map'])
        cropped = bbox_crop_px(row.filename, target_ratio, row.pred_bbox)
        if target_size:
            cropped = cv2.resize(cropped, dsize=target_size, interpolation=cv2.INTER_CUBIC)
        result.append((category, cropped))

    return result


def detect_img_orientation(shape: tuple) -> str:
    if not len(shape) == 2:
        'Print wrong tuple shape.'
    h, w = shape
    if h < w:
        img_pos = 'landscape'
    elif h > w:
        img_pos = 'portrait'
    else:
        img_pos = 'square'
    return img_pos


def bbox_crop_px(
        image: Union[str, np.ndarray],
        target_ratio: tuple,
        bbox: tuple,
        bbox_vs_img_ratio: Optional[float]=0.35,
        bbox_vs_pet_ratio: Optional[float]=2.,
    ) -> np.ndarray:

    if type(image) is not np.ndarray:
        img = Image.open(image)
        img_arr = np.asarray(img)
    else:
        img_arr = image

    img_size = img_arr.shape[:2]  # (h, w)
    h, w = img_size
    bbox_h, bbox_w = bbox[3] * h, bbox[2] * w
    # max_bbox = (bbox_w, bbox_h)

    img_orient = detect_img_orientation(img_size)
    limiting_size_idx = (img_size).index(min(img_size))  # h, w

    target_orient = detect_img_orientation(target_ratio)

    center_w, center_h = (
        (bbox[0] * w) + (bbox_w / 2),
        (bbox[1] * h) + (bbox_h / 2),
    )

    start_w, start_h = [0, 0]
    end_w, end_h = [w, h]

    if target_orient == 'square':

        if bbox_w > w * bbox_vs_img_ratio or bbox_h > h * bbox_vs_img_ratio:
            size = img_size[limiting_size_idx]
            if img_orient == 'landscape' and limiting_size_idx == 0:
                start_w = max(center_w - (size / 2), 0)
                end_w = center_w + (size / 2) if start_w > 0 else size
            elif img_orient == 'portrait' and limiting_size_idx == 1:
                part_size = size / 6
                start_h = max(center_h - (part_size * 2), 0)
                end_h = center_h + (part_size * 4) if start_h > 0 else size

        else:
            size = max(bbox_h, bbox_w) * bbox_vs_pet_ratio
            part_size = size / 6
            start_w = max(center_w - (size / 2), 0)
            end_w = center_w + (size / 2) if start_w > 0 else size
            start_h = max(center_h - (part_size *2), 0)
            end_h = center_h + (part_size * 4) if start_h > 0 else size

        # plt.imshow(img_arr[int(start_h):int(end_h), int(start_w):int(end_w)])
        # plt.show()
        return img_arr[int(start_h):int(end_h), int(start_w):int(end_w)]
    else:
        print(f'{target_orient} target_ratio not supported yet! Choose (1, 1) or check again later :)')


def files_to_df(folder: str, allowed_extensions: set) -> pd.DataFrame:
    all_files = []
    for filename in glob.glob(folder + f"/*", recursive=True):
        try:
            img_id = filename.split('/')[-1]
            extension = filename.split('.')[-1]
            if extension in allowed_extensions:
                all_files.append({
                    'filename': filename,
                    'ids': img_id,
                })
        except:
            pass
    return pd.DataFrame(all_files)
