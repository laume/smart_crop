from glob import glob
import pandas as pd
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle, Circle
import numpy as np
from tensorflow import keras
from fastprogress import progress_bar
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from typing import Optional
import math


def files_to_df(folder: str, file_type: str) -> pd.DataFrame:
    all_files = []
    for filename in glob(str(folder / f"**/*.{file_type}"), recursive=True):
        try:
            img_id = filename.split('/')[-1]
            all_files.append({
                'filename': filename,
                'ids': img_id,
            })
        except:
            pass
    return pd.DataFrame(all_files)


def parse_xml(file_to_parse):
    tree = ET.parse(file_to_parse)
    root = tree.getroot()

    return {elem.tag: elem.text for elem in root.iter()}


def parse_xml_to_df(xml_path: str) -> pd.DataFrame:
    rows_list = []
    for xml_filename in glob(f'{xml_path}/*', recursive=True):
        try:
            rows_list.append(parse_xml(xml_filename))
        except:
            pass
    interesting_cols = [
        'filename',
        'name',
        'pose',
        'width',
        'height',
        'depth',
        'xmin',
        'ymin',
        'xmax',
        'ymax',
        ]
    df = pd.DataFrame(rows_list)
    if not df.empty:
        df = df[interesting_cols]
    return df


def add_bbox_ratio(dataset):
    dataset["xmin"] = dataset["xmin"] / dataset["width"]
    dataset["xmax"] = dataset["xmax"] / dataset["width"]
    dataset["ymin"] = dataset["ymin"] / dataset["height"]
    dataset["ymax"] = dataset["ymax"] / dataset["height"]
    dataset["object_width"] = dataset["xmax"] - dataset["xmin"]
    dataset["object_height"] = dataset["ymax"] - dataset["ymin"]

    return dataset


def make_rectangle(x0, y0, obj_width, obj_height, img_dims, color, linewidth=2):
    return Rectangle(
        (x0 * img_dims[1], y0 * img_dims[0]),
        obj_width * img_dims[1],
        obj_height * img_dims[0],
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )



def get_category_name(category_id: int, category_map, with_id: bool = False) -> str:
    return (
        f"{category_id} ({category_map[category_id]})"
        if with_id
        else category_map[category_id]
    )


def plot_tf_dataset_images(dataset, num_img, img_dims, category_map):
    for x, y in dataset.take(num_img):
        n_image = 0
        categories, bounding_boxes = y
        category = categories[n_image].numpy()
        bounding_box = bounding_boxes[n_image].numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(x[n_image].numpy())
        plt.text(
            bounding_box[0] * img_dims[0],
            bounding_box[1] * img_dims[1],
            get_category_name(category, category_map),
            backgroundcolor="red",
            fontsize=12,
        )
        y_rect = make_rectangle(*bounding_box, img_dims, "r")
        ax.add_patch(y_rect)
        plt.show()


def predict_one(model, dataset, img_dims, category_map, n_image=0):
    for x, y in dataset.take(1):
        fig, ax = plt.subplots(1)
        ax.imshow(x[n_image].numpy())

        p_categories, p_bounding_boxes = model.predict(x)
        p_category = p_categories[n_image].argmax()
        p_bounding_box = p_bounding_boxes[n_image]
        plt.text(
            (p_bounding_box[0] + p_bounding_box[2]) * img_dims[0] - 25,
            p_bounding_box[1] * img_dims[1],
            get_category_name(p_category, category_map),
            color="white",
            backgroundcolor="blue",
            fontsize=16,
        )
        p_rect = make_rectangle(*p_bounding_box, img_dims, "b", 3)
        ax.add_patch(p_rect)

        y_categories, y_bounding_boxes = y
        y_category = y_categories[n_image].numpy()
        y_bounding_box = y_bounding_boxes[n_image].numpy()
        plt.text(
            y_bounding_box[0] * img_dims[0] + 5,
            y_bounding_box[1] * img_dims[1],
            get_category_name(y_category, category_map),
            backgroundcolor="red",
            fontsize=16,
        )
        y_rect = make_rectangle(*y_bounding_box, img_dims, "r")
        ax.add_patch(y_rect)

        plt.show()


def make_report(model, dataset, category_map):
    print(
        classification_report(
            [category.numpy() for _, (category, _) in dataset.unbatch()],
            model.predict(dataset)[0].argmax(axis=1),
            target_names=list(category_map.values()),
        )
    )


def analyse_dataset(model, dataset, verbose: int = 0):
    images = [x.numpy() for x, _ in dataset.unbatch()]
    probs = model.predict(dataset)[0]
    pred_code = probs.argmax(axis=1)
    label_code = [int(category.numpy()) for _, (category, _) in dataset.unbatch()]
    return pd.DataFrame.from_dict(
        {
            "image": images,
            "label": label_code,
            "pred": pred_code,
            "label_probs": probs[:, label_code][np.eye(len(label_code), dtype=bool)],
            "pred_probs": probs[:, pred_code][np.eye(len(pred_code), dtype=bool)],
        }
    )


def show_categorical_predictions(
    model,
    dataset,
    category_map,
    correct: bool = False,
    ascending: bool = True,
    cols: int = 8,
    rows: int = 2,
):
    df = analyse_dataset(model, dataset)
    df = df[(df.label == df.pred) if correct else (df.label != df.pred)]
    if not df.empty:
        df.sort_values(by=["label_probs"], ascending=ascending, inplace=True)
        _, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
        for i, row in enumerate(df.head(cols * rows).itertuples()):
            idx = (i // cols, i % cols) if rows > 1 else i % cols
            ax[idx].axis("off")
            ax[idx].imshow(row.image)
            ax[idx].set_title(
                f"{get_category_name(row.label, category_map)}\n{get_category_name(row.pred, category_map)}\n{row.label_probs:.4f}:{row.pred_probs:.4f}"
            )
    else:
        return 'Nothing to show! 100% accuracy!'


def get_predicted_bbox(model, dataset, data_df, verbose: int = 0, testing: bool = False):

    images = [x.numpy() for x, _ in dataset.unbatch()]
    prob_cat, pred_bbox = model.predict(dataset)
    pred_bbox_tuple = [(x) for x in pred_bbox]
    pred_code = prob_cat.argmax(axis=1)
    if not testing:
        label_code = [int(category.numpy()) for _, (category, _) in dataset.unbatch()]
        real_bbox = [category[1].numpy() for _, category in dataset.unbatch()]
        return pd.DataFrame.from_dict(
            {
                "image": images,
                "label": label_code,
                "pred": pred_code,
                "real_bbox": real_bbox,
                "pred_bbox": pred_bbox_tuple,
                "ids": data_df['ids'].tolist(),
            },
        )
    else:
        return pd.DataFrame.from_dict(
            {
                "image": images,
                "pred": pred_code,
                "pred_bbox": pred_bbox_tuple,
                "ids": data_df['ids'],
            },
        )


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
    return I / U


def show_iou(
        model,
        dataset,
        sorting_param,
        data_df,
        category_map,
        img_dims,
        ascending: bool = True,
        cols: int = 8,
        rows: int = 2,
        testing: bool = False,
):
    df = get_predicted_bbox(model, dataset, data_df, testing)
    if not testing:
        df['iou'] = [iou(x.real_bbox, x.pred_bbox) for x in df.itertuples()]
        if not df.empty:
            print('IoU info', df.iou.describe())
            df.sort_values(by=[sorting_param], ascending=ascending, inplace=True)
            _, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
            for i, row in enumerate(df.head(cols * rows).itertuples()):
                idx = (i // cols, i % cols) if rows > 1 else i % cols
                ax[idx].axis("off")
                ax[idx].imshow(row.image)

                p_rect = make_rectangle(*row.pred_bbox, img_dims, "b", 3)
                ax[idx].add_patch(p_rect)

                y_rect = make_rectangle(*row.real_bbox, img_dims, "r")
                ax[idx].add_patch(y_rect)

                ax[idx].set_title(
                    f"id: {row.ids}\nlabel: {get_category_name(row.label, category_map)}: pred: {get_category_name(row.pred, category_map)}\nIoU: {row.iou:.4f}"
                )
        else:
            return 'Nothing to show! You created best model ever!'



def get_pred_data(model, dataset, data_df, verbose: int = 0):

    prob_cat, pred_bbox = model.predict(dataset)
    pred_bbox_tuple = [(x) for x in pred_bbox]
    pred_code = prob_cat.argmax(axis=1)
    df =  pd.DataFrame.from_dict(
        {
            "pred": pred_code,
            "pred_bbox": pred_bbox_tuple,
            "ids": data_df['ids'].tolist(),
        },
    )
    # print('Before preprocess: ', len(data_df))
    df = df.merge(data_df, left_on='ids', right_on='ids')
    # print('After preprocess: ', len(df))
    df[['pxmin', 'pymin', 'pw', 'ph']] = pd.DataFrame(df["pred_bbox"].values.tolist(),
                                                                 index=df.index)
    return df


def plot_one_dataset_img(dataframe, img_id, category_map):
    row = dataframe[dataframe["ids"] == img_id]
    for i in row.itertuples():
        filename = i.filename_x
        category = i.category
        bounding_box = [i.xmin, i.ymin, i.object_width, i.object_height]
        image = plt.imread(filename)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        plt.text(
            bounding_box[0] * i.width,
            bounding_box[1] * i.height,
            get_category_name(category, category_map),
            backgroundcolor="red",
            fontsize=12,
        )
        y_rect = make_rectangle(*bounding_box, (i.height, i.width), "r")
        ax.add_patch(y_rect)
        plt.show()


def plot_real_df_images_bbox(
        df: pd.DataFrame,
        image_count: int,
        random_plot: Optional[bool]=False,
    ) -> None:

    if image_count < 4:
        cols, rows = (image_count, 1)
    else:
        cols, rows = (4, math.ceil(image_count / 4))
    if len(df) < image_count:
        image_count = len(df)

    if not random_plot:
        images_to_plot = df.head(image_count)
    else:
        images_to_plot = df.sample(image_count)

    _, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for i, row in enumerate(images_to_plot.itertuples()):
        idx = (i // cols, i % cols) if rows > 1 else i % cols
        # bounding_box = (row.xmin, row.ymin, row.xmax, row.ymax)

        ax[idx].add_patch(Rectangle((row.xmin, row.ymin),
                                    row.xmax - row.xmin,
                                    row.ymax - row.ymin,
                                    ec = 'r', fc = 'none',
                                    lw = 2))

        ax[idx].imshow(plt.imread(row.filename_x))
        ax[idx].set_title(f'{row.ids} -> {row.name}')

    plt.show()


def plot_final_df_images_bbox(
        df: pd.DataFrame,
        image_count: int,
        random_plot: Optional[bool]=False,
    ) -> None:

    if image_count < 4:
        cols, rows = (image_count, 1)
    else:
        cols, rows = (4, math.ceil(image_count / 4))
    if len(df) < image_count:
        image_count = len(df)

    if not random_plot:
        images_to_plot = df.head(image_count)
    else:
        images_to_plot = df.sample(image_count)

    _, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for i, row in enumerate(images_to_plot.itertuples()):
        idx = (i // cols, i % cols) if rows > 1 else i % cols
        img_dims = (row.height, row.width, 3)
        bounding_box = (row.xmin, row.ymin, row.object_width, row.object_height)
        y_rect = make_rectangle(*bounding_box, img_dims, "r", 3)
        ax[idx].add_patch(y_rect)

        pred_bbox = (row.pxmin, row.pymin, row.pw, row.ph)
        p_rect = make_rectangle(*pred_bbox, img_dims, "b", 3)
        ax[idx].add_patch(p_rect)

        pred_center = (
                (row.pxmin + (row.pw / 2)) * img_dims[1],
                (row.pymin + (row.ph / 2)) * img_dims[0]
            )
        ax[idx].add_patch(Circle((pred_center), radius=10, ec='b', lw = 2, fc='b'))

        ax[idx].imshow(plt.imread(row.filename_x))
        ax[idx].set_title(f'{row.ids} -> {row.name}')

    plt.show()


def get_image_dims(arr):
    heights, widths = [], []
    for filename in progress_bar(arr):
        height, width, _ = np.array(keras.preprocessing.image.load_img(filename)).shape
        heights.append(height)
        widths.append(width)
    return np.asarray(heights), np.asarray(widths)