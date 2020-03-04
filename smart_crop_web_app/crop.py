from flask import Flask, request, redirect, render_template, send_file, after_this_request, make_response, render_template_string, send_from_directory
from .utils import smart_crop_pipleline, files_to_df
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import base64
import os
import glob

from werkzeug.utils import secure_filename


PRED_SETTINGS = {
    'img_dims': (299, 299, 3),
    'batch_size': 32,
    'category_map' : {0: 'dog', 1: 'cat'},
    'model_acrh': str('./model/Xception_model_1_3'),
    'weights': str('./model/model1_3_xception.h5'),
}

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'])
UPLOAD_FOLDER = 'uploads'
TARGET_SIZE = (400, 400)


app = Flask(__name__, static_url_path="", static_folder=UPLOAD_FOLDER)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def empty_dir(folder: str, allowed_extensions: set):
    for f in glob.glob(folder + f"/*", recursive=True):
        extension = f.split('.')[-1]
        if extension in allowed_extensions:
            os.remove(f)


@app.route('/', methods=['GET', 'POST'])
def crop_file():
    if request.method == 'GET':
        return render_template('index.html', value='hi')

    if request.method == 'POST':
        empty_dir(app.static_folder, ALLOWED_EXTENSIONS)
        print(request.files)

        # files = request.files.getlist("file")
        # for file in files:

        if 'file' not in request.files:
            print('file not uploaded')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.static_folder, filename))

        # Model predictions:
        df_from_dir = files_to_df(app.static_folder, ALLOWED_EXTENSIONS)
        result = smart_crop_pipleline(
                df_from_dir,
                'filename',
                PRED_SETTINGS,
                target_size=TARGET_SIZE
            )

        results = []
        for img in result:
            category = img[0]
            filename_res = f'{app.static_folder}/{category}.jpg'
            plt.imsave(filename_res, img[1], cmap='Greys')
            results.append((category, filename_res))


        ###
        # return send_file(os.path.join(app.static_folder, filename_res), mimetype='image/gif')
        ###

        # response = send_file(filename_res, mimetype='image/jpeg', attachment_filename=category, as_attachment=False)
        # response.headers["x-filename"] = category
        # response.headers["Access-Control-Expose-Headers"] = 'x-filename'

        # works good
        response = make_response(send_file(filename_res))
        response.headers['Predicted Category'] = category
        return response


        #### or detection example
        # return render_template('result.html', results=results)



if __name__ == "__main__":
    app.run(debug=True)
