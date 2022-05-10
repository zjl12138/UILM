from functools import reduce
import shutil
from typing import List
from flask import Flask, jsonify, request, send_from_directory, render_template
import os
from flask.helpers import url_for

from werkzeug.utils import redirect
from .mmdetapi import get_img_bbox
import tempfile
from dataclasses import dataclass
from PIL import Image, ImageDraw
import json

app = Flask(__name__)
work_dirs = '/work_dirs'
result_dir = '/result'


@dataclass
class LocalConfig:
    name: str
    checkpoints: List[str]
    config: str


local_config = None


def get_local_config() -> List[LocalConfig]:
    print(os.listdir(work_dirs))
    global local_config
    if local_config is None:
        model_dirs = [(os.path.join(work_dirs, f), f, f) for f in os.listdir(work_dirs) if
                      os.path.isdir(os.path.join(work_dirs, f)) and os.path.exists(os.path.join(work_dirs, f, f+'.py'))]
        model_dirs.extend(reduce(
            lambda x, y: x+y,
            [
                [(os.path.join(work_dirs, f, sub_f), f'{f}/{sub_f}', f) for sub_f in os.listdir(os.path.join(work_dirs, f)) if
                 os.path.isdir(os.path.join(work_dirs, f, sub_f)) and os.path.exists(os.path.join(work_dirs, f, sub_f, f+'.py'))]
                for f in os.listdir(work_dirs) if os.path.isdir(os.path.join(work_dirs, f))
            ],
            []
        ))
        print(model_dirs)
        local_config = [
            LocalConfig(
                model_dir[1],
                list(
                    sorted(
                        [f[:-4] for f in os.listdir(model_dir[0]) if
                         os.path.isfile(os.path.join(model_dir[0], f)) and f.endswith('.pth')],
                        key=lambda x: len(os.listdir(
                            model_dir[0]))+1 if x == 'latest' else (int(x[6:]) if x.startswith('epoch_') else 0),
                        reverse=True
                    )
                ),
                model_dir[2]+'.py'
            ) for model_dir in model_dirs]
    return local_config


@app.route("/get_config")
def get_config():
    return jsonify(get_local_config())

@app.route('/file/<path:path>')
def download(path):
    return send_from_directory(result_dir, path)


@app.route("/")
def new_html():
    return render_template("new_file_upload_form.html", configs=get_local_config())


@app.route("/get_bbox", methods=['POST'])
def new_get_bbox():
    if request.method == 'POST':
        form = request.form
        is_ajax = False
        if form.get("__ajax", None) == "true":
            is_ajax = True
        tmpdir = tempfile.mkdtemp()
        file = request.files['file']
        tmp_path = os.path.join(tmpdir, file.filename)
        file.save(tmp_path)
        config = form.get('config')
        checkpoint = form.get('checkpoint')
        nodisplay = form.get('nodisplay')
        local_config = next((x for x in get_local_config(
        ) if x.name == config and checkpoint in x.checkpoints), None)
        result = None
        all_label = []
        if local_config is not None:
            all_label = get_img_bbox(tmp_path,
                                     os.path.join(work_dirs, local_config.name,
                                                  local_config.config),
                                     os.path.join(work_dirs, local_config.name, checkpoint+'.pth'))
            new_image = Image.open(tmp_path).copy()
            if nodisplay is not None:
                result = jsonify(all_label)
            else:
                draw = ImageDraw.Draw(new_image)
                for label in all_label:
                    draw.rectangle(label[:-1], fill=None,
                                   outline='#ff0000', width=2)
                temp_name = next(tempfile._get_candidate_names())+'.png'
                new_image.save(os.path.join(result_dir, temp_name))
                if is_ajax:
                    result = ajax_response(True, temp_name)
                else:
                    result = redirect(
                        url_for("show_image_result", file_path=temp_name))
        shutil.rmtree(tmpdir)
        return result


@app.route("/show/<file_path>")
def show_image_result(file_path):
    return render_template('image_display.html', image=f"/file/{file_path}")


def ajax_response(status, msg):
    status_code = "ok" if status else "error"
    return json.dumps(dict(
        status=status_code,
        msg=msg,
    ))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    app.run()
