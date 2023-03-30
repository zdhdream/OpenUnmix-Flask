import sys

from flask import Flask
from flask import request
from openunmix import predict
from flask import jsonify
import torch
import torchaudio
import os

# # 获取当前文件的路径(绝对路径)
# app_path = os.path.abspath(os.path.dirname(__file__))
# # 将上级目录添加到sys.path中,以便Python解释器能够找到相关模块和包
# parent_dir = os.path.join(app_path, os.pardir)
# sys.path.append(parent_dir)


app = Flask(__name__)
app.debug = 'production'


@app.route('/modelinference', methods=['POST'])
def model_inference():
    audio_file = request.files['audio']
    print(audio_file.filename)
    sig, rate = torchaudio.load(audio_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimates = predict.separate(
        torch.as_tensor(sig).float(),
        rate=rate,
        device=device
    )
    music_dict_info = {}
    for target, estimate in estimates.items():
        audio = estimate.detach().cpu().numpy()[0]
        music_dict_info[target] = audio.tolist()

    return jsonify(music_dict_info)


@app.route("/modelInferenceBass", methods=['POST'])
def model_split_bass():
    pass


@app.route("/modelInferenceDrums", methods=['POST'])
def model_split_drums():
    pass


@app.route("/modelInferenceVocals", methods=['POST'])
def model_split_vocals():
    pass


@app.route("/modelInferenceOthers", methods=['POST'])
def model_split_others():
    pass


if __name__ == '__main__':
    app.run(debug=True, port=8111, host="0.0.0.0")
