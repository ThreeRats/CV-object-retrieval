from flask import Flask, render_template, jsonify, request
from model.search import search_similar_image
from utils import Base64Encoding, Base64Decoding
from time import time

# your dataset path
dataset_path = 'E:/AllDownLoad/images/'
app = Flask(__name__)

@app.route('/')
def index():
    # print('[DEBUG] in index')
    return render_template('index.html')

@app.route('/sendImg', methods=['POST'])
def search_image():
    img_base_64 = request.form.get('img')
    original_img = Base64Decoding(img_base_64)
    # 使用模型查找相似的图片
    start_time = time()
    similar_images = search_similar_image(original_img, dataset_path)
    end_time = time()

    json_dict = {
        'status': 1,
        'result_img_list': [
            'data:image/jpeg;base64,' + Base64Encoding(image) for image in similar_images
        ],
        'time': '{} 毫秒'.format((end_time - start_time) * 1000, 0),
    }

    return jsonify(json_dict)

if __name__ == "__main__":
    app.run(port=5500)