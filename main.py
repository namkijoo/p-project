import io
import flask
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import glob


app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((300,300)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
    return my_transforms(image_bytes).unsqueeze(0)

def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 정규화 해제하기
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()
    
#path = './test_image.jpg'


def get_prediction(image_bytes):
    class_names = ['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis']
    
    
    
    image = Image.open(io.BytesIO(image_bytes))


    
    image = transform_image(image)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        
        imshow(image.data[0], title='예측 결과: ' + class_names[preds[0]])

    return class_names[preds[0]]
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 이미지 바이트 데이터 받아오기
        file = request.files['image']
        
        image = file.read()
       

        # 분류 결과 확인 및 클라이언트에게 결과 반환
        class_name = get_prediction(image_bytes = image)
        print("결과:", {'class_name': class_name})
        return jsonify({'class_name': class_name})

if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=8)
    model.load_state_dict(torch.load('model/model_weights.pth', map_location=torch.device('cpu')))
    model.eval()

    
    app.run(host='0.0.0.0', port=8000, debug=True)