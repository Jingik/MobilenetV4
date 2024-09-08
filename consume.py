import torch
from urllib.request import urlopen
from PIL import Image
import timm

# 1. 모델 불러오기
def load_model(model_name='mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True):
    # timm 라이브러리에서 MobileNetV4 모델 불러오기
    model = timm.create_model(model_name, pretrained=pretrained)
    
    # 추론 모드로 전환
    model.eval()
    return model

# 2. 이미지 전처리
def preprocess_image_from_url(image_url):
    # URL로부터 이미지 불러오기
    img = Image.open(urlopen(image_url))
    
    # timm 모델에 맞는 전처리 설정
    model = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    # 전처리 적용 및 배치 차원 추가 (1, C, H, W)
    input_image = transform(img).unsqueeze(0)
    
    return input_image

# 3. 추론 실행
def predict(model, input_image):
    # 추론 실행 (gradient 계산하지 않음)
    with torch.no_grad():
        output = model(input_image)
    
    # Softmax 함수로 확률 계산
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # 상위 5개 클래스 예측 및 확률 계산
    top5_probabilities, top5_class_indices = torch.topk(probabilities, k=5)
    
    return top5_class_indices, top5_probabilities

# 4. 전체 코드 실행
if __name__ == "__main__":
    # 이미지 URL 설정
    image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'

    # 모델 불러오기
    model = load_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True)
    
    # 이미지 전처리
    input_image = preprocess_image_from_url(image_url)
    
    # 추론 실행
    top5_class_indices, top5_probabilities = predict(model, input_image)
    
    # 결과 출력
    print("Top-5 Predicted Classes and Probabilities:")
    for i, (class_idx, prob) in enumerate(zip(top5_class_indices, top5_probabilities)):
        print(f"{i+1}: Class {class_idx.item()} with probability {prob.item() * 100:.2f}%")
