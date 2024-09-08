import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm
from PIL import Image

# 1. 손상된 이미지 건너뛰기 함수 정의
def verify_image(path):
    """ 이미지가 손상되었는지 확인하는 함수 """
    try:
        img = Image.open(path)
        img.verify()  # 이미지가 손상되었는지 검증
        return True
    except (IOError, SyntaxError):
        print(f"Skipping corrupted image: {path}")
        return False

# 2. 모델 불러오기 (사전 학습된 MobileNetV4)
def load_pretrained_model(model_name='mobilenetv4_hybrid_large.e600_r384_in1k', num_classes=2):
    # 사전 학습된 모델 불러오기
    model = timm.create_model(model_name, pretrained=True)
    
    # 마지막 레이어를 새로운 데이터셋에 맞게 수정 (예: 2개 클래스)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    return model

# 3. 전이 학습을 위한 모델 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward 패스
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward 패스 및 옵티마이저 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 4. 모델 평가 함수
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# 5. 전체 코드 실행
if __name__ == "__main__":
    # 1. 모델 로드 및 GPU 설정
    num_classes = 2  # 분류할 클래스 수 (여기서는 이진 분류 예시)
    model = load_pretrained_model(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 2. 데이터 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNet 입력 크기에 맞춤
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 학습 및 검증 데이터셋 로드 (ImageFolder, 손상된 이미지 건너뛰기 적용)
    train_dataset = datasets.ImageFolder(root='./datasets/train', transform=transform, is_valid_file=verify_image)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = datasets.ImageFolder(root='./datasets/val', transform=transform, is_valid_file=verify_image)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 4. 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # 6. 모델 평가
    evaluate_model(model, val_loader)
