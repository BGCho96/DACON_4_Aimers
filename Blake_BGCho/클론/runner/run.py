# runner/run.py
import yaml
import os
from models.yolo_base import load_model, predict

def load_config(path="runner/classification_test_train.yaml"): # 어떤 yaml 참조할지 명시
    with open(path, 'r') as f:
        return yaml.safe_load(f)
def load_model_from_file(model_config):
    path = model_config['file']  # e.g. models/vision/cnn_classifier.py
    class_name = model_config['class']
    args = model_config.get('args', {})

    spec = importlib.util.spec_from_file_location("custom_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, class_name)
    return model_class(**args)
def train_classification_model(config,mode):
    from datasets.classification_loader import create_classification_dataloader
    from models.toy_empty_model import ToyClassifier  # 본인이 만든 모델 import
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 데이터 로더
    train_loader, class_names = create_classification_dataloader(config, mode)

    # 모델 초기화
    model = load_model_from_file(config['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 손실함수, 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train'].get('lr', 0.001))

    # 학습 루프
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{config['train']['epochs']}: Loss={running_loss:.4f}, Acc={correct/total:.4f}")

    # 저장
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"✅ 모델 저장 완료: {os.path.join(save_dir, 'model.pt')}")
def main():
    # Load configs as setted
    config = load_config()
    

    # predict
    if config['experiment']['mode'] == 'predict':
        model = load_model(config['model']['name'])
        result = predict(model, config['input']['image_path'])

        result.show()
        save_dir = config['output']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        result.save(filename=os.path.join(save_dir, "result.jpg"))

    elif config['experiment']['mode'] == 'train' or config['experiment']['mode'] == 'all':
        train_classification_model(config,config['experiment']['mode'])

if __name__ == "__main__":
    main()