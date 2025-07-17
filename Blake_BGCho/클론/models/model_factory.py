import importlib.util

def load_model_from_file(path: str, class_name: str, args: dict):
    # .py 경로에서 모듈 import
    spec = importlib.util.spec_from_file_location("custom_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 클래스 불러오기
    model_class = getattr(module, class_name)

    # 인자로 모델 인스턴스 생성
    return model_class(**args)