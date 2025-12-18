from utils.models_utils.model_base import ModelBase

def construct_model_base(model_path: str, checkpoint:str) -> ModelBase:

    if 'qwen' in model_path.lower():
        from utils.models_utils.qwen_model import QwenModel
        return QwenModel(model_path, checkpoint=checkpoint)
    elif 'gemma' in model_path.lower():
        from utils.models_utils.gemma_model import GemmaModel
        return GemmaModel(model_path, checkpoint=checkpoint) 
    else:
        raise ValueError(f"Unknown model family: {model_path}")
