from transformers import AutoTokenizer, CLIPProcessor, ViTFeatureExtractor, VisionTextDualEncoderModel


def load_korclip(model_path):
    model = VisionTextDualEncoderModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer 

