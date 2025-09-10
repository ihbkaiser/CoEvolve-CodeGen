from models.Gemini import Gemini
from models.OpenAI import ChatGPT
from models.OpenAI import GPT4
from models.OpenAI import GPT4oMini

class ModelFactory:
    @staticmethod
    def get_model_class(model_name):
        if model_name == "Gemini":
            return Gemini
        elif model_name == "ChatGPT":
            return ChatGPT
        elif model_name == "GPT4":
            return GPT4
        elif model_name == "GPT4oMini":
            return GPT4oMini
        else:
            raise Exception(f"Unknown model name {model_name}")
