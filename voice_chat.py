import os
import sounddevice as sd
from TTS.api import TTS
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from typing import Optional
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


class Client:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.history = ChatHistory()
        self.model_name = "z-ai/glm-4.5-air:free"
        print("Using model:", self.model_name)

    def run(self, user_input):
        self.history.append(user_input)
        s = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            extra_headers={
                # "X-Title": "AIVC",
            },
        )
        print("time", time.monotonic()-s)
        # Extract the response text
        ai_response = response.choices[0].message.content
        self.history.append(ai_response, role="assistant")
        return ai_response


class ChatHistory(list):
    def __init__(self):
        self.append(
            "You are a helpful AI chat bot, your answers should brrief don't go into details too much", role="system")

    def append(self, message: str, role="user"):
        obj = {
            "role": role,
            "content": message,
        }
        return super().append(obj)


class Text2Speech:
    def __init__(self, use_transformers: bool = False, transformers_model: Optional[str] = None):
        self.use_transformers = use_transformers
        self.transformers_model = transformers_model or "espnet/kan-bayashi_ljspeech_vits"
        if self.use_transformers:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            self.processor = AutoProcessor.from_pretrained(
                self.transformers_model)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.transformers_model)
            self.torch = torch
        else:

            self.model = TTS("tts_models/en/vctk/vits").to(device)
            self.speaker = "p225"

    def text_to_speech(self, text):
        if self.use_transformers:
            # Generate speech using transformers model
            inputs = self.processor(text, return_tensors="pt")
            with self.torch.no_grad():
                output = self.model.generate(**inputs)
            wav = output.cpu().numpy().squeeze()
            wav = wav / np.max(np.abs(wav))
        else:
            wav = self.model.tts(text=text, speaker=self.speaker)
            wav = np.array(wav)
            wav = wav / np.max(np.abs(wav))

        # Play the audio
        sd.play(wav, samplerate=24000)
        sd.wait()


class VoiceChat:
    def __init__(self, use_voice=False, use_transformers_tts: bool = False, transformers_model: Optional[str] = None):
        load_dotenv()
        self.client = Client()
        if use_voice:
            self.tts = Text2Speech(
                use_transformers=use_transformers_tts, transformers_model=transformers_model,
            )
        print("Setup is done")
        self.use_voice = use_voice

    def chat(self):
        print("Voice Chat AI initialized. Type 'quit' to exit.")

        while True:
            user_input = input("\nYou: ")

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            ai_response = self.client.run(user_input).strip()
            print(f"\nAI: {ai_response}")
            if self.use_voice:
                self.tts.text_to_speech(ai_response)


if __name__ == "__main__":
    # Example: pass True to use transformers TTS, or False to use Coqui TTS

    chat = VoiceChat(use_voice=False, use_transformers_tts=False)
    chat.chat()
