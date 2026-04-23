import os
from groq import Groq
from dotenv import load_dotenv
from httpx import stream

load_dotenv()

class AIModel:
    def __init__(self,api_key=None,model=None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.model = model or os.getenv('GROQ_MODEL')

        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        self.client = Groq(api_key=self.api_key)

    def chat(self,message,system_prompt=None):
        messages=[]

        if system_prompt:
            messages.append({
                "role":"system",
                "content":system_prompt
            })

        messages.append({
            "role":"user",
            "content":message
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Groq API Error: {str(e)}")


    def chat_with_history(self,messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Groq API Error: {str(e)}")

    def chat_stream(self,messages):
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise Exception(f"Groq API Error: {str(e)}")


    def get_mode_info(self):
        return {
            'model':self.model,
            'provider':'Groq',
            'max_tokens':2000
        }