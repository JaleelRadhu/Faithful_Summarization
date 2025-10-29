import json
import os
import requests
import time
import sys

class ChatGenerator:
    """
    A client for generating chat completions using a vLLM server.
    This version is compatible with the multiprocessing setup in main_new.py.
    """
    def __init__(self, model_name: str, system_prompt: str, vllm_api_url: str = None, request_timeout: int = 120, max_retries: int = 3):
        """
        Initializes the ChatGenerator client.

        Args:
            model_name (str): The name of the model being served by vLLM.
            system_prompt (str): The system prompt to use for all generations.
            vllm_api_url (str): The URL for the vLLM API endpoint.
            request_timeout (int): Timeout for the API request in seconds.
            max_retries (int): Maximum number of retries on timeout.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.api_url = vllm_api_url or os.environ.get("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
        self.timeout = request_timeout
        self.max_retries = max_retries
        # Use a session for connection pooling and performance
        self.session = requests.Session()

    def __call__(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, **kwargs) -> list:
        """
        Generates a response from the vLLM server with retry logic and returns it in the
        pipeline-compatible format: [{"generated_text": "..."}].
        
        **kwargs is included to accept additional arguments like 'return_full_text'
        that might be passed by other parts of the pipeline but are not used by the vLLM API.
        """
        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "model": self.model_name,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        
        # Use the requests library to make the API call.
        # Without a timeout, it will wait indefinitely for a response.
        response = self.session.post(self.api_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        completion = response.json()['choices'][0]['message']['content']

        # Return in the same format as the original ChatGenerator to ensure compatibility
        return [{"generated_text": completion}]