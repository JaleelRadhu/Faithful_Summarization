import requests
import json
import os

class ChatGenerator:
    """
    A client for generating chat completions using a vLLM server.
    This version is compatible with the multiprocessing setup in main_new.py.
    """
    def __init__(self, model_name: str, system_prompt: str):
        """
        Initializes the ChatGenerator client.

        Args:
            model_name (str): The name of the model being served by vLLM.
            system_prompt (str): The system prompt to use for all generations.
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        # Get API URL from environment variable, with a sensible default.
        # This makes the client more flexible without changing the code.
        self.api_url = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
        self.headers = {"Content-Type": "application/json"}

    def __call__(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, **kwargs) -> list:
        """
        Generates a response from the vLLM server and returns it in the
        pipeline-compatible format: [{"generated_text": "..."}]
        
        **kwargs is included to accept additional arguments like 'return_full_text'
        that might be passed by other parts of the pipeline but are not used by the vLLM API.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        # --- Verification Logging ---
        # Print a message before the network call. Include the worker's process ID.
        # print(f"\n[Worker PID: {os.getpid()}] ➡️  Sending request to vLLM server...")
        response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
        # Print a message after the network call completes successfully.
        # print(f"\n[Worker PID: {os.getpid()}] ⬅️  Received response from vLLM server.")
        response.raise_for_status()  # Raise an exception for bad status codes
        completion = response.json()['choices'][0]['message']['content']
        # Return in the same format as the original ChatGenerator to ensure compatibility
        return [{"generated_text": completion}]