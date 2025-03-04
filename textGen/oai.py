import os
import importlib.util
import json
import openai
import numpy as np
import inspect
from typing import List, Dict, Union, Optional, Any  

class OAI:
    def __init__(self, api_keys_path: Optional[str] = None):
        """Initialize OpenAI API key dynamically, resolving paths correctly."""

        # Dynamically resolve the path of `oai.py`
        oai_directory = os.path.dirname(os.path.abspath(__file__))

        # Default to "../data/api_keys.py", ensuring it works regardless of execution location
        default_api_keys_path = os.path.abspath(os.path.join(oai_directory, "..", "data", "api_keys.py"))

        # Use provided path or fall back to detected path
        api_keys_path = api_keys_path or default_api_keys_path

        if not os.path.exists(api_keys_path):
            print(f"⚠️ Warning: API keys file not found: {api_keys_path}")
            self.api_key = None
        else:
            # Dynamically load API keys
            spec = importlib.util.spec_from_file_location("api_keys", api_keys_path)
            api_keys = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_keys)

            # Retrieve OpenAI API key safely
            self.api_key = getattr(api_keys, "OPENAI_API_KEY", None)

        if not self.api_key:
            print("⚠️ Warning: Missing 'OPENAI_API_KEY'. Some features may not work.")
        else:
            openai.api_key = self.api_key  # Set API key for OpenAI usage
            
        # Default settings.
        self.model = "gpt-4o"
        self.temperature = 0.7
        self.max_tokens = 4096

        # Dictionary to store available tools (mapping tool names to callables)
        self.available_tools: Dict[str, Any] = {}

    def chat_completion(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        message_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        available_tools: Optional[Dict[str, Any]] = None,
        integrate_tool_results: bool = True,
        **kwargs
    ) -> str:
        """Chat completion method with message history for context retention."""
        
        messages = message_history if message_history else []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        params = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **kwargs
        }
        if tools:
            params["tools"] = tools

        response = openai.ChatCompletion.create(**params)
        message = response['choices'][0]['message']

        # If no tool calls (or integration disabled), return the content.
        if not (hasattr(message, "tool_calls") and message.tool_calls and integrate_tool_results):
            return message.get('content') or ""
        
        # Execute each tool call.
        tool_results = []
        combined_available_tools = {}
        if self.available_tools:
            combined_available_tools.update(self.available_tools)
        if available_tools:
            combined_available_tools.update(available_tools)
        for tool_call in message.tool_calls:
            func_name = tool_call.get("name")
            if not func_name and isinstance(tool_call, dict) and "function" in tool_call:
                func_name = tool_call["function"].get("name")
            result = self.execute_tool(
                {"name": func_name, "arguments": tool_call.get("arguments", "{}")},
                combined_available_tools
            )
            tool_results.append(f"Tool '{func_name}' returned: {result}")

        print("Tool results:", tool_results)

        # Integrate tool outputs with the original assistant message.
        initial_content = message.get('content') or ""
        integration_instruction = "Please incorporate the following tool results into your final answer: " + " ".join(tool_results)
        messages.append({"role": "system", "content": integration_instruction})
        messages.append({"role": "assistant", "content": initial_content})

        # Make a final call to integrate the tool results.
        integration_params = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **kwargs
        }
        final_response = openai.ChatCompletion.create(**integration_params)
        return final_response['choices'][0]['message']['content']

    
    def reasoned_completion(
        self,
        user_prompt: str,
        message_history: Optional[List[Dict[str, str]]] = None,
        reasoning_effort: str = "medium",
        model: Optional[str] = "o3-mini"
    ) -> str:
        """Generate a response with an adjustable reasoning effort, excluding system messages."""
        
        messages = [
            msg for msg in message_history if msg["role"] != "system"
        ] if message_history else []
        
        messages.append({"role": "user", "content": user_prompt})

        params = {
            "model": model,
            "messages": messages,
            "reasoning_effort": reasoning_effort
        }
        response = openai.ChatCompletion.create(**params)
        return response['choices'][0]['message']['content']

    
    def vision_analysis(
        self,
        image_url: str,
        user_prompt: str = "Describe this image in detail.",
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Perform vision analysis while retaining message history."""
        try:
            messages = message_history if message_history else []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": "You are a helpful assistant with image analysis capabilities."})
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
            
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                **kwargs
            }
            response = openai.ChatCompletion.create(**params)
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error in vision analysis: {str(e)}"

    
    def structured_output(
        self,
        user_prompt: str,
        system_prompt: str = "Return the output in structured JSON format.",
        message_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Generate structured JSON output with message history included."""
        
        messages = message_history if message_history else []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = openai.ChatCompletion.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            response_format={"type": "json_object"},
            **kwargs
        )
        content = response['choices'][0]['message']['content']
        try:
            return json.loads(content)
        except Exception as e:
            return {"error": "JSON parsing failed", "detail": str(e)}

    def get_embeddings(self, texts: Union[str, List[str]], model: str = "text-embedding-3-small") -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        response = openai.Embedding.create(
            model=model,
            input=texts
        )
        embeddings = [item['embedding'] for item in response['data']]
        return np.array(embeddings, dtype='float32')

    def convert_function_to_tool(self, func) -> Dict:
        """
        Convert a Python function to a simple tool schema.
        The function's __doc__ is used as its description.
        """
        sig = inspect.signature(func)
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            properties[name] = {"type": "string", "description": ""}
            if param.default == inspect.Parameter.empty:
                required.append(name)
        tool = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
        }
        return tool

    def function_calling(
        self,
        user_prompt: str,
        tools: List[Dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        messages = [{"role": "user", "content": user_prompt}]
        params = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **kwargs
        }
        params["tools"] = tools
        response = openai.ChatCompletion.create(**params)
        message = response['choices'][0]['message']
        if hasattr(message, "tool_calls") and message.tool_calls:
            return message.tool_calls
        return message['content']

    def execute_tool(self, tool_call: Dict, tool_functions: Dict[str, Any]) -> Any:
        """
        Execute a tool call. 'tool_functions' maps function names to callables.
        'tool_call' should have keys 'name' and 'arguments' (a JSON string).
        """
        try:
            func_name = tool_call.get("name")
            args = json.loads(tool_call.get("arguments", "{}"))
            if func_name in tool_functions:
                return tool_functions[func_name](**args)
            return f"Tool {func_name} not found."
        except Exception as e:
            return f"Error executing tool {func_name}: {str(e)}"

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        quality: str = "standard",
        style: str = "vivid",
        model: str = "dall-e-3",
        **kwargs
    ) -> List[str]:
        response = openai.Image.create(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            quality=quality,
            style=style,
            **kwargs
        )
        return [item['url'] for item in response['data']]

    def transcribe_audio(
        self,
        audio_file_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0
    ) -> str:
        """Transcribes audio using OpenAI's Whisper model."""
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature
                )
            return response["text"]
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"

    def text_to_speech(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "mp3",
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """Converts text to speech using OpenAI's TTS and optionally saves output."""
        try:
            response = openai.Audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                speed=speed,
                response_format=response_format
            )
            
            if output_path:
                with open(output_path, "wb") as audio_file:
                    audio_file.write(response.content)
                return output_path
            
            return response.content
        except Exception as e:
            return f"Error in text-to-speech conversion: {str(e)}"
        
if __name__ == '__main__':
    import time
    import sound  # Ensure we use Pythonista's sound module for playback
    import webbrowser

    # Define the path to your api_keys.py file.
    api_keys_path = os.path.join(os.path.dirname(__file__), "../data/api_keys.py")

    # Initialize the wrapper.
    oai = OAI(api_keys_path)

    # Test Structured Output: Generate a product comparison in structured format.
    print("\n== Structured Output Test ==")
    structured = oai.structured_output(
        "Compare electric vs. gas vehicles on environmental impact, cost, and performance",
        system_prompt="You are a knowledgeable automotive analyst. Provide a detailed comparison."
    )
    print("Structured Output:", json.dumps(structured, indent=2))
    
    # Test Reasoned Completion: Analyze the comparison with more depth.
    print("\n== Reasoned Completion Test ==")
    reasoned_response = oai.reasoned_completion(
        f"Assess if this vehicle comparison is comprehensive and balanced. "
        f"What important factors might be missing? Comparison JSON: {structured}",
        reasoning_effort="medium"
    )
    print("Reasoned Response:", reasoned_response)
    
    # Test Chat Completion with Tool Integration: Generate supplementary information.
    def generate_buying_guide() -> str:
        """Generate a buying guide for vehicle selection."""
        return oai.chat_completion(
            f"Create a concise buying guide based on the vehicle comparison. "
            f"Include key considerations for a typical consumer. "
            f"Use the following analysis: {structured}."
        )
    
    tool_schema = oai.convert_function_to_tool(generate_buying_guide)
    tools = [tool_schema]
    available_tools = {"generate_buying_guide": generate_buying_guide}
    
    print("\n== Chat Completion with Tool Integration Test ==")
    integrated_reply = oai.chat_completion(
        "Can you create a helpful buying guide based on this vehicle comparison?",
        tools=tools,
        available_tools=available_tools
    )
    print("Integrated Chat Reply:", integrated_reply)
    
    # Test Image Generation: Generate a visual representation of modern transportation.
    print("\n== Image Generation Test ==")
    image_urls = oai.generate_image(
        "A visually striking digital concept art comparing electric and gas vehicles, "
        "representing sustainability, technology, and modern transportation.",
        size="1024x1024",
        n=1
    )
    print("Generated Image URL(s):", image_urls)
    
    # Open the image URL in Safari.
    print("\n== Open Image in Safari Test ==")
    if image_urls:
        webbrowser.open(image_urls[0])
    
    # Test Vision Analysis: Extract insights from the generated image.
    print("\n== Vision Analysis Test ==")
    vision_reply = oai.vision_analysis(
        image_urls[0],
        user_prompt="Describe this image in the context of transportation evolution and environmental impact."
    )
    print("Vision Analysis Reply:", vision_reply)
    
    # Test Text-to-Speech (TTS) and Audio Transcription (STT): Convert the vision response to speech and transcribe it back.
    print("\n== TTS and STT Integration Test ==")
    tts_text = vision_reply if vision_reply else "No integrated reply."
    tts_output_path = os.path.join(os.path.dirname(__file__), "tts_output.mp3")
    
    print("\n== Testing TTS ==")
    print(tts_output_path)
    tts_result = oai.text_to_speech(tts_text, output_path=tts_output_path)
    
    if isinstance(tts_result, str) and os.path.exists(tts_output_path):
        print("TTS Output File Saved:", tts_output_path)
        print("Playing the generated speech...")
        sound.play_effect(tts_output_path)
        
        # Wait for audio playback to complete before transcribing
        time.sleep(3)
        
        print("\n== Testing STT ==")
        transcript = oai.transcribe_audio(tts_output_path)
        print("Transcribed Text:", transcript)
    else:
        print("TTS failed to generate an audio file.")
