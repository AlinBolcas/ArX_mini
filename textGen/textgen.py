"""
textgen.py

Unified TextGen v3.1: Advanced Integration Hub for OAI, Memory, RAG, and Tools.

This module defines the TextGen class which coordinates LLM calls via OAI,
long-term memory retrieval via RAG, and external tool integration.
Short-term memory messages are included in each call.
System context and long term contex are injected into the system message,
while additional user context ("contex") is appended to the user prompt.
Tool outputs that are too long are summarized via RAG.
"""

import os
import sys
import json

# Adjust path so that utils is loaded from one level up.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import Utils

# Import other modules by name using Utils.
OAI_module = Utils.import_file("oai.py")
Memory_module = Utils.import_file("memory.py")
RAG_module = Utils.import_file("rag.py")
Tools_module = Utils.import_file("tools.py")

# Extract the classes we need.
OAI = OAI_module.OAI
Memory = Memory_module.Memory
RAG = RAG_module.RAG
Tools = Tools_module.Tools


class TextGen:
    """
    Unified TextGen: Integration Hub for OAI, Memory, RAG, and Tools.

    Uses OAI for LLM completions with short-term memory, and injects additional context:
      - System contex (plus long term contex from memory) is added to the system message.
      - "Contex" is appended to the user prompt.
    Supports tool integration in chat_completion; long tool outputs are summarized via RAG.
    Temperature and max_tokens are exposed across all text generation methods.
    """
    def __init__(self, api_keys_path: str = None, short_term_limit: int = 8000,
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.oai = OAI(api_keys_path)
        self.memory = Memory(short_term_limit)
        self.rag = RAG(chunk_size, chunk_overlap)
        # Define self.embedding to reshape the OAI embeddings as needed.
        self.embedding = lambda text: self.oai.get_embeddings(text).reshape(-1)
        # Register available tools from the Tools module.
        self.tools = {}
        for attr_name in dir(Tools):
            if not attr_name.startswith("_"):
                attr = getattr(Tools, attr_name)
                if callable(attr):
                    self.tools[attr_name] = attr

    def _prepare_prompts(self, user_prompt: str, system_contex: str = None, contex: str = None) -> (str, str):
        """
        Prepare final system and user messages.
        
        - The system message is built from the base system_prompt (provided later) plus:
            * system_contex (if any) and
            * "Long term contex" retrieved from memory.
        - The user message is built from contex (if any) followed by the actual user prompt.
        """
        long_term_contex = self.retrieve_memory_context(user_prompt)
        final_system = ""
        if system_contex:
            final_system += system_contex + "\n"
        if long_term_contex:
            final_system += "Long term contex:\n" + long_term_contex
        final_user = ""
        if contex:
            final_user += contex + "\n"
        final_user += user_prompt
        return final_system, final_user

    def retrieve_memory_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve long term contex from memory via RAG.
        """
        insights = self.memory.retrieve_long_term()
        if insights:
            document = "\n".join(insights)
            self.rag.ingest_documents(document, self.embedding)
            return self.rag.retrieve_context(document, query, self.embedding, top_k=top_k)
        return ""

    def get_available_tools(self) -> list:
        """
        Retrieve and return a list of all available tools registered in TextGen.tools, 
        including their names and descriptions if available.

        Returns:
            list: A list of dictionaries, each containing:
                  - "name": The tool's function name.
                  - "description": The tool's docstring (if available), otherwise "No description provided."
        """
        tool_list = []

        if not hasattr(self, "tools"):
            return tool_list

        for tool_name, tool_func in self.tools.items():
            description = tool_func.__doc__.strip() if tool_func.__doc__ else "No description provided."
            tool_list.append({"name": tool_name, "description": description})

        return tool_list
        
    def chat_completion(self, user_prompt: str, system_prompt: str = "You are a helpful assistant.",
                        system_contex: str = None, contex: str = None,
                        tool_names: list = None, max_tool_word_count: int = 2000, tool_top_k: int = 5,
                        temperature: float = None, max_tokens: int = None,
                        store_interaction: bool = True, **kwargs) -> str:
        """
        Generate a chat completion response.
        
        - The final system message is constructed by combining the provided system_prompt,
          system_contex, and the "Long term contex" retrieved from memory.
        - The final user prompt is built by appending contex (if any) to the user_prompt.
        - Short-term memory is included as message history.
        - If tool_names are specified, each tool is executed. If a tool's output exceeds
          max_tool_word_count words, RAG is used to extract a concise summary (using tool_top_k).
          A follow-up LLM call then integrates these tool outputs.
        - Temperature and max_tokens are exposed.
        """
        message_history = self.memory.retrieve_short_term_formatted()
        final_system, final_user = self._prepare_prompts(user_prompt, system_contex, contex)
        # Combine the base system_prompt with our additional system context.
        combined_system = system_prompt + ("\n" + final_system if final_system else "")
        
        initial_response = self.oai.chat_completion(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature if temperature is not None else self.oai.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.oai.max_tokens,
            integrate_tool_results=False,
            **kwargs
        )
        integrated_response = initial_response
        if tool_names:
            tool_outputs = []
            for tool in tool_names:
                if tool in self.tools:
                    try:
                        result = self.tools[tool]()  # Execute tool without arguments.
                    except Exception as e:
                        result = f"Error executing tool '{tool}': {str(e)}"
                    # Convert result to string if necessary.
                    if not isinstance(result, str):
                        result_str = json.dumps(result)
                    else:
                        result_str = result
                    # Summarize tool output if too long.
                    if len(result_str.split()) > max_tool_word_count:
                        result_str = self.rag.retrieve_context(result_str, user_prompt, self.embedding, top_k=tool_top_k)
                    tool_outputs.append(f"Tool '{tool}' returned: {result_str}")
                else:
                    tool_outputs.append(f"Tool '{tool}' not found.")
            integration_prompt = (
                "Please incorporate the following tool outputs into your answer:\n" +
                "\n".join(tool_outputs) +
                "\n\nOriginal answer:\n" + initial_response
            )
            integrated_response = self.oai.chat_completion(
                user_prompt=integration_prompt,
                system_prompt=combined_system,
                message_history=message_history,
                temperature=temperature if temperature is not None else self.oai.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.oai.max_tokens,
                integrate_tool_results=False,
                **kwargs
            )
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, integrated_response)
        return integrated_response

    def structured_output(self, user_prompt: str, system_prompt: str = "Return the output in structured JSON format.",
                          system_contex: str = None, contex: str = None,
                          temperature: float = None, max_tokens: int = None,
                          store_interaction: bool = True, **kwargs) -> any:
        """
        Generate structured JSON output.
        """
        message_history = self.memory.retrieve_short_term_formatted()
        final_system, final_user = self._prepare_prompts(user_prompt, system_contex, contex)
        combined_system = system_prompt + ("\n" + final_system if final_system else "")
        response = self.oai.structured_output(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature if temperature is not None else self.oai.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.oai.max_tokens,
            **kwargs
        )
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, str(response))
        return response

    def vision_analysis(self, image_url: str, user_prompt: str,
                        system_prompt: str = "You are a helpful assistant with image analysis capabilities.",
                        system_contex: str = None, contex: str = None,
                        temperature: float = None, max_tokens: int = None,
                        store_interaction: bool = True, **kwargs) -> str:
        """
        Perform vision analysis on an image.
        """
        message_history = self.memory.retrieve_short_term_formatted()
        final_system, final_user = self._prepare_prompts(user_prompt, system_contex, contex)
        combined_system = system_prompt + ("\n" + final_system if final_system else "")
        response = self.oai.vision_analysis(
            image_url=image_url,
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature if temperature is not None else self.oai.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.oai.max_tokens,
            **kwargs
        )
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, response)
        return response
        
    def reasoned_completion(self, user_prompt: str, reasoning_effort: str = "medium",
                            system_contex: str = None, contex: str = None,
                            temperature: float = None, max_tokens: int = None,
                            store_interaction: bool = True, **kwargs) -> str:
        """
        Generate a reasoned completion.
        Combines background context with the user query, with clear markings.
        """
        message_history = self.memory.retrieve_short_term_formatted()
        final_system, final_user = self._prepare_prompts(user_prompt, system_contex, contex)
        # Combine context and query with clear markers.
        combined_prompt = ""
        if final_system:
            combined_prompt += "Background Context:\n" + final_system + "\n\n"
        combined_prompt += "User Query:\n" + final_user
        response = self.oai.reasoned_completion(
            user_prompt=combined_prompt,
            message_history=message_history,
            reasoning_effort=reasoning_effort,
            **kwargs
        )
        if store_interaction:
            self.memory.save_short_term("Reasoned Completion", user_prompt, response)
        return response


if __name__ == "__main__":
    # --- Demo of TextGen v3.1 Advanced Capabilities ---
    print("=== Unified TextGen v3.1 Demo ===\n")
    
    tg = TextGen(api_keys_path=None)
    tg.memory.reboot_memory()
    
    # Example 1: Simple chat completion.
    prompt1 = "What are three benefits of artificial intelligence in healthcare?"
    response1 = tg.chat_completion(prompt1, system_contex="You are a helpful technology consultant.")
    print("--- Chat Completion Response ---")
    print(response1)
    
    # Example 2: Chat completion with tool integration.
    prompt2 = "What are the primary features of this framework? Give me a list of the top 5 capabilities."
    response2 = tg.chat_completion(prompt2, system_contex="Analyze the codebase to provide accurate information.", tool_names=["get_codebase_snapshot"])
    print("\n--- Chat Completion with Tool Integration ---")
    print(response2)
    
    # Example 3: Structured output.
    prompt3 = "Generate a structured JSON list of 5 emerging technology trends for 2023."
    structured = tg.structured_output(prompt3)
    print("\n--- Structured Output ---")
    print(structured)
    
    # Example 4: Vision analysis.
    image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    prompt4 = "Describe this image in a short haiku."
    vision_reply = tg.vision_analysis(image_url, prompt4)
    print("\n--- Vision Analysis Response ---")
    print(vision_reply)
    
    # Example 5: Reasoned completion.
    prompt5 = "What are the main agent frameworks in AI? Explain the key methods an agent class would have when using LLM calls. Include approaches like ReAct, Reflexion, chain of thought, and tree of thought. Provide a concise overview of what's needed to build a modern agent architecture."
    reasoned_reply = tg.reasoned_completion(prompt5, system_contex="Focus on practical implementation details.")
    print("\n--- Reasoned Completion Response ---")
    print(reasoned_reply)
    
    # Example 6: Direct retrieval of long-term memory context.
    memory_context = tg.retrieve_memory_context("machine learning algorithms")
    print("\n--- Retrieved Long-Term Memory Context ---")
    print(memory_context)
    
    # Example 7: Practical test using file-based context.
    # Load external context files using Utils.
    project_info = Utils.load_file("project_info.md")
    technology_stack = Utils.load_file("technology_stack.md")
    
    if project_info is None:
        print("Error: 'project_info.md' file not found.")
    if technology_stack is None:
        print("Error: 'technology_stack.md' file not found.")
    
    if project_info and technology_stack:
        prompt7 = "Based on the project information and technology stack, suggest 5 potential enhancements or new features that would add the most value to this framework."
        # Use project_info as the system context and technology_stack as the user contex.
        response7 = tg.chat_completion(prompt7, system_contex=project_info, contex=technology_stack)
        print("\n--- Chat Completion with File Context ---")
        print(response7)
    
    print("\n=== Demo Completed ===")
