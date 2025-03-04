import os
import sys
import json

# Ensure correct path loading like in TextGen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import Utils

# Import TextGen properly
TextGen_module = Utils.import_file("textgen.py")
TextGen = TextGen_module.TextGen

class AgentGen(TextGen):
    """
    A simplified AgentGen with a single base loop and tool selection.
    """

    def __init__(self, api_keys_path: str = None, short_term_limit: int = 8000,
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(api_keys_path, short_term_limit, chunk_size, chunk_overlap)
        
    def select_best_tools(self, user_prompt: str, top_k: int = 3) -> list:
        """
        Selects the most relevant tools based on the user prompt.
        """
        import re  # Import regex for safe JSON extraction
        
        # Fetch available tools
        available_tools = self.get_available_tools()
        if not available_tools or not isinstance(available_tools, list):
            return []

        # Ensure tools have descriptions
        if all(isinstance(tool, str) for tool in available_tools):
            tool_descriptions = "\n".join(
                [f"- **{tool}**: No description available." for tool in available_tools]
            )
        elif all(isinstance(tool, dict) and "name" in tool and "description" in tool for tool in available_tools):
            tool_descriptions = "\n".join(
                [f"- **{tool['name']}**: {tool['description']}" for tool in available_tools]
            )
        else:
            return []

        # LLM Prompt
        tool_selection_prompt = (
            f"📌 TASK: Select the best tools for the following request:\n\n"
            f"🔹 **User Request**: {user_prompt}\n\n"
            f"🔧 **Available Tools**:\n{tool_descriptions}\n\n"
            f"🎯 **Instruction**: Choose the **{top_k} most relevant tools** based on their descriptions.\n"
            f"Return the response **strictly in JSON format** as:\n"
            f"`{{\"tools\": [\"ToolName1\", \"ToolName2\", ...]}}`."
        )

        # LLM Call
        raw_response = self.structured_output(
            user_prompt=tool_selection_prompt,
            system_prompt="Analyze the provided tools and return the most relevant ones **strictly in JSON format** as {'tools': [...]}.",
        )

        # Safe JSON Parsing
        try:
            if isinstance(raw_response, str):
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    parsed_response = json.loads(json_text)
                else:
                    raise ValueError("LLM response does not contain valid JSON.")
            elif isinstance(raw_response, dict):
                parsed_response = raw_response
            else:
                raise TypeError("Unexpected LLM response format.")

            # Extract tools list safely
            selected_tools = parsed_response.get("tools", []) if isinstance(parsed_response, dict) else []
            if not isinstance(selected_tools, list):
                raise ValueError("Extracted 'tools' is not a list.")
        except (json.JSONDecodeError, TypeError, ValueError):
            selected_tools = []

        return selected_tools
        
    ### BASE LOOP ###
    def base_loop(self, user_prompt: str, system_prompt: str = None, temperature: float = None,
                  max_tokens: int = None, contex: str = None, system_contex: str = None,
                  max_depth: int = 3, verbose: bool = True, tool_names: list = None) -> str:
        """
        Iteratively executes tools until an optimal response is achieved.
        """
        response = ""
        for i in range(max_depth):
            tool_names = tool_names or self.select_best_tools(user_prompt)

            response = self.chat_completion(
                user_prompt=f"🔄 REFINE ITERATIVELY USING TOOLS FOR:\n{user_prompt}\n\n"
                            f"💡 CURRENT OUTPUT:\n{response}\n"
                            "Indicate 'FINAL RESPONSE:' when the answer is complete.",
                system_prompt=system_prompt or "Use available tools when optimal to refine the response. "
                                               "Stop when the solution is optimal and explicitly state 'FINAL RESPONSE:'.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex,
                tool_names=tool_names
            )

            if verbose:
                print(f"🔄 Base Loop Iteration {i+1} Response:\n{response}\n")

            if "FINAL RESPONSE:" in response:
                break

        return response


if __name__ == "__main__":
    print("\n🚀 Running AgentGen Simple Chat...\n")

    # Initialize AgentGen
    ag = AgentGen(api_keys_path=None)

    # Load context files using Utils
    about_alin = Utils.load_file("about_Alin.md") or ""
    about_arvolve = Utils.load_file("about_Arvolve.md") or ""
    style_alin = Utils.load_file("style_Alin.md") or ""
    style_arvolve = Utils.load_file("style_Arvolve.md") or ""

    # Merge context and system context
    contex = about_alin + "\n" + about_arvolve
    system_contex = style_alin + "\n" + style_arvolve

    print("💬 Start chatting with the AI (type 'bye' to exit).")
    
    while True:
        # Print available tools
        available_tools = ag.get_available_tools()
        tool_names = [tool["name"] if isinstance(tool, dict) else tool for tool in available_tools]

        print("\n🛠️ Available Tools: " + ", ".join(tool_names) if tool_names else "No tools available.")
        
        # Let user choose tools
        chosen_tools = input("\n⚙️ Select tools to use (comma-separated, or press Enter to auto-select): ").strip()
        
        # Convert user input into a tool list
        tool_list = [tool.strip() for tool in chosen_tools.split(",") if tool.strip()] if chosen_tools else []

        user_input = input("\n👤 You: ")

        if user_input.lower() == "bye":
            print("👋 Exiting chat. Goodbye!")
            break

        # Run base loop with user input and selected tools
        try:
            response = ag.base_loop(
                user_input, contex=contex, system_contex=system_contex, verbose=True,
                tool_names=tool_list if tool_list else ag.select_best_tools(user_input)
            )
            print(f"🤖 AI:\n{response}\n")
        except Exception as e:
            print(f"❌ Error: {e}")

        if user_input.lower() == "bye":
            print("👋 Exiting chat. Goodbye!")
            break
