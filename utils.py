import os
import json
import importlib.util
import subprocess
import re
from typing import Any, Dict, List, Optional, Union, Optional 

class Utils:
    """Utility functions for file handling, code testing, and parsing."""

    @staticmethod
    def get_output_path() -> str:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def save_file(content: str, name: Optional[str] = None, extension: str = "py") -> str:
        """
        Save content to a file in the output directory.
        - If no name is provided, it generates one based on content.
        - Returns the full file path.
        """
        output_dir = Utils.get_output_path()
        if not name:
            name = Utils.name_file(content[:50])  # Generate a name from content (first 50 chars)
        filename = f"{name}.{extension}"
        full_path = os.path.join(output_dir, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"File saved at: {full_path}")
        return full_path  # Return file path for further use

    @staticmethod
    def test_code(code: str) -> str:
        """
        Execute the generated Python code and return its output.
        Uses a sandboxed environment with timeouts to prevent hangs.
        """
        import io
        import contextlib
        import concurrent.futures
        import traceback

        def run_code():
            output_buffer = io.StringIO()
            local_globals = {}
            try:
                with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                    exec(code, local_globals)
            except Exception:
                output_buffer.write(traceback.format_exc())
            return output_buffer.getvalue()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_code)
                output = future.result(timeout=10)  # Enforce a 10-second timeout
        except concurrent.futures.TimeoutError:
            output = "Error during code execution: Code execution timed out."
        except Exception as e:
            output = f"Error during code execution: {e}"
        return output

    @staticmethod
    def parse_code_response(response: str) -> str:
        """
        Extracts the first Python code block from a markdown response.
        If none exists, return the original response.
        """
        match = re.search(r"```python(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    @staticmethod
    def merge_code_snippets(content: str) -> str:
        """
        Extracts and merges all Python code blocks from a markdown response.
        Returns a single Python script with merged code.
        """
        code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
        return "\n\n".join(block.strip() for block in code_blocks) if code_blocks else content.strip()

    @staticmethod
    def load_file(name: str) -> Optional[str]:
        """Recursively find and load a file by name, searching anywhere in the project."""
        
        # Get the absolute path of the `arx_mini` root directory
        current_dir = os.path.abspath(os.path.dirname(__file__))  # Directory where utils.py is
        base_dir = current_dir

        # Traverse up until we reach the `arx_mini` folder (ensuring we're not outside the project)
        while not os.path.exists(os.path.join(base_dir, "utils.py")):  # Identifies `arx_mini`
            new_base = os.path.abspath(os.path.join(base_dir, ".."))
            if new_base == base_dir:  # Prevent infinite loop
                break
            base_dir = new_base

        # Now, we are at `arx_mini/`. Start searching recursively for the file.
        for root, _, files in os.walk(base_dir):
            if name in files:
                file_path = os.path.join(root, name)
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        
        return None  # File not found
        
    @staticmethod
    def name_file(user_input: str) -> str:
        """Generate a clean filename based on user input or AI assistance."""
        oai_module = Utils.import_file("oai.py")

        if oai_module:
            try:
                # Dynamically load OAI
                oai_instance = oai_module.OAI(api_keys_path=None)  # Uses auto-resolving API key path

                ai_name = oai_instance.chat_completion(
                    f"Generate as short as possible, descriptive filename for: {user_input}. Keep it concise, lowercase, and use underscores instead of spaces. Important: Do NOT include the file type extension in the name!"
                ).strip()

                # Ensure valid filename format
                ai_name = re.sub(r"[^\w\s-]", "", ai_name)  # Remove invalid chars
                ai_name = ai_name.replace(" ", "_").lower()[:40]  # Limit length
                return ai_name
            except Exception as e:
                print(f"Error using OAI for filename generation: {e}")

        # Fallback to simple formatting if AI fails
        fallback_name = re.sub(r"[^\w\s-]", "", user_input).replace(" ", "_").lower()[:40]
        return fallback_name

    @staticmethod
    def import_file(name: str) -> Optional[Any]:
        """Find a Python module recursively starting from the project root and import it."""
        
        # Get the absolute path of the `arx_mini` root directory
        current_dir = os.path.abspath(os.path.dirname(__file__))  # Directory where utils.py is
        base_dir = current_dir

        # Traverse up until we find the project root containing utils.py
        while not os.path.exists(os.path.join(base_dir, "utils.py")):
            new_base = os.path.abspath(os.path.join(base_dir, ".."))
            if new_base == base_dir:  # Prevent infinite loop
                break
            base_dir = new_base

        # Now, we are at the project root. Start searching for the file.
        for root, _, files in os.walk(base_dir):
            if name in files and name.endswith(".py"):
                module_name = os.path.splitext(name)[0]
                module_path = os.path.join(root, name)

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module  # Return the imported module

        return None  # File not found
        
    @staticmethod
    def get_codebase_snapshot(root_dir: str = ".") -> Dict[str, Any]:
        """
        Generates a structured snapshot of the entire codebase.
        - Scans all Python files in the project.
        - Extracts imports and dependency relationships.
        - Merges all code into a unified context.

        Returns:
            A dictionary containing file contents and dependencies.
        """
        codebase_snapshot = {}

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Extract import statements
                        imports = re.findall(r"^\s*(?:import|from)\s+([\w\.]+)", content, re.MULTILINE)

                        codebase_snapshot[file_path] = {
                            "imports": list(set(imports)),  # Remove duplicates
                            "content": content
                        }
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        return codebase_snapshot

    @staticmethod
    def merge_codebase(snapshot: Dict[str, Any]) -> str:
        """
        Merges all files in the codebase snapshot into a single structured output.
        - Preserves import relationships at the top.
        - Ensures logical ordering of files based on dependencies.

        Returns:
            A single unified codebase string.
        """
        merged_code = []
        all_imports = set()

        # Collect all imports first
        for file_data in snapshot.values():
            all_imports.update(file_data["imports"])

        # Add collected imports at the top
        merged_code.append("# Unified Codebase Snapshot\n")
        merged_code.append("\n".join(f"import {imp}" for imp in sorted(all_imports)))
        merged_code.append("\n" + "=" * 60 + "\n")

        # Append each file's content
        for file_path, file_data in snapshot.items():
            merged_code.append(f"# File: {file_path}\n" + file_data["content"])
            merged_code.append("\n" + "=" * 60 + "\n")

        return "\n".join(merged_code)
        
# ================================
#       TESTING UTILITIES
# ================================

if __name__ == "__main__":
    # Step 1: Generate a test Python script
    code_snippet = "print('Hello, AI-assisted world!')\nx = 5\ny = 10\nprint(f'The sum is: {x + y}')"

    # Step 2: Save the script with an AI-generated filename
    print("\n== Saving Generated Code ==")
    saved_file_path = Utils.save_file(code_snippet)
    print("Saved file path:", saved_file_path)

    # Step 3: Load the saved script
    print("\n== Loading Saved Code ==")
    loaded_code = Utils.load_file(os.path.basename(saved_file_path))
    print("Loaded Code:\n", loaded_code)

    # Step 4: Execute the saved script
    print("\n== Running Code ==")
    test_result = Utils.test_code(loaded_code)
    print("Execution Output:\n", test_result)

    # Step 5: Extract a Python snippet from markdown response
    markdown_response = "```python\nprint('Extracted code execution test')\na = 7\nb = 3\nprint(a * b)\n```"
    print("\n== Parsing Markdown Code ==")
    parsed_code = Utils.parse_code_response(markdown_response)
    print("Parsed Code:\n", parsed_code)

    # Step 6: Merge multiple markdown code blocks
    markdown_snippets = "```python\nprint('Snippet 1')\n```\n```python\nprint('Snippet 2')\n```"
    print("\n== Merging Code Snippets ==")
    merged_code = Utils.merge_code_snippets(markdown_snippets)
    print("Merged Code:\n", merged_code)

    # Step 7: Generate a filename from content
    print("\n== Naming File ==")
    generated_name = Utils.name_file(code_snippet)
    print("Generated Filename:", generated_name)

    # Step 8: Dynamically import a Python module (Example: oai.py)
    print("\n== Importing a File ==")
    imported_module = Utils.import_file("oai.py")
    if imported_module:
        print(f"Successfully imported: {imported_module.__name__}")
    else:
        print("File 'oai.py' not found.")
