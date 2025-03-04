import openai
import os
import re
import tempfile
import importlib.util

# Define the path to your api_keys.py file.
api_keys_path = os.path.join(os.path.dirname(__file__), "../data/api_keys.py")
print("API Keys path:", api_keys_path)

# Use importlib.util to load the api_keys module dynamically.
spec = importlib.util.spec_from_file_location("api_keys", api_keys_path)
api_keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_keys)

# Load openai_docs.md and pythonista_docs.md as text.
openai_docs_path = os.path.join(os.path.dirname(__file__), "../data/openai_docs.md")
pythonista_docs_path = os.path.join(os.path.dirname(__file__), "../data/pythonista_docs.md")

with open(openai_docs_path, "r", encoding="utf-8") as f:
    OPENAI_DOCS = f.read()

with open(pythonista_docs_path, "r", encoding="utf-8") as f:
    PYTHONISTA_DOCS = f.read()

class Factory:
    def __init__(self, openai_api_key: str = None):
        # Use provided API key or fetch from environment variable
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable.")
        openai.api_key = self.openai_api_key
        # Use the openai module directly as our client
        self.client = openai

    def parse_code_response(self, content: str) -> str:
        """
        Parse the given markdown content and extract the first Python code block, if present.
        If no code block is found, return the original content.
        """
        match = re.search(r"```python(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return content.strip()

    def merge_code_snippets(self, content: str) -> str:
        """
        Extract all Python code blocks from the given markdown content and merge them into one Python file.
        """
        code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
        if code_blocks:
            merged = "\n\n".join(block.strip() for block in code_blocks)
            return merged
        else:
            return content.strip()

    def planner(self, context: str) -> str:
        """
        Generate a detailed plan based on the provided idea.
        """
        system_prompt = "You are a world class software engineer, planner, visionary and expert software supervisor for code generation."
        user_prompt = (f"Based on the following idea, think step by step and lay out a detailed plan as a technical project desciption that outlines the necessary steps, modules, functions, and structure for the final Python script."
            f"Idea:\n{context}")
        response = self.client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()

    def coder(self, plan: str) -> str:
        """
        Generate functioning Python code based on the plan.
        The response is expected to include one or more code blocks (e.g., for multiple files).
        """
        system_prompt = "You are a world class senior software engineer who carefully writes flawless code with immaculate accuracy and attention to detail, commenting, logging, error catching, and always writing code which executes based on provided guidelines and documentations."
        user_prompt = (f"Based on the following plan, generate fully functioning Python code that implements the described idea. Ensure the code is well-structured, commented, and executable. "
            "Return the response as markdown with each file contained in a Python code block (marked with ```python ... ```)."
            f"Plan:\n{plan}"
            f"You also have access to the correct OpenAI latest syntax here: {OPENAI_DOCS}")
        response = self.client.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": (user_prompt)}
            ],
            temperature=0.6,
        )
        return response['choices'][0]['message']['content'].strip()

    def supervisor(self, code: str) -> str:
        """
        Provide constructive criticisms and suggestions for improvement on the generated code.
        """
        user_prompt = (
            f"Review the following Python code and provide constructive notes, suggestions for improvements, "
            f"and highlight any potential pitfalls.\n\nCode:\n{code}"
            f"You also have access to the correct OpenAI latest syntax here: {OPENAI_DOCS}"
        )
        response = self.client.ChatCompletion.create(
            model="o3-mini",
            reasoning_effort="medium",
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        return response['choices'][0]['message']['content'].strip()

    def doc(self, plan: str, code: str) -> str:
        """
        Generate markdown documentation for the final code based on the plan.
        """
        system_prompt = "You are a world class technical documentation writer. You write flawless github expert level consistent documentation to the provided prompt and final code."
        user_prompt = ("Based on the following plan and final code, generate comprehensive markdown documentation that explains the purpose, usage, and inner workings of the code."
            f"Plan:\n{plan}\n\nCode:\n{code}")
        response = self.client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )
        return response['choices'][0]['message']['content'].strip()

    def save_file(self, filename: str, content: str):
        """
        Parse the response to extract all Python code blocks, merge them into one file, and save it.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved file: {filename}")

    def test_code(self, code: str) -> str:
        """
        Execute the generated code in a pseudo-sandboxed environment and return its output.

        This implementation runs the code within a separate thread using a ThreadPoolExecutor,
        and captures stdout and stderr via StringIO. A timeout is enforced to prevent indefinitely
        blocking code. Note: This method does not offer full process isolation.
        """
        import io
        import contextlib
        import concurrent.futures
        import traceback

        def run_code():
            output_buffer = io.StringIO()
            local_globals = {}  # A fresh namespace for execution
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

    def run_factory(self, idea_prompt: str, retries: int = 2):
        """
        Run the full factory pipeline with an agentic loop:
          1. Planning
          2. Coding (generating a multi-file markdown response)
          3. Merging code snippets into one Python file
          4. Supervisor evaluation
          5. Documentation
          6. Saving files and sandbox testing

        If testing reveals errors, the pipeline revises the plan based on the error and supervisor feedback
        and retries up to the specified number of iterations.
        """
        current_idea = idea_prompt
        iteration = 0

        while iteration <= retries:
            print(f"Iteration {iteration}: Planning...")
            plan = self.planner(current_idea)
            print("Plan:\n", plan, "\n")

            print("Coding...")
            code_response = self.coder(plan)
            print("Generated Code Response:\n", code_response, "\n")

            merged_code = self.merge_code_snippets(code_response)
            print("Merged Code:\n", merged_code, "\n")
            self.save_file("generated_code.py", merged_code)

            print("Supervisor Evaluation...")
            supervision = self.supervisor(merged_code)
            print("Supervisor Feedback:\n", supervision, "\n")

            print("Testing Code in Sandbox...")
            test_result = self.test_code(merged_code)
            print("Test Result:\n", test_result, "\n")

            # If testing does not produce an error, break the loop.
            if "Error during code execution" not in test_result:
                break

            # Otherwise, prepare a revised idea prompt incorporating error info and supervisor feedback.
            current_idea = (
                "The code produced the following error when executed:\n" + test_result +
                "\nSupervisor feedback was:\n" + supervision +
                "\nPlease revise the plan to fix these issues and produce corrected code. "
                "Here is the current code:\n" + merged_code
            )
            iteration += 1

        print("Documentation...")
        documentation = self.doc(idea_prompt, merged_code)
        print("Documentation:\n", documentation, "\n")
        
        # Save the final merged code and documentation.
        self.save_file("generated_code.py", merged_code)
        self.save_file("documentation.md", documentation)

        return {
            "context": idea_prompt,
            "final_idea": current_idea,
            "plan": plan,
            "code": merged_code,
            "supervision": supervision,
            "documentation": documentation,
            "test_result": test_result,
            "iterations": iteration,
        }

if __name__ == '__main__':
    # Dictionary of project ideas (kept general and neutral)
    project_ideas = {
        "1": "Language Translation Tool: Develop a program that captures microphone input, transcribes it using OpenAI, translates the text into another language, and converts the translated text back to speech for audio playback.",
        "2": "Expense Tracker: Create an application that allows users to input expenses, categorize them, and generate monthly reports to help manage finances.",
        "3": "Photo Album with Slideshow: Build a program that organizes photos into albums, adds captions, and displays them in a slideshow format with transition effects.",
        "4": "Inventory Management System: Design a system to catalog items, including details like purchase date, value, and location, aiding in organization.",
        "5": "Digital Diary with Search: Implement a digital diary where users can write entries and later search for specific entries by date or keywords.",
        "6": "Unit Converter Tool: Create a utility that converts between various units of measurement, such as length, weight, and temperature.",
        "7": "Family Tree Generator: Develop an application that helps users input family member information and generates a visual family tree diagram.",
        "8": "Music Playlist Organizer: Build a program that allows users to organize music files into playlists, edit metadata, and play songs directly.",
        "9": "Book Collection Manager: Create a system for users to catalog their book collections, track which books they've read, and generate reading statistics.",
        "10": "Fitness Tracker: Design an application where users can log exercise routines, monitor progress over time, and receive suggestions for improvement."
    }

    # Display the project options
    print("Choose a project to generate:")
    for key, description in project_ideas.items():
        print(f"{key}. {description.split(':')[0]}")

    # Get user input
    choice = input("\nEnter the number of the project you want to generate: ")

    # Retrieve the selected project idea
    idea_prompt = project_ideas.get(choice)

    if idea_prompt:
        print("\nSelected Project Idea:\n", idea_prompt, "\n")
        factory = Factory(api_keys.OPENAI_API_KEY)
        results = factory.run_factory(idea_prompt + PYTHONISTA_DOCS, retries=2)
        print("Factory process completed.")
    else:
        print("Invalid selection. Please restart and choose a valid option.")
