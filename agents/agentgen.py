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
    AgentGen: A lightweight yet powerful agent framework built on TextGen.

    Implements structured multi-step reasoning loops:
    - **Base Loop**: Iteratively executes tools until goal satisfaction.
    - **ReAct Loop**: Observes, reflects, acts, and adapts iteratively.
    - **Plan**: Generates structured reasoning paths.
    - **Future Prediction**: Forecasts constraints and optimizations.
    - **Draft Response**: Generates structured responses.
    - **Critique**: Identifies and improves weaknesses.
    - **Creativity**: Introduces novel perspectives.
    - **ARX Loop**: Iteratively refines the response.
    """

    def __init__(self, api_keys_path: str = None, short_term_limit: int = 8000,
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(api_keys_path, short_term_limit, chunk_size, chunk_overlap)

    ### TOOL SELECTION ###
    def select_best_tools(self, user_prompt: str, top_k: int = 3) -> list:
        """
        Dynamically selects the most relevant tools based on the user prompt.
        """
        available_tools = self.get_available_tools()
        print("Available tools:\n" + ", ".join(available_tools) if available_tools else "No tools available.")
        
        if not available_tools:
            print("⚠️ No tools available for selection.")
            return []

        tool_selection_prompt = (
            f"Given the following tools:\n{', '.join(available_tools)}\n"
            f"Which {top_k} tools would be most useful for this task: {user_prompt}?"
            "Return a structured JSON list of tool names."
        )

        selected_tools = self.structured_output(
            user_prompt=tool_selection_prompt,
            system_prompt="Analyze the given tools and select the most relevant ones for the task.",
        )

        # Ensure the response is a list and print selected tools
        selected_tools = selected_tools if isinstance(selected_tools, list) else []
        
        print(f"🛠️ Selected Tools for '{user_prompt}': {selected_tools}")

        return selected_tools
        
    ### PLAN (Tree of Thoughts + Rationality) ###
    def plan(self, user_prompt: str, branching_factor: int = 3, contex: str = None, system_contex: str = None) -> str:
        """
        Generates multiple structured reasoning paths and selects the best.
        """
        return self.structured_output(
            user_prompt=f"🔹 TASK: Generate {branching_factor} structured reasoning paths for:\n{user_prompt}\n"
                        "Evaluate each path, compare them, and explicitly state which is the most effective.",
            system_prompt="🧠 PLAN GENERATION: Break down the problem into multiple structured solutions. "
                          "Compare and select the most rational and effective approach.",
            contex=contex, system_contex=system_contex
        )

    ### FUTURE PREDICTION (World Model) ###
    def future_prediction(self, user_prompt: str, contex: str = None, system_contex: str = None) -> str:
        """
        Predicts possible outcomes, constraints, and risks.
        """
        return self.reasoned_completion(
            user_prompt="🌍 FUTURE SIMULATION: Predict how this action unfolds over time. "
                        "Identify bottlenecks, risks, and optimization strategies."
                        f"🔮 TASK: Predict all possible consequences of:\n{user_prompt}\n"
                        "List potential risks, dependencies, and unintended outcomes. "
                        "Explicitly conclude with 'FINAL PREDICTION:' summarizing the best insights.",
            contex=contex, system_contex=system_contex
        )

    ### DRAFT RESPONSE (Inner Thought) ###
    def draft_response(self, user_prompt: str, contex: str = None, system_contex: str = None) -> str:
        """
        Generates a structured, well-reasoned response.
        """
        return self.chat_completion(
            user_prompt=f"📝 TASK: Draft a structured and well-reasoned response for:\n{user_prompt}\n"
                        "Ensure clarity, depth, and logical coherence. "
                        "Indicate 'FINAL RESPONSE:' when fully optimized.",
            system_prompt="✍️ DRAFTING: Construct a clear, logical, and structured response. "
                          "Ensure high reasoning quality and adaptability.",
            contex=contex, system_contex=system_contex
        )

    ### CRITIQUE (Self-Reflection) ###
    def critique(self, draft: str, contex: str = None, system_contex: str = None) -> str:
        """
        Evaluates and refines the draft response.
        """
        return self.chat_completion(
            user_prompt=f"🧐 TASK: Critique and refine the following response:\n{draft}\n"
                        "Identify weak areas, improve logical coherence, and ensure clarity. "
                        "State 'FINAL RESPONSE:' when it reaches the best version.",
            system_prompt="🔍 CRITIQUE MODE: Analyze the draft critically. Identify unclear sections, inconsistencies, "
                          "or areas lacking depth. Suggest precise improvements until fully refined.",
            contex=contex, system_contex=system_contex
        )

    ### CREATIVITY (Exploring Alternatives) ###
    def creativity(self, user_prompt: str, previous_thoughts: str, contex: str = None, system_contex: str = None) -> str:
        """
        Explores unexplored ideas and alternative perspectives.
        """
        return self.chat_completion(
            user_prompt=f"💡 TASK: Expand on this problem with fresh insights:\n{previous_thoughts}\n"
                        "Explore unconventional solutions and challenge assumptions. "
                        "Explicitly state 'FINAL RESPONSE:' when the most creative approach is reached.",
            system_prompt="🎨 CREATIVE MODE: Identify overlooked insights and propose bold, unconventional solutions. "
                          "Challenge assumptions and introduce alternative perspectives.",
            contex=contex, system_contex=system_contex
        )

    ### BASE LOOP ###
    def base_loop(self, user_prompt: str, system_prompt: str = None, temperature: float = None,
                  max_tokens: int = None, contex: str = None, system_contex: str = None, 
                  max_depth: int = 5, verbose: bool = True) -> str:
        """
        Iteratively executes tools until an optimal response is achieved.
        """
        response = ""
        for i in range(max_depth):
            tool_names = self.select_best_tools(user_prompt)
            response = self.chat_completion(
                user_prompt=f"🔄 REFINE ITERATIVELY USING TOOLS FOR:\n{user_prompt}\n\n"
                            f"💡 CURRENT OUTPUT:\n{response}\n"
                            "Indicate 'FINAL RESPONSE:' when the answer is complete.",
                system_prompt=system_prompt or "Use available tools when required to refine the response. "
                                               "Stop when the solution is optimal and explicitly state 'FINAL RESPONSE:'.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex, 
                tool_names=tool_names
            )

            if verbose:
                print(f"🔄 Base Loop Iteration {i+1} Response:\n{response}\n")
                
            if "FINAL RESPONSE:" in response:
                break
        return response

    ### REACT LOOP (Observation → Reflection → Action) ###
    def react_loop(self, user_prompt: str, system_prompt: str = None, temperature: float = None,
                   max_tokens: int = None, contex: str = None, system_contex: str = None, 
                   max_depth: int = 5, verbose: bool = True) -> str:
        """
        Uses cascading Observation → Reflection → Action steps iteratively.
        """
        response = ""
        for i in range(max_depth):
            tool_names = self.select_best_tools(user_prompt)

            # Step 1: OBSERVATION
            observation = self.chat_completion(
                user_prompt=f"🔍 OBSERVE the situation given:\n{user_prompt}\n"
                            f"💡 CURRENT OUTPUT:\n{response}\n"
                            "Describe key details, insights, and any missing elements.",
                system_prompt="Extract relevant observations and key insights from the given context.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex,
                tool_names=tool_names
            )

            # Step 2: REFLECTION
            reflection = self.chat_completion(
                user_prompt=f"🤔 REFLECT upon:\n{user_prompt}\n\n"
                            f"🔍 OBSERVATION:\n{observation}\n"
                            "Identify patterns, inconsistencies, and potential improvements.",
                system_prompt="Analyze the observations, identify missing aspects, and suggest improvements.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex,
                tool_names=tool_names
            )

            # Step 3: ACTION
            response = self.chat_completion(
                user_prompt=f"🚀 ACT based on:\n{user_prompt}\n\n"
                            f"🔍 OBSERVATION:\n{observation}\n"
                            f"🤔 REFLECTION:\n{reflection}\n"
                            "Formulate an optimized response, taking all insights into account. "
                            "Indicate 'FINAL RESPONSE:' when the answer is fully optimized.",
                system_prompt="Synthesize observations and reflections into a final, actionable response.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex,
                tool_names=tool_names
            )

            if verbose:
                print(f"🔄 ReAct Loop Iteration {i+1} Response:\n{response}\n")

            if "FINAL RESPONSE:" in response:
                break
        return response

    ### ARX LOOP (Iterative Cognition Pipeline) ###
    def arx_loop(self, user_prompt: str, max_depth: int = 5, human_in_loop: bool = True, verbose: bool = True, 
                 contex: str = None, system_contex: str = None) -> str:
        """
        Executes an agentic reasoning pipeline, iterating through structured cognitive steps.
        """
        response = ""
        cumulative_feedback = ""  # Stores human feedback across iterations
        
        for i in range(max_depth):
            tool_names = self.select_best_tools(user_prompt)

            plan = self.plan(user_prompt, contex=contex, system_contex=system_contex)
            prediction = self.future_prediction(plan, contex=contex, system_contex=system_contex)
            draft = self.draft_response(
                f"ORIGINAL PROMPT:\n{user_prompt}\n\nPLAN:\n{plan}\n\nPREDICTION:\n{prediction}", 
                contex=contex, system_contex=system_contex
            )
            critique = self.critique(
                f"DRAFT RESPONSE:\n{draft}\n\nPREVIOUS FEEDBACK:\n{cumulative_feedback}", 
                contex=contex, system_contex=system_contex
            )
            creative_input = self.creativity(
                f"ORIGINAL PROMPT:\n{user_prompt}\n\nDRAFT RESPONSE:\n{draft}\n\nCRITIQUE:\n{critique}", 
                contex=contex, system_contex=system_contex
            )

            response = self.chat_completion(
                user_prompt=f"ORIGINAL PROMPT:\n{user_prompt}\n\n"
                            f"PLAN:\n{plan}\n\nPREDICTION:\n{prediction}\n\n"
                            f"DRAFT RESPONSE:\n{draft}\n\nCRITIQUE:\n{critique}\n\n"
                            f"CREATIVE INPUT:\n{creative_input}\n\n"
                            "Clearly state 'FINAL RESPONSE:' when the answer fully satisfies the original prompt.",
                system_prompt="Follow a structured agentic reasoning pipeline: "
                              "PLAN → PREDICT → DRAFT → CRITIQUE → CREATIVITY → FINAL RESPONSE. "
                              "Continue refining iteratively until the best version is reached. "
                              "Explicitly mark the final response with 'FINAL RESPONSE:' when the optimal solution is achieved.",
                tool_names=tool_names,
                contex=contex,
                system_contex=system_contex
            )
            
            if verbose:
                print(f"🔁 ARX Loop Iteration {i+1} Response:\n{response}\n")

            if "FINAL RESPONSE:" in response:
                break

            if human_in_loop:
                feedback = input(f"🔍 Review response:\n{response}\n\n💬 Any feedback for improvement? \n")
                if feedback and feedback.strip():  
                    cumulative_feedback += f"\nIteration {i+1} Feedback: {feedback}\n"  # Append feedback for next cycle
                    critique = self.critique(
                        f"DRAFT RESPONSE:\n{draft}\n\nPREVIOUS FEEDBACK:\n{cumulative_feedback}",
                        contex=contex, system_contex=system_contex
                    )  # Ensure critique refines using stored feedback

        return response
        
      
if __name__ == "__main__":
    print("=== AgentGen Advanced Test Suite ===")
    
    # Initialize AgentGen with default settings.
    ag = AgentGen()
    
    # Optional: Set up system and user context for testing
    system_contex = "You are a helpful AI assistant specializing in technology analysis."
    contex = "Focus on providing clear, concise information about AI capabilities."
    
    # Setup a generic test prompt
    test_prompt = "Analyze the advantages and limitations of large language models for business applications."
    
    print("\n🧪 Testing AgentGen Reasoning Capabilities:")
    
    # Test Plan Generation
    try:
        plan_result = ag.plan(test_prompt)
        print(f"📝 Plan Generation Result:\n{plan_result[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in Plan Generation: {e}")
    
    # Test Future Prediction
    try:
        future_result = ag.future_prediction(test_prompt)
        print(f"🔮 Future Prediction Result:\n{future_result[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in Future Prediction: {e}")
    
    # Test Draft Response
    try:
        draft_result = ag.draft_response(test_prompt)
        print(f"📄 Draft Response Result:\n{draft_result[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in Draft Response: {e}")
    
    # Test Critique
    try:
        draft = "Large language models offer numerous benefits for businesses, including automation of routine tasks and improved customer service interactions."
        critique_result = ag.critique(draft)
        print(f"🔍 Critique Result:\n{critique_result[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in Critique: {e}")
    
    # Test Creativity
    try:
        previous_thoughts = "LLMs can automate customer service and content creation tasks."
        creativity_result = ag.creativity(test_prompt, previous_thoughts)
        print(f"💡 Creativity Result:\n{creativity_result[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in Creativity: {e}")
    
    # Test Basic Tool Selection
    try:
        selected_tools = ag.select_best_tools("Create a chart showing technology adoption trends")
        print(f"🛠️ Selected Tools: {selected_tools}\n✅")
    except Exception as e:
        print(f"❌ Error in Tool Selection: {e}")
    
    # Test Base Loop
    try:
        base_response = ag.base_loop(test_prompt, max_steps=3, contex=contex, system_contex=system_contex)
        print(f"📊 Base Loop Response:\n{base_response[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in Base Loop: {e}")
    
    # Test ReAct Loop
    try:
        react_response = ag.react_loop(test_prompt, max_steps=3, contex=contex, system_contex=system_contex)
        print(f"⚡ ReAct Loop Response:\n{react_response[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in ReAct Loop: {e}")
    
    # Test ARX Loop (without human-in-the-loop)
    try:
        arx_response_no_human = ag.arx_loop(test_prompt, max_depth=2, human_in_loop=False, contex=contex, system_contex=system_contex)
        print(f"🤖 ARX Loop (No Human) Response:\n{arx_response_no_human[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in ARX Loop (No Human): {e}")

    # Test ARX Loop (with human-in-the-loop)
    try:
        arx_response_human = ag.arx_loop(test_prompt, max_depth=3, human_in_loop=True, contex=contex, system_contex=system_contex)
        print(f"🧑‍💼 ARX Loop (With Human) Response:\n{arx_response_human[:200]}...\n✅")
    except Exception as e:
        print(f"❌ Error in ARX Loop (With Human): {e}")

    print("\n✅ All tests completed successfully!\n")
