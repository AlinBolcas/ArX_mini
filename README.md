# ARX-Mini: OpenAI API Wrapper for Pythonista

<video src="data/visuals/video_cover_03.mp4" autoplay loop muted></video>

ARX-Mini is a lightweight yet powerful OpenAI API wrapper developed specifically for Pythonista on iOS. It provides a minimal, low-level framework for creating complex text generation applications with memory management, retrieval-augmented generation (RAG), and tool execution capabilities.

## Features

- 🔄 **Unified Text Generation Interface** - Chat completions, structured outputs, reasoned completions, and vision analysis
- 💾 **Memory Management** - Short-term and long-term memory storage with automatic summarization
- 🔍 **Retrieval-Augmented Generation (RAG)** - Simple NumPy-based vector search for context enhancement
- 🛠️ **Tool Integration** - Execute tasks like web crawling, file manipulation, and image generation
- 🧠 **AgentGen Framework** - Lightweight agent architecture for multi-step reasoning
- 📱 **iOS & Pythonista Optimized** - Designed to work within Pythonista's constraints

## Requirements

- Pythonista 3 for iOS
- OpenAI API key
- No additional package installation required (uses only pre-installed modules)

## Setup

1. Clone this repository into your Pythonista environment
2. Add your OpenAI API key to `data/api_keys.py`:

```python
OPENAI_API_KEY = "your_openai_api_key"
```

3. Start using the framework in your projects

## Quick Start
<video src="data/visuals/video_cover_02.mp4" autoplay loop muted></video>

### Basic Chat Completion

```python
from textGen.textgen import TextGen

# Initialize TextGen
tg = TextGen()

# Simple chat completion
response = tg.chat_completion(
    user_prompt="Explain quantum computing in simple terms",
    system_prompt="You are a helpful physics teacher."
)
print(response)
```

### Using Tools

```python
# Chat completion with web search tool
response = tg.chat_completion(
    user_prompt="What's the latest news about AI regulation?",
    tool_names=["web_crawl_query"]
)
print(response)
```

### Structured Output

```python
# Generate structured JSON
data = tg.structured_output(
    user_prompt="Create a list of 5 healthy breakfast ideas",
    system_prompt="Return the output as a JSON array with 'name' and 'ingredients' fields."
)
print(data)
```

### Vision Analysis

```python
# Analyze an image
analysis = tg.vision_analysis(
    image_url="https://example.com/image.jpg",
    user_prompt="Describe what's in this image"
)
print(analysis)
```

## Architecture
ARX-Mini consists of several key components:
<video src="data/visuals/video_cover.mp4" autoplay loop muted></video>
### TextGen

The core class that orchestrates:
- LLM calls via OpenAI's API
- Long-term memory retrieval via RAG
- External tool integration
- System and user context management

### OAI

Handles direct interactions with OpenAI's API:
- Chat completions
- Structured output
- Vision analysis
- Embeddings
- Text-to-speech and image generation

### Memory

Manages conversation history:
- Short-term memory for recent interactions
- Long-term memory with automatic insight extraction
- Formatted retrieval for context inclusion

### RAG

Simple retrieval-augmented generation system:
- Text chunking
- Vector embeddings
- Cosine similarity search
- Context retrieval

### Tools

Collection of utilities accessible to the LLM:
- File operations
- Web crawling and information retrieval
- QR code generation
- Media generation (images, speech)
- And more

### AgentGen

Agent framework built on TextGen:
- Multi-step reasoning loops
- Tool selection and execution
- Iterative response refinement

## Advanced Usage
<video src="data/visuals/video_cover_04.mp4" autoplay loop muted></video>

### Memory Management

```python
from textGen.textgen import TextGen

tg = TextGen()

# Reset memory when starting a new conversation
tg.memory.reboot_memory()

# Conversation with memory
response1 = tg.chat_completion("Tell me about Mars.")
print(response1)

# Follow-up questions use previous context
response2 = tg.chat_completion("How long would it take to get there?")
print(response2)

# Access memory insights
long_term_insights = tg.memory.retrieve_long_term()
print(long_term_insights)
```

### AgentGen for Complex Tasks

```python
from agents.agentgen import AgentGen

ag = AgentGen()

# Complex task with automatic tool selection
response = ag.base_loop(
    "Research the impacts of climate change on agriculture, summarize the findings, and generate a chart visualizing key data points.",
    verbose=True
)
print(response)
```

## Project Structure

```
arx_mini/
├── data/
│   ├── api_keys.py            # API key storage
│   └── various documentation
├── textGen/
│   ├── textgen.py             # Main TextGen class
│   ├── oai.py                 # OpenAI API wrapper
│   ├── memory.py              # Memory management
│   ├── rag.py                 # Retrieval-augmented generation
│   └── tools.py               # LLM-accessible tools
├── agents/
│   ├── agentgen.py            # Advanced agent framework
│   └── agentgen_base.py       # Simplified agent version
├── utils.py                   # Shared utility functions
└── output/                    # Generated files and memory storage
```

## Limitations

- Designed specifically for Pythonista 3 on iOS
- Works within Pythonista's module constraints (no pip install)
- Simple RAG implementation uses NumPy rather than specialized vector databases
- Tool execution limited to available iOS capabilities

## License

MIT License

## Contributing

Contributions are welcome! As this project is optimized for Pythonista on iOS, please ensure any contributions maintain compatibility with the platform's constraints.
