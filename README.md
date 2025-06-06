# 🌪️ STORM Research Assistant

<!-- Project badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.6+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **STORM** (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) - A writing system for generating grounded and organized long-form articles from scratch, with comparable breadth and depth to Wikipedia pages

## 📖 Overview

STORM Research Assistant is a LangGraph-based implementation of the STORM methodology from Stanford, designed to write grounded and organized long-form articles from scratch. The system models the pre-writing stage by (1) discovering diverse perspectives for researching the given topic, (2) simulating conversations where writers with different perspectives pose questions to a topic expert grounded on trusted Internet sources, and (3) curating the collected information to create an outline before generating the final article.

### 🎯 Key Features

- **🔍 Pre-writing Stage Modeling**: Comprehensive research and outline preparation before article generation
- **🤖 Diverse Perspective Discovery**: Automatic generation of multiple expert perspectives for comprehensive topic coverage
- **💬 Simulated Expert Conversations**: Multi-perspective question asking with grounded answers from trusted sources
- **📚 Grounded Information**: All content backed by reliable Internet sources (Tavily web search and ArXiv papers)
- **📊 Structured Outline Creation**: Systematic curation of collected information into organized outlines
- **✏️ Long-form Article Generation**: Wikipedia-quality articles with introduction, detailed sections, and conclusion
- **🔄 User Feedback Integration**: Human-in-the-loop capability for refining analyst perspectives
- **⚡ Parallel Processing**: Simultaneous execution of multiple perspective interviews for efficiency
- **🎨 LangGraph Studio Support**: Full integration with LangGraph Studio for visual debugging

## 🏗️ Architecture

### System Structure

```
📁 src/storm_research/
├── 📄 __init__.py          # Package initialization
├── 🧠 graph.py            # LangGraph graph definition (main logic)
├── 📊 state.py            # State and data model definitions
├── 💬 prompts.py          # Prompt templates
├── ⚙️ configuration.py     # System configuration management
├── 🔧 tools.py            # Search tool implementations
└── 🛠️ utils.py            # Utility functions
```

### Workflow

```mermaid
graph TD
    A[Start] --> B[Discover Diverse Perspectives]
    B --> C[Generate Expert Analysts]
    C --> D{User Feedback?}
    D -->|Has Feedback| C
    D -->|No Feedback| E[Simulate Expert Conversations]
    E --> F1[Perspective 1: Q&A with Expert]
    E --> F2[Perspective 2: Q&A with Expert]
    E --> F3[Perspective 3: Q&A with Expert]
    F1 --> G1[Ground Answers in Sources]
    F2 --> G2[Ground Answers in Sources]
    F3 --> G3[Ground Answers in Sources]
    G1 --> H[Curate Information]
    G2 --> H
    G3 --> H
    H --> I[Create Structured Outline]
    I --> J[Generate Article Sections]
    J --> K[Write Introduction]
    J --> L[Write Conclusion]
    K --> M[Final Wikipedia-style Article]
    L --> M
    M --> N[End]
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for chosen LLM providers

### 1. Clone the Repository

```bash
git clone https://github.com/teddynote-lab/STORM-Research-Assistant.git
cd STORM-Research-Assistant
```

### 2. Environment Setup

```bash
# Create virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### 3. Environment Variables

Create a `.env` file in the root directory and configure the following API keys:

```env
# Required API Keys
TAVILY_API_KEY=your_tavily_api_key

# LLM Provider API Keys (choose one or more)
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Optional: LangSmith for tracing
LANGSMITH_PROJECT=STORM-Research-Assistant
LANGSMITH_API_KEY=your_langsmith_api_key
```

### 4. Running LangGraph Studio

```bash
# Install LangGraph CLI (one-time setup)
pip install langgraph-cli

# Run LangGraph Studio
uv run langgraph dev
```

Access the studio at `http://localhost:8123`

## 📝 Usage

### Basic Usage

```python
from storm_research import graph
from langchain_core.runnables import RunnableConfig

# Configuration
config = RunnableConfig(
    configurable={
        "thread_id": "research-001",
        "model": "azure_openai/gpt-4.1",  # Default model
        "max_analysts": 3,
        "max_interview_turns": 3,
    }
)

# Start article generation
inputs = {
    "topic": "The Future of Quantum Computing in Cryptography",
    "max_analysts": 3
}

# Execute (First step: Discover perspectives and generate analysts)
result = await graph.ainvoke(inputs, config)

# Provide user feedback (optional) to refine perspectives
await graph.aupdate_state(
    config,
    {"human_analyst_feedback": "Please add a cybersecurity expert perspective"},
    as_node="human_feedback"
)

# Complete the pre-writing stage and generate article
final_result = await graph.ainvoke(None, config)
print(final_result["final_report"])
```

### Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `azure/gpt-4.1` | LLM model to use (provider/model format) |
| `max_analysts` | 3 | Number of analysts to generate |
| `max_interview_turns` | 3 | Maximum interview turns per analyst |
| `tavily_max_results` | 3 | Number of Tavily search results |
| `arxiv_max_docs` | 3 | Number of ArXiv documents to retrieve |
| `parallel_interviews` | `True` | Whether to run interviews in parallel |

#### Supported Models

- **Azure OpenAI**: `azure/gpt-4.1`, `azure/gpt-4o`, `azure/gpt-4o-mini`
- **OpenAI**: `openai/gpt-4`, `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`
- **Anthropic**: `anthropic/claude-3-5-sonnet-20240620`, `anthropic/claude-3-opus-20240229`

## 📚 Examples

### Technology Research

```python
topic = "Next-Generation AI Architectures: Beyond Transformers"
```

Generated analysts might include:
- AI Architecture Researcher
- Hardware Optimization Expert
- Industry Applications Specialist

### Business Analysis

```python
topic = "The Impact of AI on Global Supply Chain Management in 2024"
```

Generated analysts might include:
- Supply Chain Expert
- AI Technology Analyst
- Business Strategy Consultant

### Academic Research

```python
topic = "Quantum Error Correction Methods for Scalable Quantum Computing"
```

Generated analysts might include:
- Quantum Physics Researcher
- Error Correction Specialist
- Hardware Implementation Expert

## 🧪 Testing

```bash
# Run all tests
make test

# Run unit tests only
python -m pytest tests/unit_tests/

# Run specific test file
make test TEST_FILE=tests/unit_tests/test_configuration.py

# Run with coverage
python -m pytest --cov=storm_research tests/

# Run integration tests
python -m pytest tests/integration_tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`make test`)
6. Run linting (`make lint`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks with:
```bash
make lint
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on Stanford's STORM paper: [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
  - STORM achieves 25% better article organization and 10% broader topic coverage compared to baseline methods
  - The methodology addresses challenges in pre-writing stages including topic research and outline preparation
- Built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://python.langchain.com/)
- Original implementation reference: [LangChain Korea Tutorial](https://github.com/teddylee777/langchain-kr/blob/main/17-LangGraph/03-Use-Cases/10-LangGraph-Research-Assistant.ipynb)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/teddynote-lab/STORM-Research-Assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/teddynote-lab/STORM-Research-Assistant/discussions)
- **Documentation**: [Wiki](https://github.com/teddynote-lab/STORM-Research-Assistant/wiki)