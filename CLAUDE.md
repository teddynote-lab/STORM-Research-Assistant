# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a ReAct (Reasoning and Action) agent built with LangGraph for LangGraph Studio. The agent follows a cyclic pattern:

1. **Input Processing**: User queries enter through `InputState` 
2. **Model Reasoning**: The `call_model` node processes the query and decides on actions
3. **Tool Execution**: If tools are needed, routes to the `tools` node for execution
4. **Iteration**: Results feed back to the model for further reasoning until a final answer is reached

### Key Components

- **graph.py**: Contains the main agent logic using StateGraph. The graph alternates between reasoning (`call_model`) and action (`tools`) nodes based on conditional routing.
- **state.py**: Defines `InputState` and `State` dataclasses that manage conversation history and tool interactions
- **tools.py**: Currently implements Tavily search - extend here to add new capabilities
- **configuration.py**: Runtime configuration including model selection (supports Anthropic/OpenAI)
- **prompts.py**: System prompts for the agent

### Model Configuration

The agent supports multiple LLM providers via the format `provider/model-name`:
- Default: `azure/gpt-4.1`
- Also supports:
  - Azure OpenAI models like `azure/gpt-4.1`
  - OpenAI models like `openai/gpt-4-turbo`
  - Anthropic models like `anthropic/claude-3-5-sonnet-20240620`

Set API keys in `.env`:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY` (required for Azure OpenAI)
- `AZURE_OPENAI_ENDPOINT` (required for Azure OpenAI)
- `TAVILY_API_KEY` (required for search functionality)

### LangGraph Studio Integration

Entry point defined in `langgraph.json` points to `./src/storm_research/graph.py:graph`. The studio provides visual debugging and state manipulation capabilities.

### Installation

```bash
# Install with uv
uv pip install -e .

# Or with pip
pip install -e .
```

## Testing after code changes

```bash
# Run the agent
uv run langgraph dev
```