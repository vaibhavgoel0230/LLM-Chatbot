## Basic LLM Chatbot

Simple Streamlit chat UI backed by a LangGraph workflow and OpenAI chat model.

### Prerequisites
- Python 3.10+ recommended
- An OpenAI API key

### Setup
1. Create and activate a virtual environment:
   - `python -m venv venv`
   - `source venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create a `.env` file with your API key:
   - `OPENAI_API_KEY=your_key_here`

### Run
- `python -m streamlit run streamlit_frontend.py`

### Project flow
- `streamlit_frontend.py` renders the chat UI, stores history in `st.session_state`, and sends new user messages to the backend.
- `langgraph_backend.py` builds a simple LangGraph `StateGraph` with a single `chat_node`.
- `chat_node` calls `ChatOpenAI` (model `gpt-4o-mini`) and returns the assistant message.
- A `MemorySaver` checkpointer keeps state by `thread_id`, and the graph is invoked on each user input.
