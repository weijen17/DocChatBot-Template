# 📄 Document Chat with AI & Web Search

A Streamlit application that allows you to chat with your documents using AI, with integrated web search capabilities powered by Tavily.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red.svg)

## ✨ Features

- 💬 **Conversational AI**: Chat naturally with your PDF and text documents
- 🔍 **Web Search Integration**: Powered by Tavily for real-time web information
- 🧠 **Memory**: Maintains conversation context across multiple queries
- 📚 **Citations**: Shows page numbers and source snippets from documents
- 🌊 **Streaming Responses**: Real-time token-by-token response generation
- 🔄 **Multiple Modes**: 
  - Document Only
  - Document + Web Search
  - Web Search Only

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Tavily API key ([Get one here](https://tavily.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DocChatBot-Template.git
   cd DocChatBot-Template
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   
   Open your browser and navigate to: `http://localhost:8501`

## 🛠️ Manual Setup (Without Docker)

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app001.py
   ```

## ⚙️ Configuration

All configurations are managed through the `.env` file:

| Variable              | Description | Default       |
|-----------------------|-------------|---------------|
| `OPENAI_API_KEY`      | Your OpenAI API key | Required      |
| `TAVILY_API_KEY`      | Your Tavily API key | Required      |
| `FAISS_INDEX_PATH`    | Path to store FAISS vector index | `faiss_index` |
| `MODEL_NAME`          | OpenAI model to use | `gpt-4o-mini` |
| `MODEL_PROVIDER`      | Model provider | `openai`      |
| `TEMPERATURE`         | Model temperature (0-1) | `0.1`         |
| `NUMBER_OF_RETRIEVAL` | Number of document chunks to retrieve | `3`           |
| `NUMBER_OF_SEARCH`    | Number of document chunks to retrieve | `3`           |

## 📖 Usage

1. **Upload a Document**: Upload a PDF or TXT file using the file uploader
2. **Choose Mode**: Select your preferred mode:
   - Document Only: Search only in your uploaded document
   - Document + Web Search: Get answers from both document and web
   - Web Search Only: Search only the web
3. **Ask Questions**: Type your questions in the chat input
4. **View Sources**: Click on expandable sections to see supporting evidence

## 🏗️ Architecture

```
├── app001.py                 # Main Streamlit application
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Vector Store**: FAISS
- **Embeddings**: OpenAI Embeddings
- **Web Search**: Tavily
- **Document Processing**: LangChain, PyPDF


## 🙏 Acknowledgments

- [OpenAI](https://openai.com) for the LLM API
- [Tavily](https://tavily.com) for web search capabilities
- [LangChain](https://langchain.com) for the orchestration framework
- [Streamlit](https://streamlit.io) for the web framework


## 🐛 Known Issues

- Large PDF files (>100 pages) may take longer to process
- FAISS index is stored locally and persists between sessions
