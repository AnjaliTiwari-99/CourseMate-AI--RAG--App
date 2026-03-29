<div align="center">

<img src="https://img.shields.io/badge/🎓-CourseMate_AI-7c3aed?style=for-the-badge&labelColor=0f0c29" alt="CourseMate AI"/>

# CourseMate AI

### *Chat with your course material. Instantly.*

An AI-powered RAG application that lets you upload any PDF — lecture notes, textbooks, research papers — and have a natural conversation with it.
<img width="1884" height="920" alt="Screenshot 2026-03-29 204602" src="https://github.com/user-attachments/assets/e4be71d3-15e3-4454-8afe-7f8f17396a81" />


[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://coursemate-ai--rag--app-3xyurjhqyxwsv9fzewzjj3.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![MistralAI](https://img.shields.io/badge/MistralAI-mistral--small--2506-FF7000?style=for-the-badge)](https://mistral.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-00ADD8?style=for-the-badge)](https://trychroma.com)

---

</div>

## ✨ What is CourseMate AI?

CourseMate AI is a **Retrieval-Augmented Generation (RAG)** application built for students, researchers, and lifelong learners. Instead of skimming through 100-page PDFs, just upload your document and ask questions in plain English — CourseMate AI reads, understands, and answers using only the content of your document.

> *No hallucinations. No guessing. Every answer is grounded in your uploaded material.*

---

## 🎬 Demo

**🔗 Try it live:** [coursemate-ai--rag--app-3xyurjhqyxwsv9fzewzjj3.streamlit.app](https://coursemate-ai--rag--app-3xyurjhqyxwsv9fzewzjj3.streamlit.app/)

Upload any PDF → Ask questions → Get document-grounded answers in seconds.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📄 **PDF Upload** | Upload any PDF directly from your browser — no pre-loading required |
| 🧠 **RAG Pipeline** | Retrieval-Augmented Generation ensures answers come from *your* document |
| 🔍 **MMR Retrieval** | Maximal Marginal Relevance search for diverse, high-quality context chunks |
| 💬 **Chat History** | Full conversation history preserved within your session |
| ⚡ **Fast Embeddings** | Mistral's `mistral-embed` model for high-quality semantic search |
| 🎨 **Polished UI** | Dark purple, minimal interface built with Streamlit + custom CSS |
| 🔒 **Document-Scoped** | If the answer isn't in the PDF, the model says so — no fabrication |
| 🔄 **Multi-PDF Support** | Upload a new PDF anytime to switch documents; chat resets automatically |

---

## 🏗️ Architecture

```
User uploads PDF
      │
      ▼
┌─────────────────┐
│  PyPDFLoader    │  ← Extracts raw text from PDF pages
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ RecursiveCharacterSplitter  │  ← chunk_size=1000, overlap=200
└────────────┬────────────────┘
             │
             ▼
┌────────────────────────┐
│  MistralAI Embeddings  │  ← mistral-embed model
│  (ThreadPoolExecutor)  │  ← isolated from async loop
└───────────┬────────────┘
            │
            ▼
┌────────────────────┐
│  ChromaDB          │  ← In-memory vector store (session-scoped)
│  (Vector Store)    │
└──────────┬─────────┘
           │
    User asks question
           │
           ▼
┌──────────────────────────┐
│  MMR Retriever           │  ← k=4, fetch_k=10, lambda=0.5
│  (Top-k similar chunks)  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────┐
│  ChatMistralAI               │  ← mistral-small-2506
│  + ChatPromptTemplate        │  ← context + question
└──────────┬───────────────────┘
           │
           ▼
      Answer to User
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom CSS |
| **LLM** | MistralAI `mistral-small-2506` |
| **Embeddings** | MistralAI `mistral-embed` |
| **Vector Store** | ChromaDB (in-memory) |
| **Orchestration** | LangChain |
| **PDF Loader** | LangChain `PyPDFLoader` |
| **Text Splitting** | `RecursiveCharacterTextSplitter` |
| **Deployment** | Streamlit Community Cloud |

---

## 📦 Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/coursemate-ai.git
cd coursemate-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get your free API key at [console.mistral.ai](https://console.mistral.ai)

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📋 Requirements

```txt
streamlit
langchain
langchain-community
langchain-mistralai
langchain-text-splitters
chromadb
pypdf
python-dotenv
httpx
```

---

## 📁 Project Structure

```
coursemate-ai/
│
├── app.py                 # Main Streamlit application
├── .env                   # API keys (not committed)
├── .env.example           # Template for environment variables
├── requirements.txt       # Python dependencies
└── README.md              # You are here
```

---

## 💡 How It Works

1. **Upload** — You upload a PDF via the drag-and-drop interface
2. **Process** — The PDF is parsed, split into overlapping chunks, and embedded using Mistral's embedding model
3. **Store** — Embeddings are stored in an in-memory ChromaDB vector store scoped to your session
4. **Retrieve** — When you ask a question, MMR search fetches the 4 most semantically relevant chunks
5. **Generate** — The retrieved context + your question are sent to `mistral-small-2506` with a strict system prompt
6. **Answer** — The LLM responds using *only* the document context — if the answer isn't there, it says so

---

## ⚙️ Configuration

You can tune the RAG pipeline parameters directly in `app.py`:

```python
# Chunking
chunk_size = 1000       # Characters per chunk
chunk_overlap = 200     # Overlap between chunks

# Retrieval (MMR)
k = 4                   # Number of chunks returned
fetch_k = 10            # Candidates fetched before MMR re-ranking
lambda_mult = 0.5       # Diversity vs relevance (0=max diversity, 1=max relevance)
```

---

## 🌐 Deployment

CourseMate AI is deployed on **Streamlit Community Cloud**.

To deploy your own instance:

1. Push your code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entry point
4. Add `MISTRAL_API_KEY` in the **Secrets** section of the Streamlit dashboard
5. Deploy 🚀

---

## 🔮 Future Improvements

- [ ] Multi-PDF support (chat across multiple documents simultaneously)
- [ ] Conversation memory across sessions
- [ ] Source citation with page numbers in responses
- [ ] Support for DOCX, TXT, and web URL ingestion
- [ ] Export conversation as PDF

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Built with ❤️ using **LangChain** · **MistralAI** · **ChromaDB** · **Streamlit**

**[⭐ Star this repo](https://github.com/your-username/coursemate-ai)** if you found it useful!

</div>
