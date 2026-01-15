# 📄 PDF Question-Answering System (RAG)
> Retrieval-Augmented Generation system using free/open-source LLMs

## 🎯 Problem Statement
Many people need to quickly extract information from large PDFs without reading hundreds of pages. This tool lets you upload any PDF and ask questions in natural language.

## 🛠️ Tech Stack
- **LLM**: Gemini API (free tier) / Llama 3.2 (local option)
- **Vector DB**: ChromaDB (open-source)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Framework**: LangChain / Python 3.11

## 🚀 Features
- ✅ Upload PDFs and ask questions
- ✅ Contextual answers with source citations
- ✅ No API costs (optional local mode)
- ✅ Web UI using Streamlit

## 📸 Demo
[Add a GIF or screenshot showing: Upload PDF → Ask Question → Get Answer]

## 🏃 Quick Start
```bash
git clone https://github.com/starlord78275/simple-Rag.git
cd simple-Rag
pip install -r requirements.txt
streamlit run app.py
```
📂 Project Structure
simple-Rag/
├── app.py              # Streamlit UI
├── data_loader.py      # PDF processing
├── simple_rag.py       # RAG logic
└── requirements.txt    # Dependencies
🧪 Testing

Tested with 50+ page technical PDFs (accuracy: 85% on factual questions)
🔮 Future Improvements

    Add multi-document search

    Support for tables/images in PDFs

    Deploy to HuggingFace Spaces

text

This format shows you understand **business value** and **software engineering**, not just ML theory.[2][3]

### 2. Add Visuals (Critical!)
Recruiters spend 10 seconds per repo. Add:
- **Screenshot**: Show the UI with a question and answer visible.
- **GIF**: Record a 15-second demo using [ScreenToGif](https://www.screentogif.com/) showing: Upload → Ask → Answer.[4][1]

### 3. Fix Repository Settings
- **Topics/Tags**: Add these topics to your repo (Settings → Topics):
  - `retrieval-augmented-generation`
  - `langchain`
  - `nlp`
  - `machine-learning`
  - `pdf-parsing`
  
  This makes your repo show up in GitHub searches.[5][4]

- **Description**: Change "simple rag project that answer you're question" to:

RAG-based PDF Q&A system using LangChain and ChromaDB (no paid APIs required)

text

### 4. Clean Up the Code
Based on your screenshot, you have `.env` committed. **Remove it immediately**:
```bash
git rm --cached .env
echo ".env" >> .gitignore
git commit -m "Remove sensitive .env file"
git push

Then add a .env.example file with fake values:
```
text
GEMINI_API_KEY=your_api_key_here

This shows you understand security.

​
5. Pin This Repository

On your GitHub profile, click "Customize your pins" and select this repo. It will appear at the top of your profile.

​
6. Create a Profile README (Optional but Powerful)

Create a new repository called starlord78275 (same as your username). Add a README like:

text
# Hi, I'm [Nitin Gavande]
**Data Science & ML Engineer** building real-world AI applications

## 🔧 Tech Stack
Python | TensorFlow | LangChain | FastAPI | React

## 🚀 Featured Projects
- [PDF Q&A System](link) - RAG with LangChain
- [Music Emotion Recognition](link) - CNN + Spotify API
- [Manga Translator](link) - OCR + Translation

📫 Reach me: [LinkedIn](https://www.linkedin.com/in/nitin-gavande-891a7b31a/) | [Email](mailto:bhimgavande.777@gmail.com)
