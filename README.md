# 404_found
team name:404 found
team leader: kapilesh


# 🦊 Study Buddy – AI-Powered Research & Learning Assistant

Study Buddy is an **AI-powered research assistant** designed to help students, professionals, and researchers quickly gather insights, analyze documents, and generate structured reports.  
It combines multiple AI agents into a single platform that can:  

- 🤝 Act as a **Personal AI Assistant** (answering questions about your background, skills, and career).  
- 🔍 Perform **Deep Research** on any topic using web searches and summarization.  
- 📂 Analyze and summarize **PDF documents** into detailed, structured reports.  
- 📧 Send results and research reports via **email**.  

---

## 🚀 Features

- **Personal AI Assistant**  
  Interact with your AI persona to answer career-related or study-related questions.  

- **Deep Research Mode**  
  Provide a topic, and Study Buddy will:  
  1. Plan web searches  
  2. Perform searches & summarize findings  
  3. Generate a cohesive long-form research report  
  4. Email the results in a clean format  

- **PDF Research Mode**  
  Upload a PDF (e.g., research paper, article, or study material) and get:  
  - A detailed summary  
  - A structured report in Markdown  
  - Suggested follow-up research questions  

- **Push Notifications**  
  Integration with **Pushover** to log user interest, unanswered questions, and feedback.  

- **Gradio Web Interface**  
  Easy-to-use tabbed interface built with [Gradio](https://www.gradio.app/).  

---

## 🛠️ Tech Stack

- **Frontend/UI**: Gradio  
- **AI Models**: OpenAI GPT-4o-mini  
- **Data Processing**: PyPDF for PDF parsing  
- **Email**: SendGrid API  
- **Task Management**: Async Python & custom multi-agent orchestration  
- **Environment Management**: python-dotenv  

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/study-buddy.git
   cd study-buddy
