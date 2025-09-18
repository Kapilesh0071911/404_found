# app.py
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader
import os, json, requests
from openai import OpenAI

# Import research manager
from research_manager import ResearchManager

load_dotenv(override=True)


# ------------------------
# Push notification helpers
# ------------------------
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Record that a user is interested and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The user's email"},
            "name": {"type": "string", "description": "The user's name"},
            "notes": {"type": "string", "description": "Extra context"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unanswered question"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]


# ------------------------
# Personal AI ("Me")
# ------------------------
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "kapilesh"

        # Load LinkedIn PDF
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = "".join([p.extract_text() or "" for p in reader.pages])

        # Load summary
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append(
                {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
            )
        return results

    def system_prompt(self):
        return (
            f"You are acting as {self.name}, answering questions about {self.name}'s career, "
            "skills, and background. Stay professional and engaging.\n\n"
            f"## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
            "If you don't know something, use record_unknown_question. "
            "Encourage users to share their email and record it with record_user_details."
        )

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [
            {"role": "user", "content": message}
        ]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


# ------------------------
# Deep Research integration
# ------------------------
async def run_research(query: str):
    async for chunk in ResearchManager().run(query):
        yield chunk


async def run_pdf_research(query: str, pdf_file):
    """Read PDF and generate a report instead of web searches"""
    if pdf_file is None:
        yield "⚠️ Please upload a PDF file."
        return

    yield "📖 Reading PDF..."
    reader = PdfReader(pdf_file.name)
    pdf_text = " ".join([page.extract_text() or "" for page in reader.pages])

    yield "📝 Writing report from PDF..."
    from writer_agent import writer_agent, ReportData
    from agents import Runner

    result = await Runner.run(
        writer_agent,
        f"Original query: {query}\nSummarized PDF content: {pdf_text}",
    )
    report = result.final_output_as(ReportData)

    yield "✅ Report complete!"
    yield report.markdown_report


# ------------------------
# Gradio App
# ------------------------
def create_ui():
    with gr.Blocks(theme=gr.themes.Default(primary_hue="green")) as ui:
        gr.Markdown("# 🦊 Study Buddy")

        with gr.Tab("🤝 Personal AI Assistant"):
            gr.ChatInterface(Me().chat, type="messages")

        with gr.Tab("🔍 Deep Research"):
            query_textbox = gr.Textbox(
                label="Research Topic",
                placeholder="e.g., Future of Web3 in Education",
            )
            run_button = gr.Button("Run Web Research", variant="primary")
            report_output = gr.Markdown(label="Report")
            run_button.click(fn=run_research, inputs=query_textbox, outputs=report_output)
            query_textbox.submit(fn=run_research, inputs=query_textbox, outputs=report_output)

        with gr.Tab("📂 PDF Research"):
            pdf_query = gr.Textbox(label="Research Topic for PDF")
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_button = gr.Button("Analyze PDF", variant="primary")
            pdf_report_output = gr.Markdown(label="PDF Report")
            pdf_button.click(fn=run_pdf_research, inputs=[pdf_query, pdf_upload], outputs=pdf_report_output)

    return ui


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(inbrowser=True)
