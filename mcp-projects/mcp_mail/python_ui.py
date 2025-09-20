import gradio as gr
import requests

SERVER_URL = "http://localhost:3000"

def send_email(to, subject, message):
    try:
        response = requests.post(f"{SERVER_URL}/send", json={
            "to": to,
            "subject": subject, 
            "message": message
        })
        
        if response.status_code == 200:
            data = response.json()
            return f"✅ Mail sent successfully!\nTo: {data['to']}\nSubject: {data['subject']}\nMessage ID: {data['messageId']}"
        else:
            return f"❌ Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"

def check_inbox(limit):
    try:
        response = requests.get(f"{SERVER_URL}/inbox?limit={limit}")
        
        if response.status_code == 200:
            data = response.json()
            emails = data.get('emails', [])
            
            if not emails:
                return "📭 No emails found"
            
            result = f"📬 Found {len(emails)} emails:\n\n"
            for i, email in enumerate(emails, 1):
                result += f"**{i}.** From: {email.get('from', 'Unknown')}\n"
                result += f"Subject: {email.get('subject', 'No Subject')}\n"
                result += f"Date: {email.get('date', 'Unknown')}\n\n"
            
            return result
        else:
            return f"❌ Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"

with gr.Blocks(title="Gmail Manager") as app:
    gr.Markdown("# 📧 Gmail Manager")
    
    with gr.Tab("Send Email"):
        with gr.Row():
            with gr.Column():
                to_input = gr.Textbox(label="To", placeholder="recipient@example.com")
                subject_input = gr.Textbox(label="Subject", placeholder="Email subject")
                message_input = gr.Textbox(label="Message", lines=5, placeholder="Your message...")
                send_btn = gr.Button("Send Email", variant="primary")
            
            with gr.Column():
                send_output = gr.Textbox(label="Result", lines=5, interactive=False)
    
    with gr.Tab("Check Inbox"):
        with gr.Row():
            with gr.Column():
                limit_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of emails")
                check_btn = gr.Button("Check Inbox", variant="primary")
            
            with gr.Column():
                inbox_output = gr.Textbox(label="Inbox", lines=10, interactive=False)
    
    send_btn.click(send_email, inputs=[to_input, subject_input, message_input], outputs=send_output)
    check_btn.click(check_inbox, inputs=[limit_input], outputs=inbox_output)

if __name__ == "__main__":
    app.launch()