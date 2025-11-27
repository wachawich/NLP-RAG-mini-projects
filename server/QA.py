import pandas as pd
import numpy as np
import os

def build_prompt(user_question: str, context: str) -> str:
    return (
        "You are a friendly assistant who provides beautifully formatted answers with clear visual hierarchy.\n\n"

        "RESPONSE STRUCTURE:\n"
        "1. Emoji + Brief greeting (1 line)\n"
        "2. Main heading using ## (e.g., ## 🧋 What is Bubble Tea?)\n"
        "3. Answer in 2-3 paragraphs with **bold** for key terms\n"
        "4. Total: 4-5 sentences\n\n"

        "FORMATTING RULES:\n"
        "- Use ## for the main question/topic heading (makes it bigger)\n"
        "- Use **bold** for important terms within paragraphs\n"
        "- Add 2-3 emojis total (heading + content)\n"
        "- Keep paragraphs separated with blank lines\n"
        "- NO === or --- separators\n"
        "- Example structure:\n"
        "  🎯 Great question!\n"
        "  \n"
        "  ## 🧋 What is Bubble Tea?\n"
        "  \n"
        "  **Bubble tea** is a drink from **Taiwan**...\n\n"

        "CONTENT RULES:\n"
        "- Use ONLY information from context\n"
        "- Be concise (4-5 sentences)\n"
        "- Answer directly and completely\n\n"

        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {user_question}\n\n"

        "Provide a well-formatted answer:\n"
    )

def final_llm_answer(groq_client, user_question: str, passages: list) -> str:
    context = "\n".join([p for p in passages])
    prompt = build_prompt(user_question, context)
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()

    # แปลง escape sequences
    answer = answer.replace("\\n\\n", "\n\n")
    answer = answer.replace("\\n", "\n")
    answer = answer.replace("\\'", "'")
    answer = answer.replace('\\"', '"')

    return answer
def ask(groq_client, result, question: str, display_markdown: bool = True):
    """
    ถามคำถามและแสดงผล

    Parameters:
    - question: คำถาม
    - display_markdown: True = แสดง Markdown สวยงาม, False = คืน text ธรรมดา
    """
    passages = result
    answer = final_llm_answer(groq_client, question, passages)

    return answer
    # if display_markdown:
    #     display(Markdown(answer))
    # sima






