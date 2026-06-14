from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from transaction_rag.config import get_settings
from transaction_rag.models import GuardrailFlag
from transaction_rag.pipeline import TransactionRAGPipeline


DATA_PATH = Path("data/assessment_transaction_data.xlsx - Transactions.csv")


def build_pipeline() -> TransactionRAGPipeline:
    settings = get_settings()
    df = pd.read_csv(DATA_PATH)
    return TransactionRAGPipeline(df=df, settings=settings)


pipeline = build_pipeline()

USER_VISIBLE_FLAGS = {
    GuardrailFlag.PROMPT_INJECTION,
    GuardrailFlag.OFF_TOPIC,
    GuardrailFlag.INPUT_LENGTH_EXCEEDED,
    GuardrailFlag.CROSS_USER_LEAKAGE,
    GuardrailFlag.OUTPUT_UNGROUNDED,
    GuardrailFlag.TOXIC_OUTPUT,
    GuardrailFlag.LOW_CONFIDENCE,
}


def respond(message: str, history: list[dict[str, str]], user_id: str):
    result = pipeline.run(user_id=user_id, prompt=message)
    content = result.response
    if result.visualizations:
        content += "\n\nGenerated visualizations:\n" + "\n".join(f"- `{path}`" for path in result.visualizations)
    visible_flags = [flag for flag in result.guardrail_flags if flag in USER_VISIBLE_FLAGS]
    if visible_flags:
        flags = ", ".join(flag.value for flag in visible_flags)
        content += f"\n\nGuardrails: `{flags}`"
    history = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": content}]
    gallery = [(str(Path(path).resolve()), Path(path).name) for path in result.visualizations]
    return history, gallery, result.model_dump(mode="json")


with gr.Blocks(title="LedgerGuard RAG") as demo:
    gr.Markdown("# LedgerGuard RAG")
    with gr.Row():
        user_id = gr.Dropdown(
            choices=pipeline.repository.user_ids,
            value=pipeline.repository.user_ids[0],
            label="User ID",
            interactive=True,
        )
    chatbot = gr.Chatbot(height=420)
    prompt = gr.Textbox(
        label="Ask about transactions",
        placeholder="What did I spend the most on last month?",
        lines=2,
    )
    submit = gr.Button("Send", variant="primary")
    with gr.Row():
        gallery = gr.Gallery(label="Visualizations", columns=3, height=280)
        json_output = gr.JSON(label="Structured Output")

    submit.click(
        respond,
        inputs=[prompt, chatbot, user_id],
        outputs=[chatbot, gallery, json_output],
    ).then(lambda: "", outputs=prompt)
    prompt.submit(
        respond,
        inputs=[prompt, chatbot, user_id],
        outputs=[chatbot, gallery, json_output],
    ).then(lambda: "", outputs=prompt)


if __name__ == "__main__":
    settings = get_settings()
    server_name = "0.0.0.0" if os.getenv("SPACE_ID") else settings.gradio_host
    demo.launch(
        server_name=server_name,
        server_port=settings.gradio_port,
        allowed_paths=[str(settings.outputs_dir.resolve())],
    )
