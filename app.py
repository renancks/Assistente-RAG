import os

import streamlit as st
from src.pipeline import PDFQueryPipeline

def main():
    st.title("🤖 Assistente de PDF com RAG")
    pdf_file = st.file_uploader("Upload do PDF", type=['pdf'])
    if pdf_file:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        try:
            pipeline = PDFQueryPipeline("temp.pdf")
            pipeline.run_pipeline()
            question = st.text_input("Faça sua pergunta sobre o PDF:")
            if question:
                with st.spinner('Gerando resposta...'):
                    response = pipeline.answer_question(question)
                    st.write("Resposta:", response)
        except Exception as e:
            st.error(f"Erro ao processar o PDF: {str(e)}")
        finally:
            # Limpa arquivo temporário
            os.remove("temp.pdf")

if __name__ == "__main__":
    main()