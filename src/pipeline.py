import os
import shutil
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_index.core import SimpleDirectoryReader
from langchain_community.docstore.document import Document
from transformers import pipeline
from groq import Groq

load_dotenv()

class PDFQueryPipeline:
    def __init__(self, pdf_path: str):
        """
        Inicializa o pipeline RAG
        Args:
            pdf_path: Caminho para o arquivo PDF
        """
        self.pdf_path = pdf_path
        
        # Inicializa componentes como None
        self.documents = None
        self.vector_store = None
        self.qa_chain = None
        
        # Configurações
        self.EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not self.GROQ_API_KEY:
            raise ValueError("A chave da API Groq não foi encontrada. Verifique o arquivo .env.")
        self.groq_client = Groq(api_key=self.GROQ_API_KEY)

    def load_pdf(self) -> List:
        """
        Carrega e extrai texto do PDF usando SimpleDirectoryReader
        Returns:
            Lista de documentos extraídos
        """
        print("📚 Carregando PDF...")
        temp_dir = "temp_dir"
        os.makedirs(temp_dir, exist_ok=True)

        # Copia o PDF para o diretório temporário
        shutil.copy(self.pdf_path, temp_dir)

        # Usa o SimpleDirectoryReader para ler o PDF
        documents = SimpleDirectoryReader(temp_dir).load_data()

        # Remove o diretório temporário
        shutil.rmtree(temp_dir)

        # Converte os documentos lidos em objetos compatíveis com o LangChain
        self.documents = [Document(page_content=doc.text) for doc in documents]

        # Opcional: Imprimir o conteúdo extraído para verificação
        #for i, doc in enumerate(self.documents):
        #    print(f"----- Documento {i+1} -----")
        #    print(doc.page_content)
        #    print("---------------------------\n")

        return self.documents

    def split_text(self) -> List:
        """
        Divide o texto em chunks menores para processamento
        Returns:
            Lista de chunks de texto
        """
        print("✂️ Dividindo texto em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        self.documents = text_splitter.split_documents(self.documents)
        return self.documents

    def create_embeddings(self):
        """
        Gera embeddings usando modelo do Hugging Face e cria base vetorial
        """
        print("🔤 Gerando embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Cria base vetorial com FAISS
        self.vector_store = FAISS.from_documents(self.documents,embeddings)

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Gera o prompt para o modelo
        """
        return f"""Utilize as informações abaixo para responder à pergunta de forma clara e concisa.

        Contexto:
        {context}

        Pergunta:
        {question}

        Resposta: """

    def answer_question(self, question: str) -> str:
        """
        Responde uma pergunta baseada no conteúdo do PDF
        Args:
            question: Pergunta do usuário
        Returns:
            Resposta gerada pelo modelo
        """
        if not self.vector_store:
            raise ValueError("Pipeline não está totalmente configurado!")
            
        print(f"❓ Respondendo pergunta: {question}")
        
        # Recupera documentos relevantes
        docs = self.vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Gera o prompt e obtém a resposta
        prompt = self.generate_prompt(question, context)
        #response = self.llm.invoke(prompt)
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil e preciso."},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=1000,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erro ao processar a resposta: {str(e)}"

    def run_pipeline(self):
        """
        Executa todo o pipeline em sequência
        """
        self.load_pdf()
        self.split_text()
        self.create_embeddings()
        print("✅ Pipeline configurado e pronto para responder perguntas!")
