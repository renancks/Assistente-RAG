# 🤖 PDF Question Answering with RAG Pipeline

Este projeto implementa um sistema de Recuperação Aumentada por Geração (RAG) que permite fazer perguntas sobre o conteúdo de documentos PDF. O sistema utiliza técnicas avançadas de processamento de linguagem natural e modelos de linguagem para fornecer respostas precisas baseadas no conteúdo do documento.

![Demo do Projeto](./assets/demo.gif)

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Pipeline RAG](#-pipeline-rag)
- [Exemplos de Uso](#-exemplos-de-uso)
- [Contribuições](#-contribuições)
- [Licença](#-licença)

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido como parte de um teste prático, com o objetivo de criar um sistema capaz de:
- Extrair texto de documentos PDF
- Gerar embeddings do texto extraído
- Utilizar um modelo de linguagem para responder perguntas baseadas no conteúdo
- Implementar um pipeline RAG completo

## ✨ Funcionalidades

- **Upload de PDF**: Interface amigável para carregar documentos PDF
- **Extração de Texto**: Processamento eficiente do conteúdo do PDF
- **Geração de Embeddings**: Conversão do texto em representações vetoriais
- **Resposta a Perguntas**: Capacidade de responder perguntas sobre o conteúdo do documento
- **Interface Web**: Interface intuitiva construída com Streamlit
- **Integração com Groq**: Utilização do modelo Mixtral-8x7B para respostas precisas

## 🛠 Tecnologias Utilizadas

- **Python 3.8+**
- **Groq**: API de modelo de linguagem de alta performance
- **LangChain**: Framework para desenvolvimento de aplicações com LLMs
- **LLAMA Index**: Biblioteca para indexação e consulta de documentos
- **Sentence Transformers**: Geração de embeddings
- **FAISS**: Biblioteca para busca eficiente de similaridade
- **Streamlit**: Framework para criação de interfaces web
- **Hugging Face Transformers**: Acesso a modelos de linguagem

## 📁 Estrutura do Projeto

```
RAG_Teste/
├── app.py                 # Aplicação principal com interface Streamlit
├── requirements.txt       # Dependências do projeto
├── README.md             # Documentação
├── src/
│   ├── __init__.py
│   └── pipeline.py       # Implementação do pipeline RAG
└── assets/
    └── demo.gif          # GIF demonstrativo da aplicação
```

## 🚀 Como Executar

1. **Clone o repositório**
```bash
git clone https://github.com/renancks/Assistente-RAG.git
cd Assistente-RAG
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure o arquivo .env**
```bash
# Crie um arquivo .env na raiz do projeto
touch .env  # Linux/Mac
# ou
type nul > .env  # Windows

# Adicione sua chave da API Groq no arquivo .env:
GROQ_API_KEY=sua_chave_aqui
```

5. **Execute a aplicação**
```bash
streamlit run app.py
```

## 🔄 Pipeline RAG

O pipeline implementado segue as seguintes etapas:

1. **Carregamento do PDF**
   - Utiliza LLAMA Index para extrair texto do PDF
   - Preserva a estrutura e formatação do documento

2. **Processamento do Texto**
   - Chunks de 500 caracteres com 50 de sobreposição
   - Preservação de contexto entre chunks
   - Limpeza e normalização do texto

3. **Geração de Embeddings**
   - Modelo all-MiniLM-L6-v2 para embeddings
   - Dimensionalidade otimizada para performance
   - Normalização de embeddings para melhor similaridade

4. **Armazenamento e Recuperação**
   - Utiliza FAISS para indexação eficiente
   - Permite busca rápida por similaridade
   - Cache de resultados para queries frequentes


5. **Geração de Respostas**
   - Integração com Groq API
   - Modelo Mixtral-8x7B para respostas
   - Sistema de prompts otimizado para contexto

## 💡 Exemplos de Uso

1. **Carregue um PDF**
   - Clique no botão "Upload do PDF"
   - Selecione um arquivo PDF do seu computador

2. **Faça uma Pergunta**
   - Digite sua pergunta na caixa de texto
   - Aguarde a resposta baseada no conteúdo do PDF

![Exemplo de Uso](./assets/example.png)

## 📊 Performance e Limitações

- Tamanho máximo de documento: 100 páginas
- Tempo médio de processamento: 2-3 segundos por página
- Uso de memória: ~500MB para documentos típicos
- Limitações da API Groq: Conforme plano de uso

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir novas funcionalidades
- Melhorar a documentação
- Submeter pull requests


---

Desenvolvido por [Renan Santos Ferreira](https://github.com/renancks) 👋
```


