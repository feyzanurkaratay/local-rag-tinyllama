import gradio as gr
import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- 1. AYARLAR ---
print("🚀 Sistem TinyLlama ile başlatılıyor... (Makas Modu)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.1,          
    top_p=0.90,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=pipe)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA ---
print("📚 Hafıza yükleniyor...")
loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

# Chunk size'ı biraz küçülttük ki gereksiz diğer konuları almasın
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. PROMPT ---
template = """<|system|>
You are a helpful assistant. 
Read the following CONTEXT carefully. It is in Turkish.
Answer the QUESTION using ONLY the information from the CONTEXT.
Answer in TURKISH language.

CONTEXT:
{context}
</s>
<|user|>
QUESTION: {question}
</s>
<|assistant|>
"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 4. TEMİZLİK FONKSİYONU (YENİ!) ---
def cevapla(soru):
    if not soru:
        return ""
    
    ham_cevap = qa_chain.invoke({"query": soru})
    metin = ham_cevap["result"]
    
    # 1. Asistan etiketinden sonrasını al
    if "<|assistant|>" in metin:
        temiz_cevap = metin.split("<|assistant|>")[-1]
    else:
        temiz_cevap = metin

    # 2. MAKASLAMA İŞLEMİ (YENİ) ✂️
    # Eğer model hızını alamayıp diğer "BÖLÜM" başlıklarına veya "Soru:" kısımlarına geçerse kes.
    kesilecek_kelimeler = ["BÖLÜM", "Bölüm", "Soru:", "3.", "4."]
    
    for kelime in kesilecek_kelimeler:
        if kelime in temiz_cevap:
            # Kelimeyi bulduğu yerden sonrasını at, öncesini al
            temiz_cevap = temiz_cevap.split(kelime)[0]

    return temiz_cevap.strip()

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: Annem banyo yapmak istemiyor, ne yapmalıyım?"),
    outputs=gr.Textbox(label="TinyLlama Cevabı"),
    title="🧠 TinyLlama Türkçe Asistanı",
    description="TinyLlama modeli ile yerel ve güvenli Alzheimer rehberi."
)

if __name__ == "__main__":
    arayuz.launch()
