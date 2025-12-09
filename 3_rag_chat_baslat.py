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
print("🚀 Sistem TinyLlama ile başlatılıyor... (Hibrit Komut Modu)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.1,          # Yaratıcılık neredeyse kapalı
    top_p=0.90,
    repetition_penalty=1.2    # Tekrarı engelle
)
llm = HuggingFacePipeline(pipeline=pipe)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA ---
print("📚 Hafıza yükleniyor...")
loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. HİBRİT PROMPT (Sır Burada!) ---
# Modele İngilizce emir verip, Türkçe çıktı istiyoruz.
# Bu yöntem TinyLlama'nın performansını %100 artırır.
template = """<|system|>
You are a helpful assistant. 
Read the following CONTEXT carefully. It is in Turkish.
Answer the QUESTION using ONLY the information from the CONTEXT.
Answer in TURKISH language. Do not invent information.

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
    # k=1 yaptık. Sadece EN iyi cevabı alsın, kafası karışmasın.
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 4. TEMİZLİK ---
def cevapla(soru):
    if not soru:
        return ""
    
    ham_cevap = qa_chain.invoke({"query": soru})
    metin = ham_cevap["result"]
    
    # Modelin teknik etiketlerini temizle
    if "<|assistant|>" in metin:
        temiz_cevap = metin.split("<|assistant|>")[-1]
    else:
        temiz_cevap = metin
        
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
