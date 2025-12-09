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
import sys

# --- 1. MODEL AYARLARI ---
print("🚀 Sistem Başlatılıyor... (DİKTATÖR MODU)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.1,          # Yaratıcılık KAPALI. Sadece okuduğunu söyler.
    top_p=0.90,
    repetition_penalty=1.2    # Tekrar etmeyi engeller.
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 2. HAFIZA YÜKLEME ---
print("📚 Hafıza yükleniyor...")
if not os.path.exists("alzheimer_veri.txt"):
    print("❌ HATA: Veri dosyası yok! Önce 1_veri_olustur.py çalıştır.")
    sys.exit()

loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

# Chunk'ları büyüttük (500) ki konu bütünlüğü bozulmasın
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parcalar = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. SERT PROMPT (YORUM YOK, SADECE OKU) ---
template = """<|system|>
You are a strict assistant. 
Read the Turkish CONTEXT below.
Answer the QUESTION using ONLY the CONTEXT.
If the answer is not in the context, say "Bilmiyorum".
Answer in TURKISH.

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
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 4. CEVAP TEMİZLEME ---
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

    # Eğer İngilizce başlarsa uyar
    if "Sure!" in temiz_cevap or "Here is" in temiz_cevap:
        return "⚠️ Model İngilizceye kaçtı. Lütfen soruyu 'Araba kullanabilir mi?' şeklinde net sorun."
        
    return temiz_cevap.strip()

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: Araba kullanabilir mi?"),
    outputs=gr.Textbox(label="Cevap"),
    title="🧠 Alzheimer Asistanı (Sıkı Yönetim)",
    description="Sadece veri tabanındaki doğru bilgileri verir. Uydurmaz."
)

if __name__ == "__main__":
    arayuz.launch(inbrowser=True)
