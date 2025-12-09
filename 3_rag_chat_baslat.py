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
from deep_translator import GoogleTranslator  # <--- YENİ OYUNCUMUZ
import os
import sys

# --- 1. AYARLAR ---
print("🚀 Sistem TinyLlama + Tercüman Modu ile başlatılıyor...")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,          # İngilizce konuşacağı için rahat olabilir
    top_p=0.90,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=pipe)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA ---
print("📚 Hafıza yükleniyor...")
if not os.path.exists("alzheimer_veri.txt"):
    print("❌ HATA: 'alzheimer_veri.txt' yok! Önce 1_veri_olustur.py çalıştır.")
    sys.exit()

loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. PROMPT (TAMAMEN İNGİLİZCE) ---
# Modele İngilizce davranıyoruz ki kafası karışmasın.
template = """<|system|>
You are a helpful assistant. 
Use the Context below to answer the Question.
If the answer is not in the context, say "I don't know".
Keep your answer short and concise.

Context:
{context}
</s>
<|user|>
Question: {question}
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

# --- 4. TERCÜMAN FONKSİYONU ---
def cevapla(soru_tr):
    if not soru_tr:
        return ""
    
    try:
        # 1. Soruyu Türkçeden İngilizceye çevir
        print(f"🇹🇷 Gelen Soru: {soru_tr}")
        soru_en = GoogleTranslator(source='tr', target='en').translate(soru_tr)
        print(f"🇺🇸 Çevrilen Soru: {soru_en}")

        # 2. Modele İngilizce sor
        ham_cevap = qa_chain.invoke({"query": soru_en})
        cevap_en = ham_cevap["result"]
        
        # Temizlik (Teknik etiketleri at)
        if "<|assistant|>" in cevap_en:
            cevap_en = cevap_en.split("<|assistant|>")[-1]
        
        print(f"🤖 Model Cevabı (EN): {cevap_en.strip()}")

        # 3. Cevabı Türkçeye çevir
        cevap_tr = GoogleTranslator(source='en', target='tr').translate(cevap_en)
        print(f"🇹🇷 Sonuç: {cevap_tr}")

        return cevap_tr
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: Annem banyo yapmak istemiyor, ne yapmalıyım?"),
    outputs=gr.Textbox(label="Türkçe Cevap"),
    title="🧠 TinyLlama Türkçe Asistanı (Tercümanlı)",
    description="Siz Türkçe sorun, TinyLlama İngilizce düşünsün, biz size Türkçe söyleyelim."
)

if __name__ == "__main__":
    arayuz.launch(inbrowser=True)
