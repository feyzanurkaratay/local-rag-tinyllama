import gradio as gr
import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
import os
import sys

# --- 1. AYARLAR ---
print("🚀 Sistem TinyLlama ile başlatılıyor... (YEREL MOD - DÜZELTİLDİ)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,
    top_p=0.90,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=pipe)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA KONTROLÜ ---
print("📚 Hafıza yükleniyor...")
if not os.path.exists("faiss_index_alzheimer"):
    print("❌ HATA: Veritabanı klasörü bulunamadı!")
    print("Lütfen önce 2_veritabani_olustur.py dosyasını çalıştırın.")
    sys.exit()

# DÜZELTME BURADA YAPILDI:
# 'allow_dangerous_deserialization=True' kısmını kaldırdık.
# Artık senin bilgisayarındaki eski sürümle de çalışır.
try:
    vector_store = FAISS.load_local("faiss_index_alzheimer", embedding_model)
except TypeError:
    # Eğer çok yeni sürüm varsa ve güvenlik uyarısı verirse diye önlem:
    vector_store = FAISS.load_local("faiss_index_alzheimer", embedding_model, allow_dangerous_deserialization=True)

print("✅ Hafıza hazır!")

# --- 3. PROMPT ---
template = """<|system|>
You are a helpful and friendly assistant. 
Use the Context below to answer the Question.
IMPORTANT: Use very simple, easy-to-understand language. Avoid medical jargon. 
Explain it as if you are talking to a friend.
If the answer is not in the context, say "I don't know".

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

# --- 4. CEVAP FONKSİYONU ---
def cevapla(soru_tr):
    if not soru_tr:
        return ""
    
    try:
        # 1. TR -> EN
        print(f"🇹🇷 Gelen Soru: {soru_tr}")
        soru_en = GoogleTranslator(source='tr', target='en').translate(soru_tr)
        print(f"🇺🇸 Çevrilen Soru: {soru_en}")

        # 2. Model Cevabı
        ham_cevap = qa_chain.invoke({"query": soru_en})
        cevap_en = ham_cevap["result"]
        
        # Temizlik
        if "<|assistant|>" in cevap_en:
            cevap_en = cevap_en.split("<|assistant|>")[-1]
        
        # 3. EN -> TR
        cevap_tr = GoogleTranslator(source='en', target='tr').translate(cevap_en)
        
        return cevap_tr
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: Annem banyo yapmak istemiyor, ne yapmalıyım?"),
    outputs=gr.Textbox(label="Asistanın Cevabı"),
    title="🧠 Alzheimer Asistanı (Yerel Versiyon)",
    description="TinyLlama modeli bilgisayarınızda çalışır."
)

if __name__ == "__main__":
    arayuz.launch(inbrowser=True)
