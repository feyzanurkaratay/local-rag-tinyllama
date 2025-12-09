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

# --- 1. AYARLAR ---
print("🚀 Sistem Başlatılıyor... (Genel Uzman Modu)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=512,       # Daha uzun cevaplar verebilsin
    do_sample=True,
    temperature=0.4,          # Yaratıcılığı artırdık (Daha doğal konuşsun)
    top_p=0.92,
    repetition_penalty=1.1
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 2. HAFIZA ---
print("📚 Hafıza yükleniyor...")
# Veri dosyası varsa yükle, yoksa hata verme (Sadece genel bilgiyle çalışabilsin diye)
vector_store = None
if os.path.exists("alzheimer_veri.txt"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    parcalar = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(parcalar, embedding_model)
    print("✅ Yerel veri kaynağı (RAG) yüklendi.")
else:
    print("⚠️ UYARI: Veri dosyası bulunamadı. Model sadece genel bilgisiyle cevap verecek.")

# --- 3. HİBRİT PROMPT (KİLİT NOKTA) ---
# Modele diyoruz ki: Önce elindeki nota bak, orada yoksa bildiğin gibi anlat.
template = """<|system|>
Sen Alzheimer konusunda uzman, yardımsever bir asistansın.
Sana bir BAĞLAM (Context) verilecek. 
Önce bu bağlamdaki bilgileri kullan. Eğer sorunun cevabı bağlamda yoksa, KENDİ GENEL BİLGİNİ kullanarak cevapla.
Her zaman TÜRKÇE cevap ver.

BAĞLAM:
{context}
</s>
<|user|>
SORU: {question}
</s>
<|assistant|>
"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# --- 4. ZİNCİRİ KUR ---
if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
else:
    # Eğer veritabanı yoksa düz LLM zinciri (Fallback)
    qa_chain = None 

# --- 5. CEVAP FONKSİYONU ---
def cevapla(soru):
    if not soru:
        return ""
    
    try:
        if qa_chain:
            # RAG ile cevapla (Veri + Genel Bilgi)
            ham_cevap = qa_chain.invoke({"query": soru})
            metin = ham_cevap["result"]
        else:
            # Sadece modelin kendi bilgisiyle cevapla
            prompt = f"<|user|>\n{soru}\n</s>\n<|assistant|>\n"
            metin = pipe(prompt)[0]['generated_text']

        # Temizlik
        if "<|assistant|>" in metin:
            temiz_cevap = metin.split("<|assistant|>")[-1]
        else:
            temiz_cevap = metin

        return temiz_cevap.strip()
        
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# --- 6. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: Alzheimer hastaları araba kullanabilir mi?"),
    outputs=gr.Textbox(label="Uzman Cevabı"),
    title="🧠 Alzheimer Uzman Asistanı (Geniş Kapsamlı)",
    description="Hem yüklenen verileri hem de genel tıbbi bilgiyi kullanarak cevap verir."
)

if __name__ == "__main__":
    arayuz.launch(inbrowser=True)
