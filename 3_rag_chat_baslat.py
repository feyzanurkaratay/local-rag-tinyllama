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

# --- 1. AYARLAR VE MODEL ---
print("🚀 Sistem başlatılıyor... (Türkçe Zorlama Modu v3.0)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,          # Düşük sıcaklık (Yaratıcılığı kısıtla)
    top_p=0.90,
    repetition_penalty=1.1    # Tekrar cezasını biraz azalttık (Çok yüksek olunca dil bozulabiliyor)
)
llm = HuggingFacePipeline(pipeline=pipe)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA ---
print("📚 Hafıza kontrol ediliyor...")
# Veriyi her seferinde tazelemek en garantisi
loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. PROMPT (ÇOK KATI TÜRKÇE KURALLARI) ---
# İngilizce konuşmasını yasaklayan ve cevabı doğrudan veriden çekmesini sağlayan şablon
template = """<|system|>
Sen Türkçe konuşan uzman bir asistansın.
SANA VERİLEN BAĞLAMDAKİ BİLGİLERİ KULLANARAK CEVAP VER.
Kendi bilgini katma. Sadece TÜRKÇE cevap ver. İngilizce konuşma.

Bağlam:
{context}
</s>
<|user|>
Soru: {question}
</s>
<|assistant|>
Cevap:"""  # Cevap: diyerek başlamaya zorluyoruz

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 4. TEMİZLİK VE ZORLAMA FONKSİYONU ---
def cevapla(soru):
    if not soru:
        return ""
    
    # Modele soruyu sor
    ham_cevap = qa_chain.invoke({"query": soru})
    metin = ham_cevap["result"]
    
    # --- TEMİZLİK ANI ---
    # Modelin ürettiği cevabın içinden sadece gerekli kısmı al
    if "<|assistant|>" in metin:
        temiz_cevap = metin.split("<|assistant|>")[-1]
    else:
        temiz_cevap = metin
        
    # Eğer "Cevap:" kelimesi varsa ondan sonrasını al
    if "Cevap:" in temiz_cevap:
        temiz_cevap = temiz_cevap.split("Cevap:")[-1]

    # Hâlâ İngilizce "Sure!" veya "Here is..." gibi kalıplar varsa temizle (Basit filtre)
    yasakli_kelimeler = ["Sure", "Here is", "In this case", "Context:", "Question:"]
    for kelime in yasakli_kelimeler:
        temiz_cevap = temiz_cevap.replace(kelime, "")

    return temiz_cevap.strip()

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: Annem banyo yapmak istemiyor, ne yapmalıyım?"),
    outputs=gr.Textbox(label="Uzman Cevabı"),
    title="🧠 Alzheimer Asistanı (Türkçe v3.0)",
    description="Akademik ve pratik bakım rehberiniz. Sadece Türkçe cevap verir."
)

if __name__ == "__main__":
    arayuz.launch()
