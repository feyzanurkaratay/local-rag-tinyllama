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
from deep_translator import GoogleTranslator
import os
import sys

# --- 1. AYARLAR ---
print("🚀 Sistem Başlatılıyor... (Yönetici Dostu - Basit Dil Modu)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,          # Biraz esneklik verdik ki doğal konuşsun
    top_p=0.90,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=pipe)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA ---
print("📚 Hafıza yükleniyor...")
# Dosya kontrolü (Hugging Face'de bazen yol sorunu olabiliyor, garantiye alalım)
if not os.path.exists("alzheimer_veri.txt"):
    # Eğer dosya yoksa boş bir tane oluştur ki kod çökmesin
    with open("alzheimer_veri.txt", "w") as f: f.write("Veri yok.")

loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. PROMPT (BURASI DEĞİŞTİ!) ---
# Modele "Basit anlat, tıbbi terim kullanma" diyoruz.
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

# --- 4. TERCÜMAN FONKSİYONU ---
def cevapla(soru_tr):
    if not soru_tr:
        return ""
    
    try:
        # 1. Türkçeden İngilizceye çevir
        soru_en = GoogleTranslator(source='tr', target='en').translate(soru_tr)

        # 2. Modele İngilizce sor (Basit dil emri burada işliyor)
        ham_cevap = qa_chain.invoke({"query": soru_en})
        cevap_en = ham_cevap["result"]
        
        # Temizlik
        if "<|assistant|>" in cevap_en:
            cevap_en = cevap_en.split("<|assistant|>")[-1]
        
        # 3. Basitleştirilmiş İngilizce cevabı Türkçeye çevir
        cevap_tr = GoogleTranslator(source='en', target='tr').translate(cevap_en)

        return cevap_tr
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: İlaçları ne zaman vermeliyim?"),
    outputs=gr.Textbox(label="Asistanın Cevabı"),
    title="🧠 Alzheimer Asistanı (Basit Anlatım)",
    description="Tıbbi terimler olmadan, herkesin anlayacağı dilde cevap verir."
)

if __name__ == "__main__":
    arayuz.launch()
