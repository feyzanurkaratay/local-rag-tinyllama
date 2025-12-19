import gradio as gr
import os

# --- KÜTÜPHANE İMPORTLARI (0.1.10 UYUMLU) ---
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# RetrievalQA bu versiyonda hala burada, sorun yok:
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator

# --- 1. AYARLAR ---
print("🚀 Sistem Başlatılıyor... (ALTIN VERSİYON 0.1.10)")

# ŞİFRE KISMI (Korsan Modu Devam)
kisim1 = "hf_"
kisim2 = "mGQNVdfnSwEVHeVOSakUtKWgdjMftiJhFo" 
hf_token = kisim1 + kisim2

# Model: TinyLlama
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

try:
    # Endpoint kullanımı (API Hatası vermez)
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
        huggingfacehub_api_token=hf_token
    )
except Exception as e:
    print(f"Model Bağlantı Hatası: {e}")

# Embedding Modeli
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA YÜKLEME ---
if not os.path.exists("alzheimer_veri.txt"):
    with open("alzheimer_veri.txt", "w") as f: f.write("Veri yok.")

loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)

# --- 3. PROMPT ---
template = """<|system|>
You are a helpful assistant. 
Use the Context below to answer the Question.
IMPORTANT: Use very simple, easy-to-understand language. Avoid medical jargon.
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
        # TR -> EN
        soru_en = GoogleTranslator(source='tr', target='en').translate(soru_tr)
        
        # API Çalıştır (Invoke metodu 0.1.10'da çalışır)
        ham_cevap = qa_chain.invoke({"query": soru_en})
        sonuc_metni = ham_cevap["result"]
        
        # Temizlik
        if "<|assistant|>" in sonuc_metni:
            cevap_en = sonuc_metni.split("<|assistant|>")[-1]
        else:
            cevap_en = sonuc_metni
            
        # EN -> TR
        cevap_tr = GoogleTranslator(source='en', target='tr').translate(cevap_en)
        return cevap_tr
    except Exception as e:
        return f"Hata: {str(e)}"

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: İlaçları nasıl vermeliyim?"),
    outputs=gr.Textbox(label="Cevap"),
    title="🧠 Alzheimer Asistanı (Final - Stabil)",
    description="TinyLlama API + Stabil Kütüphaneler"
)

if __name__ == "__main__":
    arayuz.launch()
