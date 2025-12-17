import gradio as gr
import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator

# --- 1. AYARLAR VE API BAĞLANTISI ---
print("🚀 Sistem Başlatılıyor... (TinyLlama API + Tercüman Modu)")

# Hugging Face Gizli Anahtarını alıyoruz
hf_token = os.getenv("HF_TOKEN")

# TinyLlama'yı API üzerinden çağırıyoruz (İndirme yok, CPU yorulmaz)
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.1,         # TinyLlama için düşük sıcaklık şart
    top_p=0.9,
    huggingfacehub_api_token=hf_token
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA ---
print("📚 Hafıza yükleniyor...")
if not os.path.exists("alzheimer_veri.txt"):
    with open("alzheimer_veri.txt", "w") as f: f.write("Veri yok.")

loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. PROMPT (İNGİLİZCE) ---
# TinyLlama İngilizce anladığı için prompt İngilizce kalıyor.
# Modele "Basit anlat" (simple language) emrini burada veriyoruz.
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

# --- 4. TERCÜMANLI CEVAP FONKSİYONU ---
def cevapla(soru_tr):
    if not soru_tr:
        return ""
    
    try:
        # 1. Türkçeden İngilizceye çevir (API'ye gitmeden önce)
        print(f"🇹🇷 Gelen: {soru_tr}")
        soru_en = GoogleTranslator(source='tr', target='en').translate(soru_tr)
        
        # 2. API'ye İngilizce sor
        # (İşlem Hugging Face sunucusunda yapılır)
        ham_cevap = qa_chain.invoke({"query": soru_en})
        cevap_en = ham_cevap["result"]
        
        # Temizlik
        if "<|assistant|>" in cevap_en:
            cevap_en = cevap_en.split("<|assistant|>")[-1]
            
        # 3. İngilizce cevabı Türkçeye çevir
        cevap_tr = GoogleTranslator(source='en', target='tr').translate(cevap_en)
        
        return cevap_tr

    except Exception as e:
        return f"Hata oluştu (Token veya Bağlantı): {str(e)}"

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: İlaçları nasıl vermeliyim?"),
    outputs=gr.Textbox(label="TinyLlama Cevabı (API)"),
    title="🧠 TinyLlama Asistanı (API + Tercüman)",
    description="TinyLlama modeli API üzerinden çalışır, sistem çeviri yaparak Türkçe konuşur."
)

if __name__ == "__main__":
    arayuz.launch()
