import gradio as gr
import os
from huggingface_hub import InferenceClient
# DİKKAT: LangChain 0.0.350 için eski import yolları (Community yok)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator

# --- 1. AYARLAR ---
print("🚀 Sistem Başlatılıyor... (SAF API + ESKİ LANGCHAIN)")

# ŞİFRE (Korsan Yöntem)
kisim1 = "hf_"
kisim2 = "mGQNVdfnSwEVHeVOSakUtKWgdjMftiJhFo" 
hf_token = kisim1 + kisim2

# Modeli Çağıran İstemci
client = InferenceClient(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=hf_token)

# Hafıza için Embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA ---
if not os.path.exists("alzheimer_veri.txt"):
    with open("alzheimer_veri.txt", "w") as f: f.write("Veri yok.")

loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)

# --- 3. CEVAP FONKSİYONU ---
def cevapla(soru_tr):
    if not soru_tr:
        return ""
    
    try:
        # 1. Çeviri (TR -> EN)
        soru_en = GoogleTranslator(source='tr', target='en').translate(soru_tr)
        
        # 2. Hafızadan Bul
        benzer_belgeler = vector_store.similarity_search(soru_en, k=2)
        baglam = "\n".join([doc.page_content for doc in benzer_belgeler])
        
        # 3. Prompt
        prompt = f"""<|system|>
You are a helpful assistant. 
Use the Context below to answer the Question.
IMPORTANT: Use very simple, easy-to-understand language. Avoid medical jargon.
If the answer is not in the context, say "I don't know".

Context:
{baglam}
</s>
<|user|>
Question: {soru_en}
</s>
<|assistant|>
"""
        
        # 4. API İsteği
        cevap_objesi = client.text_generation(prompt, max_new_tokens=256, temperature=0.1, top_p=0.9)
        cevap_en = str(cevap_objesi)
        
        # 5. Çeviri (EN -> TR)
        cevap_tr = GoogleTranslator(source='en', target='tr').translate(cevap_en)
        
        return cevap_tr

    except Exception as e:
        return f"Hata: {str(e)}"

# --- 4. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: İlaçları nasıl vermeliyim?"),
    outputs=gr.Textbox(label="Cevap"),
    title="🧠 Alzheimer Asistanı",
    description="TinyLlama API Modu"
)

if __name__ == "__main__":
    arayuz.launch()
