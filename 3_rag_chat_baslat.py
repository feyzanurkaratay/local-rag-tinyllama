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
print("🚀 Sistem TinyLlama ile başlatılıyor... (%100 Türkçe Modu)")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    # SICAKLIK AYARI ÇOK ÖNEMLİ:
    # 0.1 yaptık ki hayal kurmasın, sadece metni okusun.
    temperature=0.1,          
    top_p=0.90,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=pipe)

# Türkçe için en iyi embedding modeli
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- 2. HAFIZA KONTROLÜ ---
print("📚 Hafıza yükleniyor...")
if not os.path.exists("alzheimer_veri.txt"):
    print("❌ HATA: 'alzheimer_veri.txt' dosyası bulunamadı!")
    print("Lütfen önce 1_veri_olustur.py dosyasını çalıştırarak veriyi oluşturun.")
    sys.exit()

loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
docs = loader.load()

# Metni daha küçük parçalara bölüyoruz ki odaklanabilsin
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
parcalar = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(parcalar, embedding_model)
print("✅ Hafıza hazır!")

# --- 3. TÜRKÇE PROMPT (KOMUT) ---
# TinyLlama'ya Türkçe emir veriyoruz ama <|system|> etiketleri ile ciddiyet katıyoruz.
template = """<|system|>
Sen sadece aşağıdaki METİN içindeki bilgileri kullanan bir asistansın.
Dışarıdan bilgi ekleme. Uydurma yapma.
Soruyu sadece METİN'e bakarak TÜRKÇE cevapla.

METİN:
{context}
</s>
<|user|>
SORU: {question}
</s>
<|assistant|>
"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    # k=2 yaptık. En alakalı 2 parçayı getirsin.
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 4. CEVAP TEMİZLEME MOTORU ---
def cevapla(soru):
    if not soru:
        return ""
    
    # 1. Cevabı üret
    ham_cevap = qa_chain.invoke({"query": soru})
    metin = ham_cevap["result"]
    
    # 2. Teknik etiketleri temizle (<|assistant|> vb.)
    if "<|assistant|>" in metin:
        temiz_cevap = metin.split("<|assistant|>")[-1]
    else:
        temiz_cevap = metin

    # 3. İNGİLİZCE FİLTRESİ (Eğer İngilizce başlarsa uyar)
    if "The provided text" in temiz_cevap or "Sure!" in temiz_cevap:
        return "⚠️ Model İngilizce cevap vermeye çalıştı. Lütfen soruyu biraz daha farklı sorabilir misiniz?"

    # 4. Gereksiz başlıkları kes (Model bazen metindeki diğer başlıkları da okur)
    kesilecekler = ["BÖLÜM", "Soru:", "BAŞLIK:", "Tanım:"]
    for kelime in kesilecekler:
        # Eğer cevap çok kısaysa (10 karakterden az) kesme, belki cevap o kelimeyle başlıyordur.
        if kelime in temiz_cevap and len(temiz_cevap) > 50: 
             # Kelimenin geçtiği yerden sonrasını at
             parca = temiz_cevap.split(kelime)
             if len(parca[0]) > 5: # Eğer ilk parça mantıklıysa onu al
                 temiz_cevap = parca[0]

    return temiz_cevap.strip()

# --- 5. ARAYÜZ ---
arayuz = gr.Interface(
    fn=cevapla,
    inputs=gr.Textbox(lines=2, placeholder="Örn: Annem banyo yapmak istemiyor, ne yapmalıyım?"),
    outputs=gr.Textbox(label="Cevap"),
    title="🇹🇷 Türkçe RAG Asistanı (TinyLlama)",
    description="Sadece yüklenen Türkçe veriyi kullanarak cevap verir."
)

if __name__ == "__main__":
    # Tarayıcıda otomatik açılması için inbrowser=True ekledik
    arayuz.launch(inbrowser=True)
