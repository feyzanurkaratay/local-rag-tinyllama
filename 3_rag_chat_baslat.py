import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
import sys
import os

# Uyarıları gizle
warnings.filterwarnings("ignore")

def chat_baslat():
    print("🚀 Masaüstü Asistanı Başlatılıyor... (Keskin Nişancı Modu)")

    # 1. BEYİN (TinyLlama)
    print("🧠 Model yükleniyor...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        # Mac için float32 (Windows ise bfloat16 denenebilir ama float32 garantidir)
        torch_dtype=torch.float32, 
        device_map="auto",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,         # Yaratıcılık kapalı (Ciddiyet modu)
        top_p=0.90,
        repetition_penalty=1.2   # Papağan modunu engelle
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 2. HAFIZA
    print("📚 Hafıza yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Hafıza klasörünü kontrol et
    if not os.path.exists("faiss_index_alzheimer_tr"):
        print("❌ HATA: Hafıza bulunamadı! Önce '2_veritabani_olustur.py' çalıştırın.")
        return

    try:
        vector_store = FAISS.load_local("faiss_index_alzheimer_tr", embedding_model, allow_dangerous_deserialization=True)
    except:
        vector_store = FAISS.load_local("faiss_index_alzheimer_tr", embedding_model)

    # 3. KATI PROMPT (YÖNERGE)
    template = """<|system|>
Sen uzman bir Alzheimer asistanısın. SANA VERİLEN BAĞLAMI TEKRAR ETME.
Aşağıdaki bilgiyi analiz et ve soruya kısa, net bir Türkçe cevap ver.
Cevabı verdikten sonra hemen sus.

Bilgi (Bağlam):
{context}
</s>
<|user|>
Soru: {question}
</s>
<|assistant|>
"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # 4. ZİNCİR
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}), # Sadece en alakalı 2 parça
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("\n" + "*"*50)
    print("🤖 UZMAN ASİSTAN HAZIR! (Çıkmak için 'q' yazın)")
    print("*"*50)

    # 5. SOHBET DÖNGÜSÜ
    while True:
        try:
            soru = input("\n🤔 Sorunuz: ")
            if soru.lower() in ["q", "çıkış", "exit"]:
                print("👋 Görüşmek üzere!")
                break
            if not soru.strip():
                continue
            
            print("... Analiz ediliyor ...")
            
            # Cevabı al
            ham_cevap = qa_chain.invoke({"query": soru})
            metin = ham_cevap['result']

            # --- TEMİZLİK ROBOTU ---
            # Cevabın içindeki teknik etiketleri ve tekrarları temizle
            if "<|assistant|>" in metin:
                temiz_cevap = metin.split("<|assistant|>")[-1]
            else:
                temiz_cevap = metin
            
            if "Bağlam:" in temiz_cevap:
                temiz_cevap = temiz_cevap.split("Bağlam:")[0]

            print("-" * 40)
            print(f"🗣️  CEVAP: {temiz_cevap.strip()}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    chat_baslat()
