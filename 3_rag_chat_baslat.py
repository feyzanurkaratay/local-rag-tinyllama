import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
import sys

# Uyarıları gizle
warnings.filterwarnings("ignore")

def chat_baslat():
    print("🚀 TinyLlama RAG Asistanı başlatılıyor... (Düzeltilmiş Versiyon)")

    # 1. BEYİN (TinyLlama)
    print("🧠 Model yükleniyor...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float32, 
        device_map="auto",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,    # Daha tutarlı olması için düşürdük
        top_p=0.95,
        repetition_penalty=1.15  # <--- İŞTE SİHİRLİ AYAR! (Tekrar etmeyi engeller)
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 2. HAFIZA
    print("📚 Hafıza yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    try:
        # Güvenlik uyarısını aşmak için allow_dangerous_deserialization=True
        vector_store = FAISS.load_local("faiss_index_alzheimer_tr", embedding_model, allow_dangerous_deserialization=True)
    except:
        # Eski versiyonlar için yedek
        vector_store = FAISS.load_local("faiss_index_alzheimer_tr", embedding_model)

    # 3. KURAL (PROMPT) - TinyLlama'nın Kendi Özel Formatı
    # Bu format modelin nerede durması gerektiğini netleştirir.
    template = """<|system|>
Sen yardımcı bir asistansın. Aşağıdaki bağlamı (Context) kullanarak soruyu cevapla.
Cevabı verdikten sonra dur. Sadece TÜRKÇE konuş.

Bağlam:
{context}
</s>
<|user|>
{question}
</s>
<|assistant|>
"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # 4. ZİNCİR
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("\n" + "*"*50)
    print("🤖 ASİSTAN HAZIR! (Çıkmak için 'q' yazın)")
    print("*"*50)

    while True:
        try:
            soru = input("\n🤔 Sorunuz: ")
            if soru.lower() in ["q", "çıkış", "exit"]:
                print("👋 Görüşmek üzere!")
                break
            if not soru.strip():
                continue
            
            print("... Yanıt hazırlanıyor ...")
            # invoke yerine __call__ veya run kullanarak eski versiyon uyumluluğunu artıralım
            sonuc = qa_chain.invoke({"query": soru})
            
            print("-" * 40)
            # Cevabın sadece ilgili kısmını alıp temizleyelim
            cevap = sonuc['result']
            
            # Eğer model yine de saçmalarsa temizlemek için ek güvenlik:
            if "<|assistant|>" in cevap:
                cevap = cevap.split("<|assistant|>")[-1]
            
            print(f"🗣️  CEVAP: {cevap.strip()}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    chat_baslat()
