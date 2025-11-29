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
    print("🚀 TinyLlama RAG Asistanı başlatılıyor... Lütfen bekleyin.")

    # 1. BEYİN (TinyLlama)
    print("🧠 Model yükleniyor...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        # Mac M1/M2/M3 çipleri için float32 kararlılığı sağlar
        torch_dtype=torch.float32, 
        device_map="auto",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 2. HAFIZA
    print("📚 Hafıza yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    try:
        vector_store = FAISS.load_local("faiss_index_alzheimer_tr", embedding_model, allow_dangerous_deserialization=True)
    except Exception:
        # Eski langchain versiyonları için fallback
        vector_store = FAISS.load_local("faiss_index_alzheimer_tr", embedding_model)

    # 3. KURAL (PROMPT)
    template = """### Instruction:
    You are a helpful assistant. Use the context below to answer the question.
    The context is in English or Turkish. You must translate your reasoning and provide the final answer in TURKISH language.
    If the answer is not in the context, just say "Verilen metinde bu bilgi yok."

    ### Context:
    {context}

    ### Question:
    {question}

    ### Turkish Answer:"""

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
            sonuc = qa_chain.invoke({"query": soru})
            print("-" * 40)
            print(f"🗣️  CEVAP: {sonuc['result'].strip()}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    chat_baslat()