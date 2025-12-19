from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def veritabani_olustur():
    # 1. Metni Yükle
    if not os.path.exists("alzheimer_veri.txt"):
        print("❌ HATA: 'alzheimer_veri.txt' bulunamadı! Önce 1. kodu çalıştır.")
        return

    loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
    documents = loader.load()

    # 2. Metni Parçala
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 3. Embedding Modelini İndir
    print("⏳ Embedding modeli yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 4. Vektör Veritabanını Oluştur
    print("⏳ Vektör veritabanı oluşturuluyor...")
    vector_store = FAISS.from_documents(docs, embedding_model)

    # 5. Kaydet
    vector_store.save_local("faiss_index_alzheimer")
    print("✅ BAŞARILI! Veritabanı 'faiss_index_alzheimer' klasörüne kaydedildi.")

if __name__ == "__main__":
    veritabani_olustur()
