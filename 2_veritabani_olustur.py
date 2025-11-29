from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import shutil

def veritabani_kur():
    # Klasör temizliği
    klasor_adi = "faiss_index_alzheimer_tr"
    if os.path.exists(klasor_adi):
        shutil.rmtree(klasor_adi)

    print("🚀 Süreç başlıyor: Vektör veritabanı oluşturuluyor...")

    # 1. Metin Dosyasını Yükle
    try:
        loader = TextLoader("alzheimer_veri.txt", encoding="utf-8")
        docs = loader.load()
    except FileNotFoundError:
        print("❌ HATA: 'alzheimer_veri.txt' bulunamadı. Önce 1_veri_olustur.py çalıştırın.")
        return

    # 2. Metni Parçalara Böl
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    parcalar = text_splitter.split_documents(docs)
    print(f"✂️  Belge {len(parcalar)} parçaya bölündü.")

    # 3. Embedding Modelini Hazırla (Multilingual)
    print("🧠 Embedding modeli yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 4. Veritabanını Oluştur
    print("💾 Veriler FAISS veritabanına kaydediliyor...")
    vector_store = FAISS.from_documents(parcalar, embedding_model)
    vector_store.save_local(klasor_adi)

    print("-" * 40)
    print(f"✅ BAŞARILI: Veritabanı '{klasor_adi}' klasörüne kaydedildi.")
    print("-" * 40)

if __name__ == "__main__":
    veritabani_kur()