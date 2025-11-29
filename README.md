# 🦙 TinyLlama ile Yerel RAG (Retrieval-Augmented Generation)

Bu proje, internet bağlantısına ihtiyaç duymadan yerel bilgisayar üzerinde çalışan, belge tabanlı bir soru-cevap asistanıdır. **TinyLlama-1.1B** dil modeli ve **LangChain** çerçevesi kullanılarak geliştirilmiştir.

## 🚀 Özellikler
* **Tamamen Yerel:** Verileriniz buluta gitmez, tamamen kendi bilgisayarınızda işlenir.
* **Kaynak Dostu:** Küçük boyutlu TinyLlama modeli kullanıldığı için standart bilgisayarlarda (Mac M1/M2 dahil) çalışır.
* **Türkçe Yanıt:** Model İngilizce olsa bile, özel Prompt Mühendisliği ile Türkçe yanıt üretir.
* **Vektör Hafıza:** FAISS kullanılarak veriler hızlı erişim için vektör veritabanında saklanır.

## 🛠️ Kurulum

1. Depoyu klonlayın veya indirin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt