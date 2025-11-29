import os

def veri_dosyasi_olustur():
    """
    RAG sistemi için temiz bir Alzheimer bilgi metni oluşturur.
    """
    metin = """
    Alzheimer Hastalığı ve Belirtileri

    Alzheimer hastalığı, beyin hücrelerinin zamanla ölmesine neden olan ilerleyici bir nörolojik bozukluktur. 
    Hastalığın en yaygın ve belirgin semptomu hafıza kaybıdır. Başlangıçta kişi yakın zamandaki olayları veya konuşmaları hatırlamakta zorluk çeker.

    Hastalık ilerledikçe, Alzheimer hastalarında şu ana belirtiler görülür:
    1. Hafıza Kaybı: Özellikle yeni öğrenilen bilgileri unutma.
    2. Planlama ve Problem Çözmede Zorluk: Hesap yapma veya tarifleri takip etmede güçlük.
    3. Günlük İşleri Yapmada Zorluk: Tanıdık bir yere gitme veya oyun kurallarını hatırlama sorunu.
    4. Zaman ve Yer Karmaşası: Tarihleri, mevsimleri ve zamanın akışını karıştırma.
    5. Görsel ve Uzamsal Algı Sorunları: Okumada, mesafeyi tahmin etmede zorluk.
    6. Kelime Bulma Güçlüğü: Konuşurken veya yazarken doğru kelimeyi bulamama.
    7. Eşyaları Kaybetme: Eşyaları alışılmadık yerlere koyma ve geri bulamama.
    8. Yargı ve Karar Verme Yeteneğinde Azalma: Kişisel bakımın azalması veya parayı yönetememe.
    9. Sosyal hayattan çekilme ve kişilik değişiklikleri.

    Bu belirtiler kişiden kişiye değişiklik gösterebilir ancak hafıza kaybı en temel işarettir.
    """

    dosya_adi = "alzheimer_veri.txt"
    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write(metin)

    print(f"✅ Başarılı: '{dosya_adi}' dosyası oluşturuldu.")

if __name__ == "__main__":
    veri_dosyasi_olustur()