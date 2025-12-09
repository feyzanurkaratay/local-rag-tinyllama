import os

def veri_dosyasi_olustur():
    """
    RAG sistemi için KAPSAMLI Alzheimer bilgi metni oluşturur.
    İçinde iletişim, bakım, güvenlik ve ev düzeni gibi konular yer alır.
    """
    metin = """
    KONU: ALZHEIMER HASTALIĞI KAPSAMLI REHBERİ

    BÖLÜM 1: ALZHEIMER NEDİR VE BELİRTİLERİ
    Alzheimer, hafıza, düşünme ve davranış problemlerine neden olan ilerleyici bir beyin hastalığıdır. 
    En yaygın belirtiler şunlardır:
    - Yeni öğrenilen bilgileri unutma ve sürekli aynı soruları sorma.
    - Tarihleri, mevsimleri ve zaman akışını karıştırma.
    - Kelime bulmada güçlük çekme, konuşurken duraksama.
    - Eşyaları kaybetme veya yanlış yerlere koyma (örneğin gözlüğü buzdolabına koymak).

    BÖLÜM 2: ALZHEIMER HASTALARINA NASIL DAVRANILMALIDIR? (İLETİŞİM)
    Alzheimer hastasıyla iletişim kurarken sabır en önemli anahtardır. İşte dikkat edilmesi gerekenler:
    - Göz Teması: Konuşurken mutlaka göz seviyesine inin ve göz teması kurun.
    - Basit Cümleler: Kısa, net ve anlaşılır cümleler kurun. "Bugün hava güzel, parka gidelim mi?" gibi.
    - Tartışmadan Kaçınma: Hasta yanlış bir şey söylese bile (örneğin "Annem beni bekliyor" derse, annesi vefat etmiş olsa bile) onunla tartışmayın, gerçeği yüzüne vurmayın. Dikkatini dağıtın veya suyuna gidin.
    - Ses Tonu: Yumuşak, sakin ve güven veren bir ses tonu kullanın. Asla bağırmayın veya çocuk gibi azarlamayın.
    - Sözsüz İletişim: Beden dili, gülümseme ve hafif bir dokunuş, kelimelerden daha etkili olabilir.

    BÖLÜM 3: HASTA BAKIMI VE GÜNLÜK YAŞAM
    Hastanın günlük yaşamını kolaylaştırmak için rutinler oluşturulmalıdır:
    - Rutinler: Yemek, uyku ve banyo saatleri her gün aynı olmalıdır. Bu hastaya güven verir.
    - Kıyafet Seçimi: Giymesi kolay, düğmesiz veya cırt cırtlı kıyafetler tercih edilmelidir. Seçenekleri azaltmak (sadece 2 gömlek sunmak gibi) kafa karışıklığını önler.
    - Beslenme: İştah azalabilir veya yemek yemeyi unutabilirler. Küçük porsiyonlar halinde sık sık beslemek ve bol sıvı vermek önemlidir.

    BÖLÜM 4: EV GÜVENLİĞİ VE DÜZENLEMELER
    Hastalık ilerledikçe ev kazalarını önlemek için şunlar yapılmalıdır:
    - Zemin: Kaygan halılar kaldırılmalı veya sabitlenmelidir.
    - Işıklandırma: Evin içi, özellikle koridorlar ve banyo iyi aydınlatılmalıdır. Gece lambaları kullanılmalıdır.
    - Tehlikeli Eşyalar: İlaçlar, temizlik malzemeleri, bıçak ve çakmak gibi tehlikeli maddeler kilitli dolaplarda tutulmalıdır.
    - Kapılar: Hastanın evden habersiz çıkmasını önlemek için kapı kilitleri göz hizasından yukarıya veya aşağıya monte edilebilir.

    BÖLÜM 5: BAKIM VERENLER İÇİN TAVSİYELER
    Alzheimer hastasına bakmak yorucudur. Bakım veren kişi kendine zaman ayırmalı, tükenmişlik sendromuna karşı dikkatli olmalı ve gerektiğinde profesyonel destek almaktan çekinmemelidir.
    """

    dosya_adi = "alzheimer_veri.txt"
    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write(metin)

    print(f"✅ Gelişmiş veri dosyası '{dosya_adi}' oluşturuldu.")
    print("İÇERİK: Belirtiler, İletişim Yöntemleri, Ev Güvenliği ve Bakım Önerileri eklendi.")

if __name__ == "__main__":
    veri_dosyasi_olustur()
