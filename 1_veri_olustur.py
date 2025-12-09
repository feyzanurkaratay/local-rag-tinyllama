import os

def veri_dosyasi_olustur():
    """
    RAG sistemi için 'Ansiklopedi' seviyesinde kapsamlı Alzheimer verisi oluşturur.
    İçerik: Semptomlar, İletişim, Saldırganlık, Halüsinasyon, Banyo, Yemek, Araba Kullanımı vb.
    """
    metin = """
    KONU: ALZHEIMER HASTALIĞI BAKIM VE YÖNETİM REHBERİ (SSS)

    BÖLÜM 1: TEMEL BİLGİLER VE BELİRTİLER
    Soru: Alzheimer sadece hafıza kaybı mıdır?
    Cevap: Hayır. Hafıza kaybı en belirgin işarettir ama hastalık aynı zamanda karar verme, yer-yön bulma, konuşma ve kişilik değişikliklerine de yol açar.
    
    Soru: Hastalığın erken belirtileri nelerdir?
    Cevap: İsimleri unutma, aynı soruları tekrar sorma, tanıdık yerlerde kaybolma, para hesabını yapamama ve hobilerden uzaklaşma erken belirtilerdir.

    BÖLÜM 2: İLETİŞİM STRATEJİLERİ
    Soru: Beni tanımadı veya yanlış isim söyledi, ne yapmalıyım?
    Cevap: Onu düzeltmeyin veya "Ben senin kızınım!" diye zorlamayın. Bu onu utandırır ve öfkelendirir. Sadece sohbete devam edin ve ona güven verin.
    
    Soru: Sürekli "Eve gitmek istiyorum" diyor ama zaten evde. Ne demeliyim?
    Cevap: "Burası senin evin" diyerek tartışmayın. Bu cümle genellikle "Kendimi güvende hissetmiyorum" demektir. Dikkatini dağıtın, "Önce bir çay içelim, sonra bakarız" diyerek konuyu değiştirin.
    
    Soru: Söylediklerimi anlamıyor, nasıl konuşmalıyım?
    Cevap: Kısa cümleler kurun. "Ayakkabını giy, montunu al, dışarı çıkalım" demek yerine, sadece "Ayakkabını giy" deyin. O bitince diğerini söyleyin.

    BÖLÜM 3: ZORLU DAVRANIŞLAR (SALDIRGANLIK VE HALÜSİNASYON)
    Soru: Hastam çok sinirli ve saldırganlaştı, ne yapmalıyım?
    Cevap: Saldırganlık genellikle korku, ağrı veya anlaşılamama sonucudur. Sakin kalın, geri çekilin ve güvenli bir mesafe bırakın. Ortamdaki gürültüyü (TV, radyo) azaltın.
    
    Soru: Olmayan şeyleri görüyor (Halüsinasyon), ne yapmalıyım?
    Cevap: Eğer gördüğü şey onu korkutmuyorsa müdahale etmeyin. Korkuyorsa, "Orada kimse yok" diyerek inatlaşmayın. "Seni koruyacağım, burası güvenli" diyerek elini tutun veya odanın ışığını açarak gölgeleri yok edin.

    BÖLÜM 4: GÜNLÜK YAŞAM VE HİJYEN
    Soru: Banyo yapmak istemiyor, nasıl ikna edebilirim?
    Cevap: Banyo korkutucu olabilir. "Yıkanman lazım" demek yerine "Hadi ılık suyla rahatlayalım" deyin. Banyonun sıcak olduğundan emin olun. Utanıyorsa mahremiyetine saygı gösterin (havlu ile örtmek gibi).
    
    Soru: Yemek yemeyi reddediyor veya unutuyor.
    Cevap: Masa karmaşık olmasın. Tek çeşit yemek koyun. Çatal-bıçak kullanamıyorsa elle yenebilecek gıdalar (sandviç, dilim meyve) hazırlayın.
    
    Soru: Gece uyumuyor ve evde dolaşıyor (Sundowning / Akşam Sendromu).
    Cevap: Gündüz uyumasını engelleyin ve fiziksel aktivite yaptırın. Akşam saatlerinde kafein vermeyin. Evin içini loş değil, iyi aydınlatılmış tutun.

    BÖLÜM 5: GÜVENLİK VE ARABA KULLANIMI
    Soru: Araba kullanabilir mi?
    Cevap: Hayır. Alzheimer refleksleri ve karar vermeyi etkiler. Araba kullanmak hem hasta hem de başkaları için hayati tehlikedir. Anahtarları saklamanız veya arabayı görünmeyecek bir yere park etmeniz gerekebilir.
    
    Soru: Evden kaçıp kayboluyor.
    Cevap: Kapılara kilit veya açıldığında öten basit alarmlar takın. Cebine adres ve telefon numaranızın olduğu bir kart koyun. Bazı hastalar için GPS takip cihazları (saat veya kolye) hayat kurtarıcıdır.

    BÖLÜM 6: BAKIM VERENİN SAĞLIĞI
    Soru: Kendimi çok tükenmiş hissediyorum, bu normal mi?
    Cevap: Evet, çok normaldir. "Süper kahraman" olmaya çalışmayın. Diğer aile üyelerinden yardım isteyin veya profesyonel bakım desteği alın. Kendi sağlığınız bozulursa hastaya da bakamazsınız.

    BÖLÜM 7: HUKUKİ VE MALİ KONULAR
    Soru: Ne zaman vekaletname almalıyım?
    Cevap: Hastalık teşhisi konulur konulmaz, hasta henüz karar verme yetisini tamamen kaybetmemişken yasal süreçler (vasi tayini, noter işlemleri) halledilmelidir. İlerleyen evrelerde imza yetkisi kaybolur.
    """

    dosya_adi = "alzheimer_veri.txt"
    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write(metin)

    print(f"✅ DEVASA VERİ SETİ '{dosya_adi}' OLUŞTURULDU.")

if __name__ == "__main__":
    veri_dosyasi_olustur()
