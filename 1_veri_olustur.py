import os

def veri_dosyasi_olustur():
    """
    TinyLlama için 'Mega Ansiklopedi' verisi. 
    Araba kullanımı, banyo, yemek, saldırganlık vb. her şey burada yazıyor.
    """
    metin = """
BAŞLIK: ALZHEIMER HASTALIĞI HAKKINDA HER ŞEY (KAPSAMLI REHBER)

BÖLÜM 1: AKADEMİK BİLGİLER VE NEDENLERİ
Soru: Alzheimer nedir?
Cevap: Beyin hücrelerinin ölümüyle sonuçlanan, hafıza, düşünme ve davranış fonksiyonlarında azalmaya neden olan ilerleyici bir nörolojik hastalıktır.
Soru: Hastalığın biyolojik nedeni nedir?
Cevap: Beyinde 'Beta-amiloid' plaklarının birikmesi ve 'Tau' proteinlerinin düğümlenmesi sonucu sinir hücreleri arasındaki iletişim kopar.
Soru: Genetik midir?
Cevap: Erken başlangıçlı türünde genetik faktörler (APP, PSEN1) etkilidir. Geç başlangıçlı türünde APOE-e4 geni risk faktörüdür.

BÖLÜM 2: GÜVENLİK VE ARABA KULLANIMI (ÇOK ÖNEMLİ)
Soru: Alzheimer hastası araba kullanabilir mi?
Cevap: HAYIR. Alzheimer hastaları reflekslerini, yön duygularını ve karar verme yetilerini kaybederler. Araba kullanmaları hem kendileri hem de trafiktekiler için ölümcül risk taşır. Ehliyetlerine el konulmalı ve araba anahtarları saklanmalıdır.
Soru: Evden kaçıp kayboluyor, ne yapmalıyım?
Cevap: Kapılara açıldığında ses çıkaran alarmlar takın. Kapı kilidini göz hizasından yukarıya monte edin. Cebine mutlaka adresinizin ve telefonunuzun yazdığı bir kart koyun.

BÖLÜM 3: GÜNLÜK BAKIM VE HİJYEN
Soru: Annem/Babam banyo yapmak istemiyor, sudan korkuyor. Ne yapmalıyım?
Cevap: Onu zorlamayın ve "Yıkanman lazım" demeyin. "Hadi biraz rahatlayalım, ılık su iyi gelir" diyerek ikna edin. Banyonun sıcak olduğundan emin olun. Suyu kafasından aşağı birden dökmeyin, yavaşça alıştırın.
Soru: Yemek yemeyi reddediyor.
Cevap: Masa çok karmaşık olmasın. Çatal bıçak kullanamıyorsa elle yenebilen 'parmak gıdalar' (sandviç, meyve dilimi) hazırlayın. Beyaz tabakta renkli yemekler sunarak görmesini kolaylaştırın.

BÖLÜM 4: DAVRANIŞSAL SORUNLAR VE PSİKOLOJİ
Soru: Beni tanımıyor veya "Sen kimsin?" diyor.
Cevap: Bu hastalığın bir parçasıdır, kişisel algılamayın. "Ben senin oğlunum/kızınım" diye tartışmaya girmeyin. Sakince sohbete devam edin ve ona güven verin.
Soru: Akşamları çok huysuzlanıyor (Sundowning).
Cevap: Akşam sendromunu önlemek için evi iyi aydınlatın. Gölgeler hastayı korkutabilir. Akşam saatlerinde çay/kahve (kafein) vermeyin.
Soru: Olmayan şeyleri görüyor (Halüsinasyon).
Cevap: Eğer gördüğü şey onu korkutmuyorsa bozmayın. Korkuyorsa "Orada kimse yok" diye inatlaşmayın. "Ben yanındayım, seni korurum" diyerek elini tutun.

BÖLÜM 5: İLAÇ VE TEDAVİ
Soru: Hastalığın tedavisi var mı?
Cevap: Kesin bir tedavisi yoktur, hastalık tamamen iyileşmez. Ancak Donepezil, Rivastigmine ve Memantine gibi ilaçlar belirtileri geçici olarak yavaşlatabilir.

BÖLÜM 6: BAKIM VERENİN YÜKÜ
Soru: Kendimi çok yorgun ve tükenmiş hissediyorum.
Cevap: Bu çok normaldir. Her şeyi tek başınıza yapmaya çalışmayın. Aileden destek isteyin veya profesyonel yardım alın. Siz iyi olmazsanız hastanıza da bakamazsınız.
    """

    dosya_adi = "alzheimer_veri.txt"
    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write(metin)

    print(f"✅ MEGA VERİ SETİ OLUŞTURULDU: {dosya_adi}")

if __name__ == "__main__":
    veri_dosyasi_olustur()
