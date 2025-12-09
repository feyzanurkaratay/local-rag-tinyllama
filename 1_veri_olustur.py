import os

def veri_dosyasi_olustur():
    """
    RAG sistemi için AKADEMİK + PRATİK hibrit Alzheimer verisi oluşturur.
    """
    metin = """
BAŞLIK: ALZHEIMER HASTALIĞI: AKADEMİK, KLİNİK VE PRATİK BAKIM ANSİKLOPEDİSİ

BÖLÜM 1: AKADEMİK BİLGİLER VE PATOFİZYOLOJİ
Tanım: Alzheimer, bilişsel fonksiyonlarda (hafıza, dil, görsel-uzamsal beceriler) ilerleyici yıkıma neden olan nörodejeneratif bir hastalıktır.
Nörobiyolojik Mekanizmalar:
1. Amiloid Plakları (Ekstraselüler): Beta-amiloid proteininin yanlış katlanarak nöronlar arasında birikmesi, sinaptik iletişimi bozar ve hücre ölümünü tetikler.
2. Nörofibriler Yumaklar (Intraselüler): 'Tau' proteininin hiperfosforilasyonu sonucu hücre içi iskelet yapısı (mikrotübüller) çöker. Bu durum, nöron içi madde taşınmasını engeller.
3. Kolinerjik Hipotez: Öğrenme ve hafızada kritik rol oynayan 'Asetilkolin' nörotransmitterini sentezleyen nöronların kaybı, bilişsel gerilemenin temel nedenlerinden biridir.
Genetik Risk Faktörleri:
- Erken Başlangıçlı (Ailesel): APP, PSEN1 ve PSEN2 gen mutasyonları.
- Geç Başlangıçlı (Sporadik): APOE-e4 aleli taşıyıcılığı riski artırır ancak kesin neden değildir.

BÖLÜM 2: FARMAKOLOJİK TEDAVİ (İLAÇLAR)
Mevcut tedaviler hastalığı durdurmaz ancak semptomları yönetir:
1. Asetilkolinesteraz İnhibitörleri: (Donepezil, Rivastigmine, Galantamine). Asetilkolin yıkımını engelleyerek sinaptik aralıkta kalma süresini uzatır. Hafif ve orta evrede etkilidir.
2. NMDA Reseptör Antagonistleri: (Memantine). Glutamat eksitotoksisitesini (aşırı uyarılmaya bağlı hücre ölümü) önler. Orta ve ileri evrede kullanılır.

BÖLÜM 3: HASTALIK EVRELERİ VE BELİRTİLER
1. Erken Evre (Hafif):
   - Belirtiler: Yakın süreli hafıza kaybı, kelime bulma zorluğu (anomi), karmaşık görevleri (fatura ödeme) yönetememe, inisiyatif kaybı.
   - Bakım İhtiyacı: Minimaldir. Hasta bağımsız yaşayabilir ancak hatırlatıcılara ihtiyaç duyar.
2. Orta Evre:
   - Belirtiler: Tanıdık yüzleri unutma, zaman ve mekan oryantasyonunun bozulması, mevsimsiz giyinme, şüphecilik (paranoya), amaçsız gezinme (wandering).
   - Bakım İhtiyacı: Giyinme, banyo ve tuvalet gibi günlük yaşam aktivitelerinde (GYA) destek gerekir.
3. İleri Evre (Şiddetli):
   - Belirtiler: İletişim kaybı, yürüme ve oturma yetisinin kaybı, yutma güçlüğü (disfaji), idrar ve gaita inkontinansı (kaçırma).
   - Bakım İhtiyacı: 7/24 tam bakım gerektirir.

BÖLÜM 4: HASTA İLE İLETİŞİM TEKNİKLERİ
Validasyon (Onaylama) Yöntemi:
- Kural: Hastanın gerçekliğini reddetmeyin.
- Örnek: "Annem beni bekliyor" diyen (annesi ölmüş) bir hastaya "Annen öldü" demek onu her seferinde yeniden travmatize eder. Bunun yerine "Anneni çok özledin, bana en sevdiği yemeği anlatır mısın?" diyerek duygusunu onaylayın ve dikkati dağıtın.
Sözlü İletişim Stratejileri:
- Kısa ve Net Olun: Tek seferde tek komut verin. "Banyoya git, elini yıka ve gel" yerine, sırayla "Banyoya gidelim" deyin. O bitince diğerini söyleyin.
- Seçenekleri Azaltın: "Ne giymek istersin?" yerine "Mavi gömleği mi, kırmızıyı mı istersin?" diye sorun.
- 'Hayır' Demekten Kaçının: Yasaklamak yerine yönlendirin.

BÖLÜM 5: GÜNLÜK BAKIM VE HİJYEN SORUNLARI
Banyo Yapma Reddi:
- Neden: Sudan korkma, üşüme veya soyunmaktan utanma olabilir.
- Çözüm: Banyonun sıcak olduğundan emin olun. "Yıkanma zamanı" yerine "Rahatlama zamanı" diyerek stresi azaltın. Gerekirse vücudunu parça parça silerek temizleyin.
Yemek Yeme Sorunları:
- Neden: Çatal-bıçak kullanmayı unutma, iştahsızlık veya yemeği ağızda tutma.
- Çözüm: Parmak gıdalar (küçük sandviçler, dilim meyveler) sunun. Beyaz tabakta beyaz pilav gibi renk karmaşasından kaçının (kontrast renkler kullanın).

BÖLÜM 6: DAVRANIŞSAL SORUNLAR VE YÖNETİMİ
Sundowning (Akşam Sendromu):
- Tanım: Akşamüzeri artan huzursuzluk, kafa karışıklığı ve ajitasyon.
- Yönetim: Evi akşam saatlerinde çok iyi aydınlatın (gölgeleri yok edin). Gündüz uyumasını sınırlayın. Akşam kafein vermeyin.
Saldırganlık ve Ajitasyon:
- Neden: Genellikle karşılanmamış bir ihtiyaç (ağrı, açlık, tuvalet) veya korku tetikler.
- Yönetim: Sakin kalın, güvenli mesafe bırakın. Ortamdaki gürültüyü (TV) kapatın. Asla tartışmayın veya fiziksel güç kullanmayın.
Halüsinasyonlar:
- Yönetim: Gördüğü şey onu korkutmuyorsa müdahale etmeyin. Korkuyorsa "Orada kimse yok" diye inatlaşmayın, "Ben yanındayım, güvendesin" diyerek fiziksel temasla sakinleştirin.

BÖLÜM 7: EV GÜVENLİĞİ VE DÜZENLEMELER
Düşme Riski: Kaygan halıları kaldırın veya sabitleyin. Koridorlara ve banyoya gece lambaları takın.
Gezinme ve Kaybolma: Kapılara, açıldığında ses çıkaran basit ziller takın. Hastanın cebine mutlaka kimlik ve iletişim bilgilerinin olduğu bir kart koyun.
Mutfak Güvenliği: Ocak düğmelerine çocuk kilidi takın. Bıçak, makas, ilaç ve temizlik malzemelerini kilitli dolaplarda saklayın.

BÖLÜM 8: BAKIM VEREN SAĞLIĞI VE HUKUKİ SÜREÇLER
Tükenmişlik Sendromu: Bakım veren kişilerde depresyon riski yüksektir. "Süper kahraman" olmaya çalışmayın, profesyonel yardım veya aile desteği isteyin.
Hukuki Konular: Hastalık teşhisi konulur konulmaz, hasta henüz karar verme yetisini kaybetmemişken vasi tayini ve noter işlemleri (vekaletname) tamamlanmalıdır.
    """

    dosya_adi = "alzheimer_veri.txt"
    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write(metin)

    print(f"✅ Gelişmiş Ansiklopedi verisi '{dosya_adi}' dosyasına yazıldı.")

if __name__ == "__main__":
    veri_dosyasi_olustur()
