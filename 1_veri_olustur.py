import os

def veri_dosyasi_olustur():
    """
    RAG sistemi için SADELEŞTİRİLMİŞ (Halk Dili) Alzheimer verisi.
    """
    metin = """
BAŞLIK: ALZHEIMER HASTALIĞI: HERKES İÇİN ANLAŞILIR REHBER

BÖLÜM 1: ALZHEIMER NEDİR? (BASİT ANLATIM)
Tanım: Alzheimer, beynin yavaş yavaş küçüldüğü ve hafızanın zayıfladığı bir hastalıktır. Unutkanlıkla başlar, zamanla günlük işleri yapmayı zorlaştırır.
Neden Olur?: Beyinde bazı proteinler (plaklar) birikir ve bu durum beyin hücrelerinin birbirleriyle konuşmasını engeller. Hücreler zamanla ölür.
Genetik mi?: Ailede varsa risk artabilir ama kesinlikle çocuğa geçer diye bir kural yoktur. Yaşlılık en büyük risktir.

BÖLÜM 2: İLAÇ VE TEDAVİ (BASİTÇE)
Tedavisi Var mı?: Hastalığı tamamen geçiren bir ilaç henüz yok. Ancak ilerlemesini yavaşlatan ve hastayı rahatlatan ilaçlar var.
İlaçlar Ne Yapar?:
1. Hafıza İlaçları (Donepezil vb.): Beyindeki haberleşme maddelerini artırarak hafızayı geçici olarak güçlendirir.
2. Sakinleştiriciler (Memantine vb.): Beynin aşırı yorulmasını engeller, hastayı daha sakin tutar.
Önemli Not: İlacın dozu ve saati çok önemlidir. Doktorun verdiği saatlerin dışına çıkılmamalıdır. İlaç rejimi yerine, ilacın düzenli kullanımı takip edilmelidir.

BÖLÜM 3: BELİRTİLER VE EVRELER
1. Başlangıç: İsimleri unutma, "Ocağı kapattım mı?" diye sürekli kontrol etme, para hesabını karıştırma.
2. Orta Dönem: Yolu kaybetme, mevsimsiz giyinme (yazın mont giymek gibi), banyo yapmayı istememe, "Sen kimsin?" diye sorma.
3. İleri Dönem: Konuşamama, yemeği yutamama, tuvaletini tutamama. Tam bakım gerekir.

BÖLÜM 4: HASTAYLA NASIL KONUŞMALIYIZ?
Altın Kural: Asla tartışmayın. "Yanlış söylüyorsun" demeyin.
Örnek: Hasta "Annem beni bekliyor" derse (annesi vefat etmiş olsa bile), "Annen öldü" demeyin. "Anneni özledin galiba, haydi bana ondan bahset" diyerek suyuna gidin.
Ses Tonu: Yumuşak ve sakin konuşun. Bebekle konuşur gibi değil, saygılı bir yetişkinle konuşur gibi davranın.

BÖLÜM 5: GÜNLÜK SORUNLAR VE ÇÖZÜMLER
Banyo Yapmak İstemiyor:
- Neden: Sudan korkuyor veya üşüyor olabilir.
- Çözüm: "Yıkanman lazım" demeyin. "Hadi ılık suyla rahatlayalım" diyerek ikna edin. Suyu kafasından aşağı birden dökmeyin.
Yemek Yemiyor:
- Çözüm: Belki çatal kullanmayı unuttu. Eline alıp yiyebileceği küçük sandviçler veya dilimlenmiş meyveler verin. Tabağı çok doldurmayın.

BÖLÜM 6: HUYSUZLUK VE SALDIRGANLIK
Neden Kızıyor?: Genellikle bir yeri ağrıyordur, tuvaleti gelmiştir veya korkmuştur ama bunu söyleyemediği için bağırır.
Ne Yapmalı?: Sakin olun. Üzerine gitmeyin. Ortamdaki gürültüyü (TV, radyo) kapatın. Elini tutarak güven verin.
Akşam Huysuzluğu (Güneş Batarken): Akşamları kafaları daha çok karışır. Evi akşamüzeri iyice aydınlatın, karanlık köşe kalmasın. Akşam çay-kahve vermeyin.

BÖLÜM 7: EVDE GÜVENLİK
Düşmemesi İçin: Kaygan halıları kaldırın. Gece tuvalete kalkarsa diye koridora gece lambası takın.
Kaybolmaması İçin: Kapıya zil takın, açılınca haberiniz olsun. Cebine mutlaka adresinizin yazdığı bir kağıt koyun.
Tehlikeli Şeyler: İlaçları, bıçakları ve temizlik malzemelerini kilitli dolaplara koyun.
Araba Kullanımı: Kesinlikle tehlikelidir. Refleksler zayıfladığı için araba kullanmalarına izin verilmemelidir.

BÖLÜM 8: BAKIM VEREN KİŞİ (SİZ)
Kendinizi Unutmayın: Bu çok zor bir görev. "Her şeyi ben yaparım" demeyin, yardım isteyin. Siz hasta olursanız ona kimse bakamaz.
    """

    dosya_adi = "alzheimer_veri.txt"
    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write(metin)

    print(f"✅ SADELEŞTİRİLMİŞ VERİ DOSYASI OLUŞTURULDU: {dosya_adi}")

if __name__ == "__main__":
    veri_dosyasi_olustur()
