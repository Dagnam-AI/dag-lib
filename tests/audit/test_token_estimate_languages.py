"""The estimate against the student's own count on realistic texts, one per language and script.

Each ``qwen`` is the length of the text in Qwen2.5's own tokens, measured once and kept as a
literal. The bands are wide enough for a fitted estimate and tight enough that a rate or a
word price that is wrong for a language fails here.
"""

from __future__ import annotations

import json

import pytest

from dagnam.audit.token_estimate import estimate

INVOICE = json.dumps(
    {
        "invoice": "INV-20260927-0042",
        "total": 1234.56,
        "currency": "EUR",
        "lines": [{"sku": "A10023", "qty": 12, "description": "USB-C charging cable"}],
    }
)
CODE = """def cap_train(train, strata, keys, *, limit):
    if len(train) <= limit:
        return list(train)
    by_stratum = {}
    for i in train:
        by_stratum.setdefault(strata[i], []).append(i)
    return sorted(by_stratum)
"""


@pytest.mark.parametrize(
    ("text", "qwen"),
    [
        # Each text's length in Qwen2.5's own tokens (the qlora-sft-chat base), measured.
        (
            "Refunds are issued to the original payment method within five business days of"
            " approval. A customer may return any unopened item within thirty days of delivery"
            " for a full refund.",
            34,
        ),
        (INVOICE, 69),
        (CODE, 58),
    ],
    ids=["english", "json", "code"],
)
def test_estimate_is_close_to_the_student_s_own_count(text: str, qwen: int) -> None:
    # English prose, JSON and code, each measured against Qwen2.5's own count. Code is counted
    # high on purpose (1.24 here, 1.12 at the median over a corpus): an identifier with
    # underscores or camelCase seams costs more than the student spends on it.
    assert 0.9 * qwen <= estimate(text) <= 1.3 * qwen


@pytest.mark.parametrize(
    ("text", "qwen"),
    [
        (
            "您好\uff0c我上周下的订单号为48213的包裹今天才到\uff0c而且箱子已经破损\uff0c还少了两件我已经付款的商品。因为我月底之前必须用到这些东西\uff0c所以想问一下能否全额退款或者尽快换货\uff1f另外\uff0c请告诉我如何退回损坏的商品。我全天都可以接电话。非常感谢您的帮助。",
            71,
        ),
        (
            "お世話になっております。先週注文した注文番号48213の荷物が遅れて届き、箱が破損していました。また、支払い済みの商品が二つ入っていませんでした。月末までに必要なため、全額返金または早急な交換が可能かどうか教えていただけますか。破損した商品の返品方法もご案内いただけると助かります。終日電話に出られます。よろしくお願いいたします。",
            108,
        ),
        (
            "안녕하세요, 지난주에 주문한 48213번 주문 건으로 연락드립니다. 택배가 늦게 도착했고 상자가 파손되어 있었으며 결제한 상품 두 개가 빠져 있었습니다. 이번 달 말까지 꼭 필요한 물건이라서 전액 환불이나 빠른 교환이 가능한지 알고 싶습니다. 파손된 상품을 반품하는 방법도 안내해 주실 수 있을까요? 하루 종일 전화 연락이 가능합니다. 도움에 미리 감사드립니다.",
            132,
        ),
        (
            "नमस्ते, मैं अपने ऑर्डर नंबर 48213 के बारे में लिख रहा हूँ जो देर से पहुँचा और डिब्बा क्षतिग्रस्त था। इसके अलावा, दो उत्पाद गायब थे जिनका मैंने पहले ही भुगतान कर दिया था। मैं जानना चाहता हूँ कि क्या पूरा रिफंड या जल्दी बदलाव संभव है, क्योंकि मुझे इन चीज़ों की महीने के अंत से पहले ज़रूरत है। क्या आप यह भी बता सकते हैं कि खराब उत्पाद कैसे वापस भेजें? धन्यवाद।",
            342,
        ),
        (
            "สวัสดี ฉันติดต่อเกี่ยวกับคำสั่งซื้อที่มาถึงล่าช้าและกล่องได้รับความเสียหาย สินค้าที่ชำระเงินแล้วขาดไปสองรายการ ฉันต้องการขอคืนเงินเต็มจำนวนหรือเปลี่ยนสินค้าโดยเร็ว กรุณาแจ้งขั้นตอนการส่งคืนสินค้าที่เสียหาย และระยะเวลาที่จะได้รับเงินคืน ขอบคุณสำหรับความช่วยเหลือ",
            139,
        ),
        (
            "নমস্কার, আমি গত সপ্তাহে দেওয়া 48213 নম্বর অর্ডারটি নিয়ে লিখছি। পার্সেলটি দেরিতে পৌঁছেছে এবং বাক্সটি ক্ষতিগ্রস্ত ছিল; তাছাড়া যে দুটি জিনিসের দাম আমি আগেই দিয়েছি সেগুলো ছিল না। মাসের শেষের আগে এই জিনিসগুলো আমার দরকার, তাই জানতে চাই পুরো টাকা ফেরত বা দ্রুত বদল সম্ভব কি না। ক্ষতিগ্রস্ত পণ্য কীভাবে ফেরত পাঠাব তাও কি জানাতে পারবেন? আমি সারাদিন ফোনে পাওয়া যাব। আপনার সাহায্যের জন্য আগাম ধন্যবাদ।",
            413,
        ),
        (
            "வணக்கம், கடந்த வாரம் நான் செய்த 48213 என்ற எண் கொண்ட ஆர்டர் பற்றி எழுதுகிறேன். பார்சல் தாமதமாக வந்தது, பெட்டியும் சேதமடைந்திருந்தது; மேலும் நான் ஏற்கனவே பணம் செலுத்திய இரண்டு பொருட்கள் இல்லை. இந்த பொருட்கள் மாத இறுதிக்குள் எனக்குத் தேவை, எனவே முழு பணத்தையும் திரும்பப் பெற முடியுமா அல்லது விரைவாக மாற்றித் தர முடியுமா என்று அறிய விரும்புகிறேன். சேதமடைந்த பொருட்களை எப்படித் திருப்பி அனுப்புவது என்பதையும் விளக்க முடியுமா? நாள் முழுவதும் தொலைபேசியில் தொடர்பு கொள்ளலாம். உங்கள் உதவிக்கு முன்கூட்டியே நன்றி.",
            556,
        ),
    ],
    ids=["chinese", "japanese", "korean", "hindi", "thai", "bengali", "tamil"],
)
def test_estimate_calibrates_script_rates(text: str, qwen: int) -> None:
    # Support requests in each script, counted with real Qwen2.5. Han is priced by the
    # vocabulary's own tokens, so Simplified (1.04 here) and Traditional need no rates of
    # their own; Korean is the highest (1.14) and Hindi the lowest (0.98).
    assert 0.95 * qwen <= estimate(text) <= 1.25 * qwen


@pytest.mark.parametrize(
    ("text", "qwen"),
    [
        (
            "Buongiorno, vi scrivo riguardo all'ordine numero 48213 che ho effettuato la settimana scorsa. Il pacco è arrivato in ritardo e la scatola era danneggiata; inoltre mancavano due articoli che avevo già pagato. Ho bisogno di questi prodotti entro la fine del mese, quindi vorrei sapere se è possibile ottenere un rimborso completo oppure una sostituzione rapida. Potreste anche spiegarmi come restituire la merce danneggiata? Sono disponibile al telefono tutto il giorno. Grazie in anticipo per il vostro aiuto.",
            156,
        ),
        (
            "Goedemiddag, ik schrijf u over bestelling nummer 48213 die ik vorige week heb geplaatst. Het pakket kwam te laat aan en de doos was beschadigd; bovendien ontbraken er twee artikelen waarvoor ik al had betaald. Ik heb deze spullen voor het einde van de maand nodig, dus ik wil graag weten of een volledige terugbetaling of een snelle vervanging mogelijk is. Kunt u mij ook uitleggen hoe ik de beschadigde goederen kan terugsturen? Ik ben de hele dag telefonisch bereikbaar. Alvast bedankt voor uw hulp.",
            157,
        ),
        (
            "Guten Tag, ich schreibe Ihnen wegen der Bestellung Nummer 48213, die ich letzte Woche aufgegeben habe. Das Paket kam verspätet an und der Karton war beschädigt; außerdem fehlten zwei Artikel, die ich bereits bezahlt hatte. Ich brauche diese Sachen vor Ende des Monats, deshalb möchte ich wissen, ob eine vollständige Rückerstattung oder ein schneller Ersatz möglich ist. Könnten Sie mir auch erklären, wie ich die beschädigte Ware zurückschicken kann? Ich bin den ganzen Tag telefonisch erreichbar. Vielen Dank im Voraus für Ihre Hilfe.",
            149,
        ),
        (
            "Bonjour, je vous écris au sujet de la commande numéro 48213 que j'ai passée la semaine dernière. Le colis est arrivé en retard et le carton était abîmé ; de plus, il manquait deux articles que j'avais déjà payés. J'ai besoin de ces produits avant la fin du mois, je voudrais donc savoir s'il est possible d'obtenir un remboursement complet ou un remplacement rapide. Pourriez-vous aussi m'expliquer comment renvoyer la marchandise endommagée ? Je suis joignable par téléphone toute la journée. Merci d'avance pour votre aide.",
            142,
        ),
        (
            "Buenos días, les escribo por el pedido número 48213 que realicé la semana pasada. El paquete llegó con retraso y la caja estaba dañada; además, faltaban dos artículos que ya había pagado. Necesito estos productos antes de fin de mes, así que me gustaría saber si es posible obtener un reembolso completo o un cambio rápido. ¿Podrían explicarme también cómo devolver la mercancía dañada? Estoy disponible por teléfono durante todo el día. Muchas gracias de antemano por su ayuda.",
            129,
        ),
        (
            "Bom dia, escrevo a respeito do pedido número 48213 que fiz na semana passada. A encomenda chegou atrasada e a caixa estava danificada; além disso, faltavam dois artigos que eu já tinha pago. Preciso destes produtos antes do fim do mês, por isso gostaria de saber se é possível obter um reembolso total ou uma troca rápida. Poderiam também explicar-me como devolver a mercadoria danificada? Estou disponível por telefone durante todo o dia. Desde já agradeço a vossa ajuda.",
            135,
        ),
        (
            "Hej, jag skriver angående beställning nummer 48213 som jag lade förra veckan. Paketet kom för sent och kartongen var skadad; dessutom saknades två varor som jag redan hade betalat för. Jag behöver de här sakerna före månadens slut, så jag skulle vilja veta om det är möjligt att få full återbetalning eller en snabb ersättning. Kan ni också förklara hur jag skickar tillbaka de skadade varorna? Jag går att nå på telefon hela dagen. Tack på förhand för hjälpen.",
            150,
        ),
        (
            "Hej, jeg skriver angående ordre nummer 48213, som jeg afgav i sidste uge. Pakken kom for sent, og kassen var beskadiget; desuden manglede der to varer, som jeg allerede havde betalt for. Jeg har brug for tingene inden månedens udgang, så jeg vil gerne vide, om det er muligt at få pengene tilbage eller en hurtig ombytning. Kan I også forklare, hvordan jeg sender de beskadigede varer retur? Jeg kan træffes på telefon hele dagen. På forhånd tak for hjælpen.",
            151,
        ),
        (
            "Hei, jeg skriver angående bestilling nummer 48213 som jeg la inn forrige uke. Pakken kom for sent og esken var skadet; i tillegg manglet det to varer som jeg allerede hadde betalt for. Jeg trenger disse tingene før slutten av måneden, så jeg vil gjerne vite om det er mulig å få full refusjon eller en rask erstatning. Kan dere også forklare hvordan jeg sender de skadde varene tilbake? Jeg er tilgjengelig på telefon hele dagen. På forhånd takk for hjelpen.",
            144,
        ),
        (
            "Hei, kirjoitan tilauksesta numero 48213, jonka tein viime viikolla. Paketti saapui myöhässä ja laatikko oli vaurioitunut; lisäksi siitä puuttui kaksi tuotetta, jotka olin jo maksanut. Tarvitsen nämä tavarat ennen kuun loppua, joten haluaisin tietää, onko mahdollista saada täysi hyvitys tai nopea vaihto. Voisitteko myös kertoa, miten palautan vaurioituneet tuotteet? Olen tavoitettavissa puhelimitse koko päivän. Kiitos jo etukäteen avustanne.",
            175,
        ),
        (
            "Dzień dobry, piszę w sprawie zamówienia numer 48213, które złożyłem w zeszłym tygodniu. Paczka dotarła z opóźnieniem, a karton był uszkodzony; ponadto brakowało dwóch produktów, za które już zapłaciłem. Potrzebuję tych rzeczy przed końcem miesiąca, dlatego chciałbym wiedzieć, czy możliwy jest pełny zwrot pieniędzy lub szybka wymiana. Czy mogliby Państwo również wyjaśnić, jak odesłać uszkodzony towar? Jestem dostępny pod telefonem przez cały dzień. Z góry dziękuję za pomoc.",
            161,
        ),
        (
            "Dobrý den, píšu ohledně objednávky číslo 48213, kterou jsem zadal minulý týden. Balík dorazil se zpožděním a krabice byla poškozená; navíc chyběly dvě položky, které jsem už zaplatil. Tyto věci potřebuji do konce měsíce, proto bych rád věděl, zda je možné získat plnou náhradu nebo rychlou výměnu. Mohli byste mi také vysvětlit, jak mám poškozené zboží vrátit? Jsem k zastižení na telefonu po celý den. Předem děkuji za pomoc.",
            207,
        ),
        (
            "Merhaba, geçen hafta verdiğim 48213 numaral\u0131 sipariş hakk\u0131nda yaz\u0131yorum. Paket geç geldi ve kutu hasarl\u0131yd\u0131; ayr\u0131ca paras\u0131n\u0131 ödediğim iki ürün eksikti. Bu ürünlere ay sonundan önce ihtiyac\u0131m var, bu yüzden tam para iadesi veya h\u0131zl\u0131 bir değişim mümkün mü öğrenmek istiyorum. Hasarl\u0131 ürünleri nas\u0131l geri göndereceğimi de aç\u0131klayabilir misiniz? Gün boyunca telefonla ulaşabilirsiniz. Yard\u0131m\u0131n\u0131z için şimdiden teşekkür ederim.",
            134,
        ),
        (
            "Selamat siang, saya menulis mengenai pesanan nomor 48213 yang saya buat minggu lalu. Paketnya datang terlambat dan kotaknya rusak; selain itu, ada dua barang yang sudah saya bayar tetapi tidak ada di dalamnya. Saya membutuhkan barang-barang ini sebelum akhir bulan, jadi saya ingin tahu apakah pengembalian dana penuh atau penggantian cepat bisa dilakukan. Bisakah Anda juga menjelaskan cara mengembalikan barang yang rusak? Saya dapat dihubungi lewat telepon sepanjang hari. Terima kasih sebelumnya atas bantuan Anda.",
            155,
        ),
        (
            "Xin chào, tôi viết thư này về đơn hàng số 48213 mà tôi đã đặt tuần trước. Gói hàng đến trễ và hộp bị hư hỏng; ngoài ra còn thiếu hai món mà tôi đã thanh toán. Tôi cần những món này trước cuối tháng, vì vậy tôi muốn biết liệu có thể được hoàn tiền đầy đủ hoặc đổi hàng nhanh hay không. Anh chị có thể hướng dẫn tôi cách gửi trả hàng bị hỏng không? Tôi có thể nghe điện thoại cả ngày. Xin cảm ơn trước vì sự giúp đỡ.",
            116,
        ),
        (
            "Bună ziua, vă scriu în legătură cu comanda numărul 48213 pe care am plasat-o săptămâna trecută. Coletul a ajuns cu întârziere, iar cutia era deteriorată; în plus, lipseau două articole pe care le plătisem deja. Am nevoie de aceste produse înainte de sfârșitul lunii, așa că aș dori să știu dacă este posibilă o rambursare completă sau o înlocuire rapidă. Ați putea să-mi explicați și cum pot returna marfa deteriorată? Sunt disponibil la telefon toată ziua. Vă mulțumesc anticipat pentru ajutor.",
            181,
        ),
        (
            "Jó napot kívánok, a múlt héten leadott 48213-as számú rendelésem miatt írok. A csomag késve érkezett meg, és a doboz sérült volt; ráadásul hiányzott két termék, amelyeket már kifizettem. Ezekre a dolgokra a hónap vége előtt szükségem van, ezért szeretném tudni, hogy lehetséges-e a teljes visszatérítés vagy a gyors csere. El tudnák magyarázni azt is, hogyan küldjem vissza a sérült árut? Egész nap elérhető vagyok telefonon. Előre is köszönöm a segítségüket.",
            193,
        ),
    ],
    ids=[
        "italian",
        "dutch",
        "german",
        "french",
        "spanish",
        "portuguese",
        "swedish",
        "danish",
        "norwegian",
        "finnish",
        "polish",
        "czech",
        "turkish",
        "indonesian",
        "vietnamese",
        "romanian",
        "hungarian",
    ],
)
def test_estimate_holds_for_latin_script_languages(text: str, qwen: int) -> None:
    # Every ASCII word of up to eight letters was one token, which is true of English only:
    # Italian and Dutch came out at 0.7x, Finnish at 0.6x, and a workload whose rows were
    # all over the student's budget passed as trainable. Each text's length is Qwen2.5's own.
    assert 0.95 * qwen <= estimate(text) <= 1.25 * qwen


def test_estimate_keeps_english_within_its_tighter_band() -> None:
    text = "Hello, I am writing about order number 48213, which I placed last week. The parcel arrived late and the box was damaged, and two of the items I had already paid for were missing. I need these things before the end of the month, so I would like to know whether a full refund or a quick replacement is possible. Could you also tell me how to send the damaged goods back? I can take a call at any time of day. Thank you in advance for your help."
    assert 0.95 * 104 <= estimate(text) <= 1.15 * 104  # Qwen2.5: 104
