# Adat-előkészítési állapot 2026. október 31.

Az ellenőrzés a helyi adatokon futott; GitHubra nem történt feltöltés. Az eredeti képek, címkék és a korábbi tanítási felosztás változatlanok.

## Eredmények

- 705 RGB-kép, 141 fájlnév szerinti eredeti csoport, 1395 címkézett kocka.
- Tanítóhalmaz: 565 kép / 113 csoport; validáció: 70 / 14; teszt: 70 / 14.
- A részhalmazok között nincs fájlnév szerinti csoportátfedés vagy bájtszinten azonos kép.
- Az eredeti képek és címkék egyszer szerepelnek a felosztásban, tartalmuk az eredetivel egyezik.
- 690 képen két, 15 képen egy címkézett kocka szerepel. Az egykockás képek mind a tanítóhalmazban vannak.
- Öt eredeti címkefájlban képhatáron túlnyúló keret van, mind ugyanannak a felvételnek a változatai.
- A jelentés 10 hibabejegyzése és 30 figyelmeztetése az eredeti és a tanítóhalmazban lévő példányokat külön számolja; nem 40 különböző problémás képről van szó.

## Célzott vizuális ellenőrzés

A négy megjelölt eredeti képet a címkézett előnézeten ellenőriztük. Ez nem a teljes adathalmaz szemrevételezéses jóváhagyása.

| Eredeti csoport | Megfigyelés | Javaslat |
|---|---|---|
| 20201224_205955 | Egyetlen kocka, felső értéke 4; a címke ezt jelöli. | Nem hiányzó második címke. Tanításhoz megtartható, de a szigorúan kétkockás értékeléstől elkülönítendő. |
| 20201224_205957 | Egyetlen kocka, felső értéke 2; a címke ezt jelöli. | Ugyanaz a kezelés, mint fent. |
| 20201224_210000 | Egyetlen kocka, felső értéke 3; a címke ezt jelöli. | Ugyanaz a kezelés, mint fent. |
| 20201224_210506 | Egy teljes és egy részben képen kívüli kocka; az utóbbi kerete túlnyúlik a képen. | A csapattal eldöntendő: a teljes csoport kizárása a jól látható kockákra szűkített feladatból, vagy külön másolatban dokumentált keretjavítás és részleges láthatóság kezelése. |

A címkéket nem javítottuk találomra és nem töröltünk képeket. A szigorú újrafelosztás a hibás keretnél a másolás előtt leállt, új adathalmaz nem készült. A régi tanítások eredményeit az eddig használt felosztással együtt kell kezelni.

## Elkészült helyi eszközök és ellenőrzések

- `audit_dataset.py`: formátum-, képolvashatóság-, átfedés- és teljességellenőrzés; osztályonkénti összesítés; külön felülvizsgálati lista és futási időpont.
- `preview_labels.py`: címkézett, helyben megnyitható galéria; célzott szűrés az ellenőrzési jelentés alapján; olvashatatlan képek jelzése.
- `create_split.py`: csoportos 80/10/10 felosztás, felülírás elleni védelem, mentett besorolási jegyzék. A jelenlegi hibás címkékkel szándékosan leáll.
- Három célzott egységteszt sikeresen lefutott: címkeérvényesség; csoportos felosztás és felülírás tiltása; hibás keret, más néven átmásolt kép és eredetitől eltérő állomány felismerése.

## Megnyitás és újrafuttatás

A teljes eredeti-kép galéria: `label_preview/index.html` (141 kép).
A célzott galéria: `review_preview/index.html` (4 kép).
A részletes gépi jelentés: `data_audit/audit.json`; az összesítés: `data_audit/summary.md`.

A projekt mappájából:

```powershell
.venv/Scripts/python.exe audit_dataset.py
.venv/Scripts/python.exe preview_labels.py --review-json reports/data_audit/audit.json --originals-only --output reports/review_preview
```

Az ellenőrző nem nulla kilépési kódja a megtalált adathibákat jelzi. A következő feladat a többi eredeti és az átalakított képek vizuális átnézése, a problémás csoport kezelésének egyeztetése és az esetleges javítás külön, követhető adatváltozatban történő elvégzése.
