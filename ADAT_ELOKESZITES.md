# Adat-előkészítés és ellenőrzés

Ez a rész az első adatbázis ellenőrzését, címkéinek megjelenítését és reprodukálható felosztását végzi. Nem tanít modellt. Az eredeti adatokat és a már meglévő tanítási felosztást nem módosítja.

## Futtatás

A parancsokat a projekt mappájában futtasd. Az ellenőrzéshez, a galériához és az adatellenőrző tesztjeihez Python és Pillow szükséges. Új gépen először hozd létre a környezetet: `python -m venv .venv`, majd telepítsd a képolvasót: `.venv/Scripts/python.exe -m pip install "Pillow>=10.1.0"`. A felosztó csak a Python standard könyvtárát használja. A teljes projekt további függőségei a requirements.txt fájlban vannak.

```powershell
.venv/Scripts/python.exe audit_dataset.py
.venv/Scripts/python.exe preview_labels.py --originals-only
.venv/Scripts/python.exe -m unittest test_dataset_tools -v
```

A `reports/data_audit/summary.md` az összesítést, az `audit.json` a részletes hibákat és figyelmeztetéseket tartalmazza. Az ellenőrző hibánál nem nulla kilépési kódot ad. A `reports/label_preview/index.html` fájlt böngészőben nyisd meg. A galéria a címkékből rajzol, nem modelljóslatokat mutat. A `--originals-only` nélkül az átalakított képek is megjelennek; a `--limit 20` korlátozza a mennyiséget. Más mappák a `--images`, `--labels`, `--output` kapcsolókkal választhatók ki.

## Mit ellenőrzünk

- A képek olvashatóságát, a hiányzó és pár nélküli címkéket, az azonos nevű képek többértelműségét.
- Címkénként öt mezőt, egész 0–5 osztályazonosítót, véges és normalizált koordinátákat, pozitív méreteket és képen belüli kereteket. Az üres címke ennél a kétkockás adatbázisnál hiba.
- A két kockától eltérő objektumszámot figyelmeztetésként, az osztályok gyakoriságát minden halmazban.
- Az eredeti felvételek fájlnév szerinti csoportjainak és a képfájlok SHA-256 lenyomatainak átfedését a halmazok között.
- A kiosztás teljességét, egyértelműségét és a felosztott képek/címkék egyezését az eredetiekkel.

A helyes formátum nem bizonyítja, hogy a címke a megfelelő kockát vagy pontértéket jelöli. Ehhez nézd át a galériát: a keret illeszkedjen a kockára, a felirat a felső lap értéke legyen. Az átalakított példáknál a keret kövesse a geometriai változást. A feliratban a pontérték és zárójelben a 0-alapú osztályazonosító látható.

A fájlnév szerinti csoportosítás és az azonos bájtok keresése nem ismeri fel az összes közel azonos sorozatfelvételt. Ha ilyet találunk, a kapcsolódó eredeti csoportokat is együtt kell felosztani. Egy ilyen új felosztás új tanítást és kiértékelést igényel; a meglévő eredmények nem vihetők át automatikusan.

## Új felosztás biztonságosan

**Jelenlegi állapot:** az eredeti adatban képhatáron túlnyúló címkék vannak. A szigorú felosztó ezek miatt még a másolás előtt leáll; ez várt adatminőségi jelzés. Új adathalmaz csak a javítás vagy a dokumentált kizárás eldöntése után készíthető. A meglévő `data/dice_yolo` felosztás továbbra is rendelkezésre áll.

```powershell
.venv/Scripts/python.exe create_split.py
.venv/Scripts/python.exe audit_dataset.py --split data/dice_yolo_prepared --output reports/prepared_audit
```

Az alapértelmezett kimenet az új `data/dice_yolo_prepared` mappa. Már létező kimenetnél a program leáll; semmit nem töröl és nem kever korábbi felosztással. Másik új mappa `--output`, más véletlenmag `--seed` segítségével választható. Az eredeti forrás a `--raw` kapcsolóval állítható be.

Az arány eredeti képcsoportok szerint 80/10/10, a véletlenmag 42. A csoportazonosító a fájlnév kiterjesztés és első `__` előtti része. Az RGB `images` mappát használjuk; a párhuzamos `grey` változatokat nem tesszük hozzá még egyszer. A `split_manifest.json` rögzíti a beállításokat és minden fájl besorolását. Az adatok és a beállítások változatlansága esetén a kiosztás megismételhető. A `data.yaml` JSON-szintaxisú, amely érvényes YAML is.

A már meglévő `data/dice_yolo` és a modell tanítási beállításai változatlanok maradnak. A tanításért felelős csapattaggal kell egyeztetni, ha új felosztást szeretnénk használni.

## Forrás és hozzájárulás

Adatforrás: [Nouman Ahsan – Dice Detection Upper View](https://www.kaggle.com/datasets/nomihsa965/dice-detection-upper-view).

A `create_split.py` és `test_split.py` már szerepelt a csapat repójában. A jelen kiegészítés az adatellenőrzőt, a címkegalériát, az adatösszesítést, a felosztó felülírás elleni védelmét és besorolási jegyzékét, valamint ezek dokumentációját és célzott tesztjeit adja hozzá. A saját beszámolóban ezt a kiegészítést érdemes elkülöníteni a korábbi csapatmunkától.
