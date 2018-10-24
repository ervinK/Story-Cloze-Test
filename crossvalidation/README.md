wmdConverter:
-beolvasom a kb 70-30 arányban szétszedett validation halmaz train és test felét
-elvégzem rajtuk a szükséges lingvisztikai átalakításokat (kisbetű, contractionök)
-majd felveszem mind a 4 mondat WMD-jét egyszer az első, majd a második befejezéshez mérve
-hozzáveszem a WMD-k összegét is
-majd ezt exportálom egy .csv-be
-ezt alkalmazom a train és test részre is

wmd_logreg:
-felveszem a train rész alapján az adatokat
-majd predictelem, hogy a test rész, hogyan értékelődik ki ennek függvényében

kimenet:

True pozitiv 0,1 osztalyozas: 0.50390625
True pozitiv 1 osztalyozas: 0.21875
True negativ 0 osztalyozas: 0.28515625
Fals pozitiv 1 osztalyozas: 0.21484375
Fals negativ 0 osztalyozas: 0.28125

következtetés:
-nagyobb részben probléma az, hogy nem tituláljuk befejezésnek az adott befejezést
-ez abból következik, hogy az osztályozás nem páronként diszjunkt, így lehet 0 0 és 1 1 értékelésünk is
