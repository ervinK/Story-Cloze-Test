wmdConverter:<br />
-beolvasom a kb 70-30 arányban szétszedett validation halmaz train és test felét<br />
-elvégzem rajtuk a szükséges lingvisztikai átalakításokat (kisbetű, contractionök)<br />
-majd felveszem mind a 4 mondat WMD-jét egyszer az első, majd a második befejezéshez mérve<br />
-hozzáveszem a WMD-k összegét is<br />
-majd ezt exportálom egy .csv-be<br />
-ezt alkalmazom a train és test részre is<br />

wmd_logreg:<br />
-felveszem a train rész alapján az adatokat<br />
-majd predictelem, hogy a test rész, hogyan értékelődik ki ennek függvényében<br />

kimenet:<br />

True pozitiv 0,1 osztalyozas: 0.50390625<br />
True pozitiv 1 osztalyozas: 0.21875<br />
True negativ 0 osztalyozas: 0.28515625<br />
Fals pozitiv 1 osztalyozas: 0.21484375<br />
Fals negativ 0 osztalyozas: 0.28125<br />
<br />
következtetés:<br />
-nagyobb részben probléma az, hogy nem tituláljuk befejezésnek az adott befejezést<br />
-ez abból következik, hogy az osztályozás nem páronként diszjunkt, így lehet 0 0 és 1 1 értékelésünk is<br />
