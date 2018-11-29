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
<br/>
<br/>
LOGREG a LogReg.predict_proba(X_test) függvény alkalmazásával: 03_wmd_logreg_probability.py<br />
A train és valid data ugyanaz, mint a korábbiaknál.<br />
Visszaad egy listát, ami tartalmazza a már felvett pontok szerinti osztályozási valószínűségeket.<br />
Így nem függetlenül döntünk az osztályozásról, hanem ahol a 2 mondat közül megkapjuk, melyik valószínűbb a helyes befejezésre!<br />
Az előzőekhez képest duplázódott a pontosság, de így sem valami jó.<br />
A fals negatív viszont kb itt is ugyan olyan.<br />
Accuracy: 0.4739583333333333<br />
False Positive: 0.2526041666666667<br />
False Negative: 0.2734375<br /><br /><br /><br />
A wmd-s csv-ket meghatározó converterben egyazon modellel taníttattam be a validation és a teszt adatokat, így javult a pontosság:<br />

Accuracy: 0.5078534031413613<br />
False Positive: 0.23821989528795812<br />
False Negative: 0.25392670157068065<br />
<br/><br/><br/>
Nem a feladathoz tartozo train data-n, hanem google-s keyedvectoron tanulva:<br/>
Accuracy: 0.5390625<br/>
False Positive: 0.24739583333333334<br/>
False Negative: 0.21354166666666666<br/>
