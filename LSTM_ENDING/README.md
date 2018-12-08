Osztályozzuk a mondatokat: befejezes 1, nem befejezes 0.<br />
A model tanításához az eredeti train data-t úgy alakítottam át, hogy <br />
egyenlo aranyban legyenek az 1, 2, 3, 4 mondatok és a befejezesek, mivel <br />
ennélkül nagyon elrontotta a kiértékelést a 4x annyi 0 osztályba tartozó mondat.<br />
Tehát ha elérte a befejezések számát az 1,2,3,4 mondatok száma, akkor már csak befejezés kerül a feldolgozott train data-ba.<br />
Megpróbáltam a tanítást w2v embeddinggel is a tanítást, de hosszas próbálkozások után<br />
inkább a példakódban is feltűntetett BOW mellett tettem le a voksom.<br />
A szovegek atesnek az alapvető filterezéseken, majd ezután készül belőlük egy BOW reprezentáció.<br />
<br /><br />
3 periódus után a pontosság:<br />
105333/105333 [==============================] - 1240s 12ms/step - loss: 0.6755 - acc: 0.5611 - val_loss: 0.7034 - val_acc: 0.5005<br />
Epoch 2/3
105333/105333 [==============================] - 1141s 11ms/step - loss: 0.6271 - acc: 0.6543 - val_loss: 0.7395 - val_acc: 0.5069<br />
Epoch 3/3
105333/105333 [==============================] - 1133s 11ms/step - loss: 0.5989 - acc: 0.6630 - val_loss: 0.7706 - val_acc: 0.5155<br />
Accuracy: 51.55%
