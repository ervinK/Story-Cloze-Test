Csak befejezések figyelembe vétele LSI modellel<br />
<br />
<br />
Az alapvető ötlet az, hogy a tényleges train data-ból az 5. mondatokat tanítom be a modelnek.<br />
Természetesen a mondatok preprocesszált formáit.<br />
Így kapok egy olyan modelt, ami kb az "értelmes" befejezésekkel van tisztában.<br />
A kísérlet az volt, hogy az új mondatoknak veszem a top100 most similar mondatát, és azoknak az összegét figyelem, amelyikhez több hasonlóvan, az nyer.<br />
Azért gondoltam rá, mivel csak jó mondatokkal lett betanítva, ezért evidens lenne, hogy érzékelje a szemantikus különbségeket jó és rossz mondatok között. <br />
<br />
<br />
Eredmények: <br />
<br />
Megadtam változtatható paraméternek, hogy mennyi szöveget vegyen figyelembe, illetve mennyi példán teszteljen a program. Ezeket próbálom most hangolni, eddig a most similar 100 vizsgálatával 100 példán 53% volt a legjobb pontosság.<br />
Pozitívum: vettem pár példát csak szemmel, és olyan mondatpárokat NAGYON elvágott egymástól jó irányba, amiket eddig mondjuk a Word2Vec/WMD kettősnek gondolt.
