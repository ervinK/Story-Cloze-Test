A feltevés az volt, hogy az egyes tanítópéldák a feldolgozás után nagyon hasonló cosine <br />
similarity-vel rendelkeznek, és ez a tulajdonság a különböző osztályba tartozástól független.<br />
Ezért vizsgáljuk a validacios halmazban vett mondatok mindegyikére a top70 hasonló mondatot(~50k van, tehát ez a szám véleményem szerint még belefér). <br />
Abban az esetben, ha egy mondat(vagy mondatok) sok mondat top-hasonlóság listájában ott van/vannak, <br />
akkor az igazolja, hogy egy model osztályozás során kvázi overfitel a hasonló szerkezetű mondatok miatt.<br />
