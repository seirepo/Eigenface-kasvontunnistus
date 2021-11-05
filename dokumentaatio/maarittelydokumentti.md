1. Mitä ohjelmointikieltä käytät? Kerro myös mitä muita kieliä hallitset siinä määrin, että pystyt tarvittaessa vertaisarvioimaan niillä tehtyjä projekteja.

3. Mitä ongelmaa ratkaiset ja miksi valitsit kyseiset algoritmit/tietorakenteet?
## Projekti
Aion toteuttaa Eigenface-kasvojentunnistusohjelman. Lähes kaikki käytettävät algoritmit ovat erilaisia matriisioperaatioita, joten tarvittavat tietorakenteet ovat luonnollisesti vektori ja matriisi. Ohjelma kouluttamiseksi tarvitaan riittävästi kasvokuvia eri henkilöistä, jotka rajataan oikean kokoisiksi, esimerkiksi 100x100. Näistä kasvokuvista muodostetaan pääkomponenttianalyysin (PCA) avulla ominaiskasvoja, eli vektoreita, jotka voidaan esittää myös kuvana. Ne virittävät kasvoavaruuden, joka on aliavaruus yleisessä kuva-avaruudessa. Ohjelman avulla voidaan selvittää sille annetusta kuvasta, onko se lähellä kasvoavaruutta, eli muistuttaako se kasvoja.

## Algoritmit ja tietorakenteet
Työssä täytyy toteuttaa sopivat tietorakenteet kuvaamaan vektoria ja matriisia, sekä niille vaadittavat operaatiot. Näitä on vektoreille esimerkiksi pistetulo, vektorien yhteenlasku, skalaarilla kertominen ja transpoosi, ja matriiseille ainakin matriisitulo ja transpoosi, ja mahdollisesti esim. [][]-operaatio.



Ohjelmalle voi antaa syötteeksi tietynkokoisen kuvan, jonka jälkeen sen tulisi kertoa, onko kuvassa kasvot vai ei. Syötteeksi annettu kuva projisoidaan kasvoavaruuteen, ja esitetään ominaiskasvojen lineaarikombinaationa, jonka jälkeen selvitetään kuinka kaukana se on kasvoaliavaruudesta ja muista kasvoista. Jos kuva on lähellä kasvoavaruutta, sen tulkitaan sisältävän kasvot, ja muussa tapauksessa ei. Ensimmäisessä tapauksessa voidaan selvittää kuvavektorin etäisyys kaikista tunnetuista kasvoluokista. Esimerkiksi yksi tai useampi kuva samasta henkilöstä voi muodostaa kasvoluokan. Näin kuvassa oleva henkilö voidaan identifioida, jos se on tarpeeksi lähellä jotain kasvoluokkaa. Jos kuva ei ole lähellä mitään tunnettua kasvoluokkaa,
Tarvittava tila riippuu kuvien koosta ja määrästä $M$. Esim. 100x100-kokoiset kuvat talletetaan 10000-dimensioisena vektorina. Muodostettavat matriisit ovat tässä tapauksessa enimmillään $M \times 10000$.

Projekti toteutetaan Pythonilla. Voin vertaisarvioda Javalla tehtyjä projekteja. Koodi, kommentointi ja dokumentit kirjoitetaan suomeksi. Koulutusohjelma tietojenkäsittelytieteen kandidaatti (TKT).

Lähteet
M. Turk; A. Pentland (1991). "Eigenfaces for recognition" (PDF). Journal of Cognitive Neuroscience. 3 (1): 71–86. doi:10.1162/jocn.1991.3.1.71. PMID 23964806.
