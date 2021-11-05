1. Mitä ohjelmointikieltä käytät? Kerro myös mitä muita kieliä hallitset siinä määrin, että pystyt tarvittaessa vertaisarvioimaan niillä tehtyjä projekteja.
Projekti toteutetaan Pythonilla. Voin vertaisarvioda Javalla tehtyjä projekteja. Koodi, kommentointi ja dokumentit kirjoitetaan suomeksi. Koulutusohjelma tietojenkäsittelytieteen kandidaatti (TKT).

2. Mitä algoritmeja ja tietorakenteita toteutat työssäsi?
Työssä täytyy toteuttaa todennäköisesti sopiva tietorakenne kuvaamaan vektoria ja matriisia, sekä niille vaadittavat operaatiot. Näitä on vektoreille esimerkiksi pistetulo, vektorien yhteenlasku, skalaarilla kertominen ja transpoosi, ja matriiseille ainakin matriisitulo ja transpoosi, ja mahdollisesti esim. [][]-operaatio.

3. Mitä ongelmaa ratkaiset ja miksi valitsit kyseiset algoritmit/tietorakenteet?
Aion toteuttaa Eigenface-kasvojentunnistusohjelman. Lähes kaikki käytettävät algoritmit ovat erilaisia matriisioperaatioita, joten tarvittavat tietorakenteet ovat luonnollisesti vektori ja matriisi. Ohjelma kouluttamiseksi tarvitaan riittävästi kasvokuvia eri henkilöistä, jotka rajataan oikean kokoisiksi, esimerkiksi 100x100. Näistä kasvokuvista muodostetaan pääkomponenttianalyysin (PCA) avulla ominaiskasvoja, eli vektoreita, jotka voidaan esittää myös kuvana. Ne virittävät kasvoavaruuden, joka on aliavaruus yleisessä kuva-avaruudessa. Ohjelman avulla voidaan selvittää sille annetusta kuvasta, onko se lähellä kasvoavaruutta, eli muistuttaako se kasvoja.

4. Mitä syötteitä ohjelma saa ja miten näitä käytetään?
Ohjelmalle voi antaa syötteeksi tietynkokoisen kuvan, jonka jälkeen sen tulisi kertoa, onko kuvassa kasvot vai ei. Syötteeksi annettu kuva projisoidaan kasvoavaruuteen, ja esitetään ominaiskasvojen lineaarikombinaationa, jonka jälkeen selvitetään kuinka kaukana se on kasvoaliavaruudesta ja muista kasvoista. Jos kuva on lähellä kasvoavaruutta, sen tulkitaan sisältävän kasvot, ja muussa tapauksessa ei. Ensimmäisessä tapauksessa voidaan selvittää kuvavektorin etäisyys kaikista tunnetuista kasvoluokista. Esimerkiksi yksi tai useampi kuva samasta henkilöstä voi muodostaa kasvoluokan. Näin kuvassa oleva henkilö voidaan identifioida, jos se on tarpeeksi lähellä jotain kasvoluokkaa. Jos kuva ei ole lähellä mitään tunnettua kasvoluokkaa,

5. Tavoitteena olevat aika- ja tilavaativuudet (m.m. O-analyysit)
Tarvittava tila riippuu kuvien koosta ja määrästä $M$. Esim. 100x100-kokoiset kuvat talletetaan 10000-dimensioisena vektorina. Muodostettavat matriisit ovat tässä tapauksessa enimmillään $M \times 10000$.

6. Lähteet
M. Turk; A. Pentland (1991). "Eigenfaces for recognition" (PDF). Journal of Cognitive Neuroscience. 3 (1): 71–86. doi:10.1162/jocn.1991.3.1.71. PMID 23964806.


7. Kurssin hallintaan liittyvistä syistä määrittelydokumentissä tulee mainita opinto-ohjelma johon kuulut. Esimerkiksi tietojenkäsittelytieteen kandidaatti (TKT) tai bachelor’s in science (bSc)

8. Määrittelydokumentissa tulee myös mainita projektin dokumentaatiossa käytetty kieli (todennäköisesti sama kuin määrittelydokumentin kieli). Projektin koodin, kommenttien ja dokumenttien teksti on valitulla kielellä. Tyypillisesti suomi tai englanti. Tämä vaatimus liittyy projektin puolen välin paikkeilla järjestettäviin koodikatselmointeihin. Tavoitteena on että projektien sisäiset kielivalinnat ovat johdonmukaisia.

