## Projekti
Aion toteuttaa Eigenface-kasvojentunnistusohjelman julkaisussa Eigenfaces for recognition (Turk & Pentland, 1991) kuvatun algoritmin avulla. Lähes kaikki algoritmin vaativat toiminnallisuudet ovat erilaisia matriisioperaatioita, joten tarvittavat tietorakenteet ovat luonnollisesti vektori ja matriisi. Lisäksi ohjelman kouluttamiseksi tarvitaan riittävästi kasvokuvia eri henkilöistä, kuva-aineistona käytetään vuosina 1992-1994 kuvattuja kasvoja (AT&T Laboratories Cambridge). 
Näistä kasvokuvista valitaan noin 80 % ja muodostetaan niistä pääkomponenttianalyysin (PCA) avulla ominaiskasvoja. Ohjelman tarkoitus on tunnistaa lopuissa 20 % kuvissa olevat henkilöt.

## Algoritmit ja tietorakenteet
Työssä täytyy toteuttaa sopivat tietorakenteet kuvaamaan vektoria ja matriisia sekä niille vaadittavat operaatiot, jotka toteutetaan käyttämällä Pythonin Numpy-kirjastoa. Kuvien tunnistaminen toteutetaan k:n lähimmän naapurin menetelmällä.

## Toiminta
Ohjelmalla voi tunnistaa henkilöitä sille annetuista kuvista. Kuva projisoidaan kasvoavaruuteen, ja esitetään ominaiskasvojen lineaarikombinaationa, jonka jälkeen selvitetään sen etäisyys muista kasvoista. Kuva luokitellaan siihen luokkaan, joka on eniten edustettuna sen naapureissa, kun luokan muodostaa jokin tietty henkilö.

Tarvittava tila riippuu kuvien koosta ja määrästä $M$. Esim. 100x100-kokoiset kuvat talletetaan 10000-dimensioisena vektorina. Koulutuksen jälkeen kuvat on mahdollista säilyttää vain mukaan valittujen ominaiskasvojen kertoimina, jolloin ne vievät vähemmän tilaa. Muodostettavat matriisit ovat tässä tapauksessa enimmillään $M \times 10000$. Näin ollen operaatioiden aikavaativuus on $O(n^2 \cdot M)$, jossa $n^2$ on kuvavektorin pituus. Matriisioperaatioista taas matriisitulon, determinantin ja matriisiyhtälön osalta aikavaativuus tulee olemaan $O(n^3)$. Koko ohjelman aikavaativuus voisi tulla olemaan $O(n^3)$ ja tilavaativuus $O(n^2 \cdot m)$.

Projekti toteutetaan Pythonilla. Voin vertaisarvioda Javalla tehtyjä projekteja. Koodi, kommentointi ja dokumentit kirjoitetaan suomeksi. Koulutusohjelma on tietojenkäsittelytieteen kandidaatti (TKT).

## Lähteet
M. Turk; A. Pentland (1991). "Eigenfaces for recognition" (PDF). Journal of Cognitive Neuroscience. 3 (1): 71–86. http://dx.doi.org/10.1162/jocn.1991.3.1.71

Kuva-aineisto: AT&T Laboratories Cambridge, 1992-1994
