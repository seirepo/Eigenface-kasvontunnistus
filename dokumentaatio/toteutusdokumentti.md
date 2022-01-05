# Toteutusdokumentti

## Ohjelman rakenne
Ohjelma käynnistettäessä suoritetaan alustetaan `App`-olio ja suoritetaan sen metodi `suorita`. Ensin ladataan kuva-aineisto, luodaan siitä tarvittavat luokkaoliot ja muodostetaan niiden harjoituskuvista matriisi, jonka avulla lasketaan tarvittavat ominaiskasvot. Tämän jälkeen projisoidaan harjoituskuvat saatujen ominaiskasvojen virittämään avaruuteen ja tallennetaan niiden koordinaatit. Lopuksi käydään läpi kaikki testikuvat: ne projisoidaan myös kasvoavaruuteen, ja selvitetään k:n lähimmän naapurin menetelmällä k naapuria valitulla etäisyysmitalla ja määritellään lähin harjoituskuva ja luokka, johon testikuva kuuluu. Lopuksi tulos esitetään käyttöliittymässä jokaisen testikuvan osalta.

## Saavutetut aika- ja tilavaativuudet
Kaikki operaatiot riippuvat kuvien koosta *n* ja määrästä *m*. Suurin käsiteltävä matriisi on kooltaan *n²×m*. Kun kuvat on esitetty M:n ominaiskasvon avulla, niistä säilytetään vain M koordinaattia, jolloin ne mahtuvat *M×m*-matriisiin. Niiden lisäksi ominaiskasvot säilytetään *n²×M*-matriisissa.

## Puutteet ja parannusehdotukset
Käyttöliittymä on hieman sekava, kun siinä näytetään kaikki 160 kuvaa.

## Lähteet
M. Turk; A. Pentland (1991). "Eigenfaces for recognition" (PDF). Journal of Cognitive Neuroscience. 3 (1): 71–86. http://dx.doi.org/10.1162/jocn.1991.3.1.71

P. Pankka (2021). Lineaarialgebra ja matriisilaskenta I-II, kurssimateriaali

J. Kun (2011). Eigenfaces, for Facial Recognition. https://jeremykun.com/2011/07/27/eigenfaces/

Kuva-aineisto: AT&T Laboratories Cambridge, 1992-1994
