# Viikko 3

## 17.11. keskiviikko
Toteutettu calculate_eigenfaces-metodi loppuun, se on tarpeettoman iso. Luotu operations.py-moduuli ominaiskasvojen laskemiseen liittyvälle toiminnallisuudelle, ja siirretty sinne mm. tämä calculate_eigenfaces individual-luokasta. Toimintaa testattu toistaiseksi vain mainissa.

## 18.11. torstai
Lisätty testit operations-moduulille ja siirretty sinne individual-luokan testit, ja lisätty jotain muita testejä. Metodia calculate_eigenfaces ei testata toistaiseksi muuten kuin tarkistamalla palautettavan matriisin koko. Sille on hankala kirjoittaa järkevämpiä testejä sen koon takia, joten ehkä se täytyy hajottaa useammaksi pieneksi funktioksi. Täytyy kuitenkin toteuttaa se ennen sitä kokonaisuudessaan.
Lisätty konstruktoritesti individualin testifileen, jossa ei muuten ole mitään.

## 19.11. perjantai
Suoritettu lähteenä käytettävän paperin menetelmä pienelle kuvasetille mainissa, tulokset ei vaikuttaneet kovin järkeviltä. Kaikki aika meni lähinnä ohjelman rakenteen miettimiseen.

## 20.11. lauantai
Luettu ohjelmassa nyt vihdoin kaikki kuvat ja tehty niistä Individual-olioita. Lisätty individual-luokkaan attribuuteiksi training_images ja test_images, jotka valitaan luokan train_test_split-metodissa: toistaiseksi ensimmäiset 8 kuvaa menee training-settiin ja loput 2 testisettiin.
Tehty operations-moduuliin metodi, joka palauttaa tuplena matriisin kaikista training-kuvista ja matriisin kaikista testisetin kuvista. Lopulta vaikuttaa siltä, ettei kaikkia testikuvia sisältävää matriisia tarvita välttämättä mihinkään.

Lisäksi jatkettu calculate_eigenfaces-metodia siten, että se palauttaa joko parametrina annetun k:n verran ominaisvektoreita. Jos sitä ei ole annettu, palautetaan niin monta ominaisvektoria, mitä tarvitaan siihen, että ne edustavat kohtalaisesti training settiä. Tätä varten sovellettu koodia wikipedian Eigenface-artikkelista. Tämä tekee metodin testaamisesta vaikeaa, jos parametria k ei anna. Lisäksi nimi ei ole erityisen kuvaava, koska kyseessä ei vielä ole lopulliset eigenfacet, vaan välivaihe.

Laskettu mainissa training-kuville keskiarvo ja eigenfacet. Lisäksi lisätty luokka Faces, joka osoittautui ainakin toistaiseksi tarpeettomaksi.

## Ajankäyttö
17 h

## Ongelmia
Ohjelman toiminta on epäselvää ja testaaminen hankalaa. Jos teen metodin jossa suoritetaan käytännössä vain peräkkäisiä numpy-metodeja (esim. calculate_eigenfaces tällä hetkellä), onko sen yksikkötestaaminen mielekästä?

## Seuraava viikko
Ohjelman rakenne täytyisi saada jotenkin kuntoon tai vähintäänkin pitää järkevänä. Lisäksi täytyy vihdoin toteuttaa metodi joka projisoi annetun vektorin ominaisvektorien virittämään avaruuteen. Lisäksi viimeistään lopussa lasketut ominaisvektorit ja eigenfacet voisi olla järkevää kirjoittaa johonkin fileen ja lukea aina sieltä, koska niiden laskeminen vie kuitenkin jonkin verran aikaa.