# Viikko 2

## 9.11. tiistai
Sovittu ohjauksessa projektin toteutuksesta tarkemmin. Muutettu suunnitelmaa: tehdään jokaiselle henkilölle omat ominaiskasvot harojitusdatan pohjalta, ja pyritään tunnistamaan eri henkilöt käyttäen k-lähimmän naapurin menetelmää. Lisätty kansio koodifileille ja sinne kansio testeille. Asennettu pylint kehityksenaikaiseksi riippuvuudeksi.

## 10.11. keskiviikko
Suunniteltu paperilla toteutusta ja käyttöliittymää, tuloksena ei vielä mitään järkevää. Etsitty sopivia kuvasettejä ohjelmaa varten ja todettu että sopivaa settiä on todella hankala löytää. Jotta pääsen alkuun, valitsin [tämän](https://www.kaggle.com/serkanpeldek/face-recognition-on-olivetti-dataset/notebook), jossa on 10 kuvaa 40 eri henkilöstä, koska data oli siinä sopivassa muodossa. Nyt harjoitusdataan valikoituu 8 kuvaa ja ohjelman testausta varten jää vain 2 jokaisesta henkilöstä.

## 11.11. torstai
Tehty alustava luokka yhden henkilön kuvia varten. Asennettu numpy ja matplotlib kuvien näyttämistä varten. Selvitetty miten numpy toimii, miten ohjelma lukee kuvat, missä muodossa ne ovat ja miten ne voidaan näyttää. Tähän meni lähes koko aika. Kuvat tallennettu yksilöluokkaan nyt matriisina, jossa jokainen sarake vastaa yhtä kuvaa. Tätä varten luotu metodi ja kirjoitettu sille muutama testi. Luotu toinen metodi näiden kuvien näyttämistä varten.

## 12.11. perjantai
Jatkettu funktioiden toteutusta yksilöluokkaan ja tehty niistä toistaiseksi staattisia sekä muokattu eilen tehtyjä testejä.

## 13.11. lauantai
Luotu yksilöluokkaan metodi calculate_eigenface ja aloitettu sen toteuttaminen: toistaiseksi se tuottaa vain kuvamatriisin, jossa jokaisesta harjoitusdata kuvasta on vähennetty niiden keskiarvo. Lisätty README:n badget testeistä ja testikattavuudesta.

## Ajankäyttö
Aikaa kului yhteensä noin 18 h.

## Seuraava viikko
Seuraavalla viikolla pitäisi saada funktio calculate_eigenface tehtyä, jonka jälkeen toistaiseksi vain yhtä henkilöä käsittelevän ohjelman voisi laajentaa koskemaan kaikkia muitakin. Parhaassa tapauksessa minulla olisi tällöin valmiina suurin osa lopullista vertailua varten.
