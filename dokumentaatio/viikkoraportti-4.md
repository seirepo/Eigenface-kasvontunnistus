# Viikko 4

## 24.11.2021 ke
Löydetty bugi calculate_eigenfaces-metodista kohdassa, jossa lasketaan itse ominaiskasvoja. Ulkoistettu se omaan get_eigenfaces-metodiin ja lisätty sille yksinkertainen testi.

## 25.11.2021 to
Kokeiltu kuvien projisointia, se ei onnistunut ollenkaan. Laskettujen ominaiskasvojen arvot ovat kymmenkertaisia suhteessa keskiarvokasvoihin, jonka arvot on välillä $[0,1]$. Skaalattu siis myös ominaiskasvojen kaikki arvot tälle välille. Projisointi ei kuitenkaan onnistunut kovin hyvin, koska saaduilla koordinaateilla ei pystytty rekonstruoimaan projisoitavaa kasvoa lähellekään alkuperäistä. Ilmeisesti ominaisvektoreille olisi tarvittu vielä jokin kerroin. [Tästä blogista](https://jeremykun.com/2011/07/27/eigenfaces/) sain kuitenkin vinkin helpommalle tavalle laskea koordinaatteja, kun ominaiskasvojen muodostama kanta on ortonormaali.
Muokattu calculate_eigenfaces-metodi palauttamaan ortonormalisoidut ominaiskasvot, joiden arvot on skaalattu välille $[0,1]$. Tämän jälkeen projisointi onnistui.

## 26.11.2021 pe
Datan saakin käyttöön suoraan scikit learn-kirjaston kautta. Lisätty se riippuvuudeksi, ja nyt ohjelma lataa kuva-aineiston data-kansioon.