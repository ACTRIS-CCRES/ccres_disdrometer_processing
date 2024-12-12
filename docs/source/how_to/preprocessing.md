# Further information on preprocesssing

 ![Schema](../assets/Schema_fonctionnel_dcrcc-Page-2.drawio.png)
Focus on the preprocessing step (needs to be slightly modified)


The preprocessing step is made by a specific command line (inputs are listed in the page before).
It is based on several scripts :


- the script _ccres_disdrometer_processing.open_weather_netcdf_ is used to open and reformat input daily weather netCDF files from CloudNet, and outputs a -(xarray.Dataset :
* renommage des variables météo ws, wd, u, v, ta, hur, ps, ams_pr, ams_cp
* rééchantillonnage sur vecteur de pas de temps parfaits @ 1 minute (la valeur affectée à une variable à T est la première valeur trouvée dans l’intervalle [T-30s,T+30s] pour cette variable dans le fichier d’origine
* attributs globaux récupérés du fichier d’origine ; métadonnées pour les variables

- Le script ccres_disdrometer_processing.open_radar_netcdf ouvre et reformatte les fichiers DCR journaliers venant de Cloudnet et renvoie un xarray.Dataset :
* extraction des données de réflectivité et vitesse Doppler de 0 à 2500m d’altitude (Zdcr, DVdcr)
* rééchantillonnage sur temps parfaits @ 1 minute : Zdcr(T) = médiane(Z([T-30s, T+30s]) et Dvdcr(T) = moyenne(DV([T-30s, T+30s])
* attributs

- Le script ccres_disdrometer_processing.open_disdro_netcdf ouvre et reformatte un fichier journalier DD CloudNet et fait le forward modeling des réflectivités, et renvoie un xr.Dataset :
* ouverture, renommages, resampling @1mn, méta : fonction read_parsivel_cloudnet_choice (variante selon Thies/Parsivel en entrée car les variables dans les fichiers CloudNet diffèrent un peu) ; création d’une dimension « radar_frequencies » pour produire les calculs de diffusion à toutes les fréquences qu’on peut rencontrer dans le réseau d’instruments.
* La fonction reflectivity_model_multilambda_measmodV_hvfov réalise le calcul des variables de diffusion (réflectivités Mie/T-matrice, moments, atténuation...), et renvoie en sortie un fichier avec les données disdro reformattées + les variables calculées par le modèle de diffusion.
Pour les réflectivités on calcule 16 cas : (4 fréquences) * (2 géométries(Horizontal FOV, Vertical FOV)) * (2 variantes de calcul des vitesses de chute en fonction du diamètre pour la normalisation des réflectivités : moyenne pondérée par classe de diamètre basée sur la DSD ou bien calcul de V(diamètre) à partir d’un modèle)
Entrées :
1/ Le Dataset avec les données DD reformattées
2/ les vecteurs de coefficients de rétrodiffusion Mie/T-matrice et atténuation T-matrice issus de la fonction ccres_disdrometer_processing.scattering.scattering_prop() pour les 8 configurations (4 fréquences, 2 géométries)
3/ le nombre de classes de diamètre à considérer dans les calculs de moments/Z (on tronque les 5 dernières classes de diamètre)
4/ Le choix du modèle de calcul des vitesses de chute.


Hypothèses faites :
* F la surface d’échantillonnage, est propre à chaque disdro et est fournie en fichier de config : Parsivel : la même pour tous les instruments ; Thies : dépend de l’instrument (valeur de AU propre à chaque Thies). F intervient dans la normalisation des réflectivités
* la formule utilisée pour modéliser les vitesses de chute est celle de (Gun and Kinzer, 1949). Il y a d’autres formules proposées dans la littérature (Khvorostyanov and Curry 2002, Atlas and Ulbrich 1977, …), pas implémentées dans le code mais peut-être à tester, mais l’usage de la formule de Gun et Kinzer est très répandu
* Pour les calculs de diffusion par T-matrix on considère que les gouttes ne sont pas sphériques mais elliptiques, de forme décrite par le ratio grand axe/petit axe = f(Diamètre). La relation utilisée est un fit polynomial d’ordre 4 en D proposé par (Andsager et al. 1999) basé sur une formule de (Beard, Chuang 1987)
* On a des gouttes d’eau entièrement liquides. Dans le code on utilise l’indice de réfraction n = 2.99645 + 1.54866j Je crois que c’est valable pour f=94GHz et T = 25°C. C’était la valeur utilisée quand le code calculait seulement les réflectivités Mie pour le BASTA. Devrait-on prendre n(T,f) ou a minima n(f), à quel point cela impacterait les résultats ? Il y a des tables de valeurs des coefficients en fonction de la fréquence, un calcul est décrit dans (Ray 1972) en reprenant des équations de (Cole and Cole 1941) et il y a des tables de valeurs et des fit n(f) dans d’autres papiers plus récents)


Le fichier de preprocessing est le fruit de la concaténation des 3 fichiers reformattés (1 par instrument), 2 s’il n’y a pas de données météo.
La CLI ccres_disdrometer_processing preprocess, réalise le prétraitement en prenant en entrée le chemin vers les 3 fichiers CloudNet, le fichier de config (format TOML) correspondant à la station et aux instruments à traiter, et le chemin où sauvegarder le fichier résultat.
