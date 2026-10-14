# Singing Voice Separation
L'idee de ce projet est de tests l'avancer des modeles dans la sereparation de voix musicale avec les instruments qui compose l'audio.

Les contraintes qui vont servir de base sont celle du materielle utilise et serviront de fil conduteur.
4Go de vram pour l'entrainement, donc comment un petit model peut utiliser l'avancer de l'etat de l'art pour parvenir à realiser cette tache.

## 0. Creation Baseline avec modele historique ([Jansson et al., *Singing Voice Separation with Deep U-Net Convolutional Networks*, ISMIR 2017](https://archives.ismir.net/ismir2017/paper/000171.pdf))
Pour commencer, j'ai reimplementé la base qui sert de base au domaine avec un UNet. Il consiste d'un encodeur-decoder avec des couches de convolution et concatenation. L'entree du modele est la magnitude d'un son mono, puis on applique un stft en sorti de decoder on va vont venir construire un masque entre [0, 1] pour permettre de definir la quantite de voix pour chaque temps x fequence. Ce masque permet ensuite la reconstruction du morceau via la phase de la mixture + le voix estime (mixture * masque).

## 1. Dataset MUSDB18
Le dataset utilise est le MUSDB18, donc la version mixture est compressé et n'est pas exactement egale à voix + instrumental, donc pour l'apprentissage on utilise la mixture real qu'on reconstruit avec la voix + instrumental. Donc en inference on à une erreur interne au model.

### Variation entre la mixture officielle et voix + instrumental

(`python -m analysis.mixture_gap`, sur les 150 morceaux)

![ecart mixture](figures/mixture_gap.png)

- Le residu `mixture − (voix + instrumental)` est a **−30,7 dB** de la mixture en median (10e percentile : −28,4 dB ; entre −26 et −39 dB pour tous les morceaux sauf un, voir plus bas). L'ecart est donc faible.
- Il est concentre au-dessus de **17 kHz** (residu superieur a la mixture au-dela, probablement a cause de la coupure passe-bas de l'encodage AAC, non verifie), c'est-a-dire dans une zone ou il n'y a presque pas d'energie utile.
- Pour savoir si ca a un impact sur la separation, j'ai calcule le masque oracle sur la mixture reconstruite puis je l'ai applique a la vraie mixture (48 morceaux de test avec voix exploitable, 3 extraits de 6 s chacun) : SI-SDR median de **10,21 dB** (mixture reconstruite) contre **10,17 dB** (vraie mixture), soit **0,05 dB** d'ecart median. L'erreur interne est donc negligeable pour ce probleme.
- Exception : *PR - Oh No* (mixture a seulement 7,6 dB du residu) et *PR - Happy Daze*, deux morceaux Native Instruments dont la voix est quasi inaudible (RMS median de −86 dB pour *Happy Daze*, 67 % des fenetres de 1 s sous −60 dB). Le residu y est plus fort que la voix (SNR voix/residu minimal de −9,4 dB).

[TODO]{test entrainement avec vraie mixture et comparer} : le proxy ci-dessus (oracle) suggere un effet quasi nul, mais le test complet (entrainer et evaluer sur `mixture.wav`) n'est pas encore lance.

### Repartition des sons (style, interprete)

(`python -m analysis.dataset_distribution`)

![repartition](figures/dataset_distribution.png)

| Genre | train (84) | val (16) | test (50) |
|---|---|---|---|
| Pop/Rock | 33 % | 44 % | **72 %** |
| Rock | 18 % | 12 % | 0 % |
| Singer/Songwriter | 14 % | 12 % | 0 % |
| Heavy Metal | 6 % | 19 % | 8 % |
| Pop | 12 % | 6 % | 0 % |
| Rap | 6 % | 0 % | 6 % |
| Electronic | 5 % | 0 % | 8 % |
| Jazz | 4 % | 0 % | 0 % |
| Country | 2 % | 6 % | 0 % |
| Reggae | 0 % | 0 % | 4 % |
| inconnu (absent du README MUSDB) | 0 % | 0 % | 2 % |

- Le test est tres desequilibre : 72 % de Pop/Rock, **aucun** Rock, Singer/Songwriter, Pop ni Jazz, et il ne contient aucun morceau MedleyDB (que des DSD, 2 Native Instruments et 2 du concours Heise). Les morceaux MedleyDB (46 % du train+val) ne servent donc qu'a l'entrainement.
- Le reggae n'apparait qu'en test (2 morceaux) et le jazz qu'en train (3). Les scores par genre sur le test reposent sur 2 a 4 morceaux pour tous les genres sauf Pop/Rock (36) : a prendre comme des ordres de grandeur.
- Interpretes : 3 artistes sont a la fois dans le train et le test (*Hollow Ground*, *Skelpolu*, *The Long Wait*), soit 3 morceaux de test sur 50. Le split train/val est fait par artiste, il n'y a aucun artiste commun entre train et val.
- Durees : mediane autour de 4 min dans les trois sous-ensembles ; le train contient quelques morceaux tres courts (*Music Delta*, de 13 a 26 s) et le plus long fait 10,5 min.

![artistes](figures/dataset_artists.png)

## 2. Iteration
Lors de l'implementation, j'ai pu me rendre compte que la tete du reseau appliquait un ReLu en trop et donc venait plafonner la sortie de 46% des cellules ce qui reduisait fortement l'apprentissage.

Ce papier fondateur donne des resultats qui ne sont pas comparable à d'autre papier mais j'ai pu utiliser un oracle qui va venir definir un plafond pour l'estimation.

Avec ce premiere entrainement, on peut se rendre compte de nombreux point qui pourrait etre ameliore pour essaye de donner plus de contexte au modele:
- l'utilisation de la phase de la mixture ne permet pas de correctement reconstruire les frequence ou les instruements etait dominant. Pour ce faire on pourrait le donner en entree comme à pu le faire [Choi et al., ISMIR 2020](https://arxiv.org/abs/1912.02591) (U-Net entraines sur le spectrogramme complexe) ou estimer directement un masque complexe, dont l'amplitude peut depasser 1, comme [Kong et al., ISMIR 2021](https://arxiv.org/abs/2109.05418) (ResUNet qui separe l'estimation de l'amplitude et celle de la phase), ce qui implique d'utiliser soit 2 canaux pour apprendre le masque ou bien apprendre un masque complexe, à verifier ci l'apprentissage est faisable sur la machine.
- la compression en signal mono est egalement un approximation qui peut etre facilement change dans l'etat actuel en gardant le stero et en doublant les canaux d'entree.
- pour tenter d'ajouter des notions plus recente on peut modifier le bottlenet pour y ajouter de l'attention comment à pu le faire à son epoque le papier avec son RNN, on pourrait donc ajouter un Transformer meme si l'information est enormement compressé à ce niveau et que pour avoir de l'attention vraiment interessant il faudrait augementer le nombre de canaux recu au bottlenet et donc augementer la taille de l'entree, a voir si cela est faisable sur la machine.
- une autre idee qui a ete introduit par le papier [TS-BSMamba2, Bai et al., 2024](https://arxiv.org/abs/2409.06245) (premier etage de masquage, puis second etage qui apprend un residu pour corriger ce que le masque a manque) est d'utiliser un second model par la suite pour essayer de supprimer les residues d'instrument dans l'output.
- une approche de generation, meme si moins bon que l'etat de l'art actuel, pourrait etre utilise avec comme idee de ne pas creer de son qui est simplement une moyenne mauvaise entre la voix et l'instrument pour les frequences partage mais de toujours recreer un voix pur dans cette situation.
- dans la meme logique de separation des frequences partage, l'instoration de mask pour l'instrumental avec des poids partage pourrait egalement aide comme à pu le faire [Kadandale et al., *Multi-channel U-Net for Music Source Separation*, 2020](https://arxiv.org/abs/2003.10414) (un seul reseau pour plusieurs sources, entraine avec une loss multi-taches ponderee)

Chaque iteration sera ensuite documente avec pour idee démetre une hypothese que le changement dans le modele pourrait apporter/corrige et de comparer les resultats avec l'hypothese.

### 0. Comparaison baseline avec oracle

Actuellement apres avoir, corrige les problemes d'implementation et sans augmentation de donnees, la baseline est fixe sur le jeu de test à :
SDR médian sur 50 morceaux : 4.70 dB  IC95% bootstrap [3.80, 5.92]
SI-SDRi médian sur 50 morceaux : 10.03 dB  IC95% bootstrap [8.80, 10.60]
SDR médian oracle : 9.21 dB  IC95% bootstrap [8.55, 10.24]
SI-SDRi médian oracle : 16.20 dB  IC95% bootstrap [14.72, 17.32]

(`python -m analysis.baseline_vs_oracle`, a partir de `outputs/oracle_baseline.json`)

![baseline vs oracle](figures/baseline_vs_oracle.png)

L'oracle est le masque ideal borne a [0, 1] applique a la mixture avec sa phase (meme reconstruction que le modele). C'est donc le meilleur resultat possible pour ce type de modele, pas un plafond absolu.

- **Le modele atteint environ la moitie de l'oracle** : 4,70 dB contre 9,21 dB, soit un ecart median de **4,32 dB** (IC95 % bootstrap [4,04, 4,69]). le modèle est moins bon que l’oracle sur 50 morceaux sur 50. Il reste environ 4,3 dB de marge sur l’estimation du masque, avant d’atteindre la limite imposée par la phase de la mixture. Au-delà, l’amélioration demanderait de traiter la phase (entrée complexe ou masque complexe).
- **La difficulte vient surtout du morceau** : la correlation de rang entre SDR du modele et SDR de l'oracle est de **0,90** (Spearman). Les morceaux difficiles pour le modele le sont deja pour l'oracle (voix noyee dans l'instrumental, bins ou voix et instruments se superposent).
- **L'ecart oracle − modele est assez regulier** (mediane 4,3 dB, la plupart des morceaux entre 3 et 6 dB), ce qui plaide pour une perte « de modele » plutot que pour quelques cas pathologiques. La valeur extreme a 16,4 dB (*PR - Happy Daze*) est un artefact : la voix y est quasi inaudible (voir plus haut), son SDR oracle vaut 0,00 dB et le SDR du modele −16,4 dB, alors que son SI-SDRi oracle est de 29,8 dB. Les deux morceaux PR sont les seuls du test avec un SDR oracle quasi nul et tirent le genre « Electronic » vers le bas.
- **Le SDR median est une mesure plus severe que le SI-SDRi** (4,70 contre 10,03 dB pour le modele) : le SDR museval penalise la distorsion de la voix et pas seulement la fuite d'instruments, et il est calcule par fenetre de 1 s. Ces chiffres sont en mono sur MUSDB18 compresse avec une mixture reconstruite, et ne sont **pas comparables** aux papiers (stereo, MUSDB18-HQ, cSDR).
- **Par genre** (figure ci-dessous), l'ecart est le plus grand sur le Rap (+5,6 dB, n = 3) et le Reggae (+5,2 dB, n = 2) ; Pop/Rock (n = 36) est a +4,2 dB. Avec 2 a 4 morceaux par genre en dehors du Pop/Rock, ces differences ne sont pas significatives.

![par genre](figures/baseline_by_genre.png)

Cette baseline est entrainee sans augmentation de donnees : le chiffre sert de reference pour les iterations suivantes, a comparer avec `python -m analysis.compare_runs`.

### 1. Passage de Mono à Stereo 

l'idee est que le stereo donne une information spacial sur le son avec la voix qui est majoritairement regroupe au centre alors que les instruments sont repartie sur l'ensemble du spectre.

(`python -m analysis.stereo_analysis`) Decomposition Mid = (L+R)/2 (c'est exactement le mono du pipeline) et Side = (L−R)/2 : une source au centre a un Side proche de zero.

![spectrogrammes stereo](figures/stereo_spectrograms.png)

Sur les 94 morceaux d'entrainement/validation avec une voix exploitable (fenetre centrale de 20 s) :

![statistiques stereo](figures/stereo_statistics.png)

| | voix | accompagnement |
|---|---|---|
| correlation L/R mediane | 0,94 | 0,84 |
| energie dans Side (mediane) | 3,0 % | 8,4 % |
| Side/Mid median | −15,1 dB | −10,4 dB |
| Side/Mid median entre 400 Hz et 12,8 kHz | −15 a −19 dB | −6 a −7 dB |
| morceaux quasi mono (correlation > 0,95) | 48 % | 12 % |

L'hypothese est confirmee en tendance : la voix est plus centree que l'accompagnement (correlation plus forte sur 72 % des morceaux), avec **un ecart de 9 a 12 dB de largeur stereo** entre 400 Hz et 12,8 kHz, la ou se trouve l'essentiel de la voix. En dessous de 100 Hz les deux sont centres (basse, grosse caisse), donc la stereo n'aide pas sur les graves. Mais ce n'est pas une regle : 52 % des voix ont une correlation inferieure a 0,95 (10e percentile de correlation a 0,64 : reverb stereo, doublages), et 12 % des accompagnements sont quasi mono. Le modele doit donc apprendre un indice utile mais bruite, pas appliquer un filtre « centre ».

l'ajout de ce second canal devrait permettre au modele d'utiliser cette information pour trouver des liens dans les frequences afin de mieux separer dans le masque.

Resultats sur les 50 morceaux de test (`uv run python -m src.evaluate CKPT -o outputs/stereo.json`, puis `python -m analysis.compare_runs outputs/oracle_baseline.json outputs/stereo.json --names mono stereo`) :

| | mono | stereo |
|---|---|---|
| SDR median (IC95 % bootstrap) | 4,70 dB [3,80 ; 5,92] | 4,50 dB [3,56 ; 5,38] |
| SI-SDRi median | 10,03 dB [8,80 ; 10,60] | 9,31 dB [8,41 ; 10,65] |
| SDR median oracle | 9,21 dB | 9,09 dB |
| ecart oracle − modele (SDR median) | 4,32 dB | 4,59 dB |

![mono vs stereo](figures/compare_mono_stereo_sans_exclus.png)

(figure sans les 2 morceaux *PR*, voir plus bas)

- **L'hypothese n'est pas confirmee : le stereo n'ameliore pas la separation, il est legerement moins bon.** Difference appariee stereo − mono : **−0,30 dB** de SDR median (IC95 % [−0,52 ; −0,09], Wilcoxon p = 0,019), et **−0,52 dB** de SI-SDRi median (IC95 % [−0,70 ; −0,31]). Le stereo n'est meilleur que sur 18 morceaux sur 50 en SDR (15 sur 50 en SI-SDRi).
- **L'effet est petit et fragile.** Sans les deux morceaux *PR* (voix quasi inaudible, SDR peu interpretable, *Happy Daze* passe de −16,4 a −27,7 dB et pese seul sur la moyenne), la difference mediane est de −0,26 dB avec un IC95 % [−0,46 ; +0,02] et p = 0,049 : on ne peut pas exclure un effet nul. Il n'y a aucun gain, mais pas non plus une perte nette.
- **L'ecart a l'oracle va dans le meme sens** : 4,59 dB en stereo contre 4,32 dB en mono, donc le stereo est un peu plus loin de son propre plafond.
- **Precautions avant de conclure :**
  - Une seule graine par configuration : on ne connait pas la variabilite entre deux entrainements identiques (voir `docs/ROADMAP.md`, etape 0). L'IC95 % ci-dessus ne mesure que la variabilite entre morceaux.
  - Les deux evaluations ne sont pas exactement le meme protocole : la baseline est evaluee en mono (SDR sur 1 canal), le modele stereo en stereo (SDR museval sur 2 canaux). L'oracle lui-meme change de 0,25 dB en median entre les deux (de −1,7 a +2,7 dB selon le morceau), donc une partie de la difference peut venir de la mesure et pas du modele. Il faudrait evaluer la sortie du modele stereo mixee en mono pour comparer a conditions egales (pas encore fait).
  - Ce n'est pas teste, mais le resultat est coherent avec le fait que le modele ne recoit que l'amplitude de L et de R : la difference de phase entre canaux est perdue, et l'indice « centre » (L ≈ R) doit etre retrouve par la premiere convolution (2 → 16 canaux) alors qu'un signal Mid/Side le donnerait directement. Piste a tester ensuite : entree Mid/Side a la place de L/R, ou un seul masque applique au Mid.

### 2. Changement de la loss