# SRGAN

Papier :  W. Shi, J. Caballero, F. Huszar, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert et Z. Wang, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, 2017
https://arxiv.org/abs/1609.04802

Code original : https://github.com/leftthomas/SRGAN

## Contributions

train.py :
- ajout des commentaires
- Utilisation de nouveaux dossiers pour les datasets : lignes 24-26 et 37-39
- Sauvegarde des résultats : 130-132
- modification de la taille du batch : ligne 42

test.py :
- ajout d'un paramètre pour choisir le dossier des images de test : lignes 31 et 40
- ajout de la sauvegarde de l'image HR : ligne 72

## Dépendances

- anaconda
- pytorch
- opencv

## Entraînement

Les données sont préparées directement dans train.py

```bash
python train.py --upscale_factor 2
                --num_epochs 300          
```

Les paramètres utilisés pour l'entraînement ou les tests ne sont pas forcément ceux par défaut dans le code

## Test

Test d'une image :

```bash
python test_image.py --upscale_factor 2 
                     --model_name "netG_2_t91.pth" 
                     --image_adress "...../datasets/Set14/" 
                     --image_name "img_014.png"
```
