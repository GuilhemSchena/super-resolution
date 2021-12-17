# SRCNN

Papier :  J. Kim, J. Kwon Lee, K. Mu Lee, Deeply-recursive convolutional network for image super-resolution, 2016
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.pdf

Code original : https://github.com/jiny2001/deeply-recursive-cnn-tf

## Contributions

train.py :
- ajout des commentaires
- Mise en place d'un entraînement par epoch : lignes 66, 137-139
- Sauvegarde des résultats : 130-132

test.py :
- ajout de mesure de temps : lignes 10, 82 et 87
- affichage des temps : lignes 89-90

models.py :
- ajout mesure de temps dans le forward (pour faire des tests) : 461, 473
- Sauvegarde des résultats (partie 2) : lignes 423-432

## Dépendances

- tensorflow 1.15
- scipy 1.1.0
- numpy
- pillow

## Entraînement

Les données sont préparées directement dans train.py

```bash
python main.py --data_dir "/content/drive/MyDrive/SYS843/Github/sr/datasets"
               --training_set "ScSR" 
               --stride_size 14 
               --batch_size 33 
               --scale 2 
               --nb_epochs 300 
               --batch_num 16              
```

## Test

Evaluation d'un set de test :

```bash
python main.py --is_training "False"
               --dataset "set5"
               --feature_num 96 
               --inference_depth 9 
               --scale 2
```
Test d'une image :

```bash
python test.py --file "..../img_011.png" 
               --feature_num 96 
               --inference_depth 9 
               --scale 2
```
