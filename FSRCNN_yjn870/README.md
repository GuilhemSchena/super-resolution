# SRCNN

Papier :  Chao Dong, Chen Change Loy, Xiaoou Tang, Accelerating the super-resolution convolutional neural network, 2016
https://arxiv.org/abs/1608.00367

Code original : https://github.com/yjn870/FSRCNN-pytorch

## Contributions

train.py :
- ajout des commentaires
- ajout sauvegarde des résultats : lignes 70-72 et 124-131

test.py :
- ajout de mesure de temps : lignes 48, 50 et 59
- affichage des temps : lignes 64-65

models.py :
- ajout mesure de temps dans le forward (pour faire des tests) : lignes 26, 42, 46 et 47

## Dépendances

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Entraînement

Préparer les données avant avec prepare.py :

```bash
python prepare.py --images-dir "/dossier_avec_les_images" \
                  --output-path "/dossier_avec_le_images_préparées" \
                  --eval "False" \
```

```bash
python train.py --train-file "dossier_avec_le_images_préparées/91-image_x2.h5" \
                --eval-file "dossier_avec_le_images_préparées/Set5_x2.h5" \
                --outputs-dir "/outputs" \
                --scale 2 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 300 \
                --num-workers 8 \
                --seed 123                
```

## Test

```bash
python test.py --weights-file "weights/91-image_x2_weights/x2/best.pth" \
               --image-file "data/Set5/butterfly.png" \
               --scale 2
```
