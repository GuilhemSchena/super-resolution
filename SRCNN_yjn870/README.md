# SRCNN

Papier :  Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Learning a deep convolutional network for image super-resolution, 2014
https://arxiv.org/pdf/1501.00092.pdf

Code original : https://github.com/yjn870/SRCNN-pytorch

## Contributions

Travail principalement dans train.py :
- ajout des commentaires
- ajout sauvegarde des résultats : lignes 69-71 et 124-131
- remplacement de la ligne 37 par device = torch.device("cuda") suite à des problèmes avec Google Colab

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
               --image-file "data/butterfly_GT.bmp" \
               --scale 2
```
