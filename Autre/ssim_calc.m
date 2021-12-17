nom = "img_001";

path = "C:\Users\guilh\OneDrive\Documents\Travail\ETS\SYS843\Projet\2_Expérimentation\resultats\DRCN\D9F96_pretrained\"

img0 = imread(path + nom + ".png");
imgbc = imread(path + nom + "_bicubic.png");
imgsr = imread(path + nom + "_result.png");

figure('Name', '0')
imshow(img0)

figure('Name', 'sr')
imshow(imgsr)

figure('Name', 'bc')
imshow(imgbc)

ssim_val_bc = ssim(img0, imgbc)
pnsr_val_bc = psnr(img0, imgbc)

ssim_val_sr = ssim(img0, imgsr)
pnsr_val_sr = psnr(img0, imgsr)
