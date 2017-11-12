sigma = 2.5;
win_size = round(8.0 * sigma + 1.0);
if (rem(win_size,2) == 0) 
    win_size = win_size + 1;
end
gaussian_temp = gaussian_template(win_size,sigma);

img1 = imread('data/dog.bmp','bmp');
img2 = imread('data/cat.bmp','bmp');

img_1_R = double(img1(:,:,1));
img_1_G = double(img1(:,:,2));
img_1_B = double(img1(:,:,3));

img_1_R = convolve(img_1_R,gaussian_temp);
img_1_G = convolve(img_1_G,gaussian_temp);
img_1_B = convolve(img_1_B,gaussian_temp);
img_1_low = cat(3, img_1_R, img_1_G, img_1_B );
figure(1), clf;
subplot(2,2,1), imshow(uint8(img_1_low));
subplot(2,2,2), imshow(uint8(img1));

sigma = 5.0;
win_size = round(8.0 * sigma + 1.0);
if (rem(win_size,2) == 0) 
    win_size = win_size + 1;
end
gaussian_temp = gaussian_template(win_size,sigma);
img_2_R = double(img2(:,:,1));
img_2_G = double(img2(:,:,2));
img_2_B = double(img2(:,:,3));

img_2_R = img_2_R - convolve(img_2_R,gaussian_temp);
img_2_G = img_2_G - convolve(img_2_G,gaussian_temp);
img_2_B = img_2_B - convolve(img_2_B,gaussian_temp);
img_2_high = cat(3, img_2_R, img_2_G, img_2_B ) * 1.1;
figure(2), clf;
subplot(2,2,1), imshow(uint8(img_2_high));
subplot(2,2,2), imshow(uint8(img2));

hybrid = uint8(img_1_low + img_2_high);
figure(3), clf;
subplot(3,3,1), imshow(hybrid);
subplot(3,3,2), imshow(img1);
subplot(3,3,3), imshow(img2);

imwrite(hybrid, 'hybrid.jpg');

% figure(4), clf
% h1 = ave(double(img1(:,:,1)),45);
% h1 = ave(h1,7);
% h1 = ave(h1,8);
% h1 = ave(h1,20);
% h1 = uint8(h1);
% subplot(2,2,1), imshow(h1);
% subplot(2,2,2), imshow(img1);
% sh = [h1,uint8(img1)];
% imshow(sh);
