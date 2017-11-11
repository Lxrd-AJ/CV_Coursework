sigma = 2.5;
win_size = round(8.0 * sigma + 1.0);
if (rem(win_size,2) == 0) 
    win_size = win_size + 1;
end
gaussian_temp = gaussian_template(win_size,sigma);

img1 = imread('data/bird.bmp','bmp');
img2 = imread('data/plane.bmp','bmp');

img_1_R = double(img1(:,:,1));
img_1_G = double(img1(:,:,2));
img_1_B = double(img1(:,:,3));

img_1_R = convolve(img_1_R,gaussian_temp);
img_1_G = convolve(img_1_G,gaussian_temp);
img_1_B = convolve(img_1_B,gaussian_temp);
img_1_low = cat(3, img_1_R, img_1_G, img_1_B );
figure(1)
imshow(uint8(img_1_low));

sigma = 6.0;
win_size = 7;
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
img_2_high = cat(3, img_2_R, img_2_G, img_2_B ) + 0.5;
figure(2)
imshow(uint8(img_2_high));

hybrid = img_1_low + img_2_high;
figure(3)
imshow(uint8(hybrid));




















% Show img1 
% figure(1), clf,
% imshow(img1, [0 255]);
% % Show img2
% mini=min(min(img2));
% maxi=max(max(img2));
% figure(2), clf,
% imagesc(img2, [mini maxi])
% 
% 
% figure(3), clf,
% imagesc(low_freq);
% 
% high_freq = convolve(img2, gaussian_temp);
% high_freq = img2 - high_freq;
% figure(4), clf,
% imagesc(high_freq);
% 
% hybrid = low_freq + high_freq;
% figure(5), clf,
% % imagesc(hybrid);
% imshow(hybrid, [0 255]);