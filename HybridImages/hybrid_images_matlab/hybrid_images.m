sigma = 4.5;
win_size = round(8.0 * sigma + 1.0);
if (rem(win_size,2) == 0) 
    win_size = win_size + 1;
end
pad = (win_size - 1) / 2;
gaussian_temp = gaussian_template(win_size,sigma);

img1 = imread('data/einstein.bmp','bmp');
img2 = imread('data/marilyn.bmp','bmp');

[img_1_R, img_1_G, img_1_B] = convolution_channels( img1, gaussian_temp );
img_1_low = cat(3, img_1_R, img_1_G, img_1_B );

figure(1), clf;
subplot(2,2,1), imshow(uint8(img_1_low));
subplot(2,2,2), imshow(uint8(img1));


[img_2_R, img_2_G, img_2_B] = convolution_channels( img2, gaussian_temp );
img_2_high = cat(3, img_2_R, img_2_G, img_2_B );
img_2_high = double(img2) - img_2_high;

figure(2), clf;
subplot(2,2,1), imshow(uint8(img_2_high));
subplot(2,2,2), imshow(uint8(img2));

hybrid = uint8(img_1_low + img_2_high);
figure(3), clf;
subplot(3,3,1), imshow(hybrid);
subplot(3,3,2), imshow(img1);
subplot(3,3,3), imshow(img2);

% Resize the image for visualisation purposes
hybrid_bigger = imresize( hybrid, 1.5 );
hybrid_smaller = imresize( hybrid, 0.5 );
figure(4), clf;
subplot(2,2,1), imshow(hybrid_smaller);
subplot(2,2,2), imshow(hybrid_bigger);

imwrite(hybrid, 'hybrid.jpg');
% 
% Throw-away Code to resize some images
% img1 = imresize( img1, [340 400] );
% img1 = uint8(img1);
% img2 = imresize( img2, [340 400] );
% img2 = uint8(img2);
% img = imread('trump_hilary.jpg','jpg');
% img = imresize( img, 0.5 );
% imwrite( img, 'th_half.jpg' );
% img = imresize( img, 0.5 );
% imwrite( img, 'th_half_half.jpg' );
% 
% 
