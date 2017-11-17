function [red, green, blue] = convolution_channels( image, template )
win_size = size(template, 1);
pad = (win_size - 1) / 2;
% Pad the image to prepare it for convolution
image = padarray( image, [pad pad], 'both' );
red = convolve( double(image(:,:,1)), template );
green = convolve( double(image(:,:,2)), template );
blue = convolve( double(image(:,:,3)), template );
% Remove the padding as the convolution operation doesn't resize the image
red = red( pad+1:end-pad,pad+1:end-pad );
green = green( pad+1:end-pad,pad+1:end-pad );
blue = blue( pad+1:end-pad,pad+1:end-pad );


