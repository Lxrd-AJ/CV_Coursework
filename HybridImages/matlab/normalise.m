% function normalised = normalise(image)
% [rows,cols]=size(image); 
% 
% %set minimum
% minim=min(min(image));
% 
% %work out range of input levels
% range=max(max(image))-minim;
% 
% %normalise the image
% % for x = 1:cols %address all columns
% %   for y = 1:rows %address all rows
% %     normalised(y,x)=floor((image(y,x)-minim)*255/range);
% %   end
% % end
% 
% normalised = floor( (image - minim) * 255/range );