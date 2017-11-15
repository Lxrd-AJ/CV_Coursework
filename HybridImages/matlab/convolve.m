function convolved = convolve(image,template)

[irows,icols]=size(image);
[trows,tcols]=size(template);
temp(1:irows,1:icols)=0;
trhalf=floor(trows/2); 
tchalf=floor(tcols/2); 

for x = trhalf+1:icols-trhalf 
  for y = tchalf+1:irows-tchalf
    sum=0;
    for iwin = 1:trows 
      for jwin = 1:tcols
        sum=sum+image(y+jwin-tchalf-1,x+iwin-trhalf-1)*template(jwin,iwin);
        temp(y,x)= sum;
      end
    end
  end
end

minim=min(min(temp));
range=max(max(temp))-minim;
convolved = floor( (temp - minim) * 255/range );
