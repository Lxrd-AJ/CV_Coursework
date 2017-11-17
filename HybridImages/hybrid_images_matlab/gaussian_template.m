function template = gaussian_template(winsize,sigma)
centre=floor(winsize/2)+1;
sum=0;

for i=1:winsize
  for j=1:winsize
    template(j,i)=exp(-(((j-centre)*(j-centre))+((i-centre)*(i-centre)))/(2*sigma*sigma));
    sum=sum+template(j,i);
  end
end

template=template/sum;
