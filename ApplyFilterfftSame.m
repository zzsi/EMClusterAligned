function [allFiltered] = ApplyFilterfftSame(I, allFilter, localOrNot, localHalfx, localHalfy, doubleOrNot, thresholdFactor)
% filter images by a bank of filters
% I: input images
% allFilter: filter bank
numImage = size(I, 2); % number of training images
numOrient = size(allFilter, 2);  % number of orientations
h = (size(allFilter{1}, 1)-1)/2;  % half size of filters
[sx, sy] = size(I{1});  % size of images
allFiltered = cell(numImage, numOrient);  % filtered images
for (i = 1 : numImage)
   fftI{i} = fft2(I{i}, sx+h+h, sy+h+h);
end
for  (o = 1 : numOrient)
   fftf{o} = fft2(allFilter{1, o}, sx+h+h, sy+h+h); 
end

for i = 1 : numImage
   %tot = 0.;
   fftIi = fftI{i}; 
   for o = 1 : numOrient
      fftfo = fftf{o}; 
      out = ifft2(fftIi.*fftfo); 
      filtered = out(h+1:h+sx, h+1:h+sy); 
      re = real(filtered); im = imag(filtered); 
      energy = re.*re + im.*im;  % compute the local energy
      allFiltered{i, o} = single(energy); 
   end
   %ave = tot/numOrient; 
   %for o = 1 : numOrient
   %    allFiltered{i, o} = allFiltered{i, o}/ave;  % normalizing by whitening
   %end
   if ((localOrNot>0)&&(doubleOrNot>0))
       mexc_LocalNormalizeDouble(sx, sy, numOrient, h, localHalfx, localHalfy, allFiltered(i, :), thresholdFactor); 
   end
   if ((localOrNot>0)&&(doubleOrNot<0))
       mexc_LocalNormalize(sx, sy, numOrient, h, localHalfx, localHalfy, allFiltered(i, :), thresholdFactor); 
   end
   for o = 1:numOrient
       allFiltered{i,o}([1:h sx-h:sx],:) = 0;
       allFiltered{i,o}(:,[1:h sy-h:sy]) = 0;
   end
end
