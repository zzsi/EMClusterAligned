close all;
disp(['================> Getting the exponential family model using negative images']);
%% read in negative examples 
imageName = dir('negativeImage/*.jpg');  % folder that contains negative examples as q(I)
numImageNeg = size(imageName, 1); % number of negative examples
imageSize = zeros(numImageNeg, 2); % size of negative examples
for i = 1 : numImageNeg
    tmpIm = imread(['negativeImage' '/' imageName(i).name]);
    if size(tmpIm,3) == 3
        tmpIm = rgb2gray(tmpIm);
    end
    I{i} = imresize(single(tmpIm), 1, 'nearest'); 
    imageSize(i, :) = size(I{i});
end
sizex = min(imageSize(:, 1)); sizey = min(imageSize(:, 2)); 
for i = 1 : numImageNeg
    I{i} = single(I{i}(1:sizex, 1:sizey)); 
end   % make the sizes of the training images to be the same
%% filtering negative examples
for s = 1 : numScale
    storeExponentialModelName = ['storedExponentialModel' num2str(s) '.mat'];
    if exist(storeExponentialModelName,'file')
        continue;
    end
    scaleFilter = scales(s); 
    localHalfx = floor(localHalfx1*scales(s)/scales(1));
    localHalfy = floor(localHalfy1*scales(s)/scales(1));
    [allFilter, allSymbol] = MakeFilter(scaleFilter, numOrient);  % generate Gabor filters 
    halfFilterSize = (size(allFilter{1}, 1)-1)/2;  % half size of Gabor
    disp(['start filtering negative images at Gabor length ' num2str(halfFilterSize*2+1)]);
    tic
    allFilteredImage = ApplyFilterfftSame(I, allFilter, localOrNot, localHalfx, localHalfy, doubleOrNot, thresholdFactor);  % filter training images
    disp(['filtering time: ' num2str(toc) ' seconds']);
    %% compute histogram of q()
    numBin = floor(saturation/binSize)+1;  % binnumbers
    histog = zeros(numBin, 1, 'single');  % store F  
    disp(['start histogramming negative images at Gabor length ' num2str(halfFilterSize*2+1)]);
    mexc_Histogram(numImageNeg, numOrient, allFilteredImage, halfFilterSize, sizex, sizey, binSize, numBin, histog, saturation);
    disp(['histogramming time: ' num2str(toc) ' seconds']);
    %% compute stored lambda, expectation, logZ
    r = (0:(numBin-1))*binSize;
    %{
    figure; h1=plot(r, histog);  
    title('background density'); xlabel('sigmoid(response)'); ylabel('density');
    set(gca,'FontSize',16); % adjust the display format and save the plot
    set(get(gca,'XLabel'),'FontSize',18);
    set(get(gca,'YLabel'),'FontSize',18);
    set(get(gca,'Title'),'FontSize',20);
    set(h1,'Linewidth',3);
    set(gcf, 'PaperPositionMode', 'auto');
    print('-depsc', ['output/backgroundDensity' num2str(s) '.eps']);
    %}

    storedExpectation = zeros(numStoredPoint, 1); 
    Z = zeros(numStoredPoint, 1); 
    for (k=1:numStoredPoint)
       lambda = (k-1.)*spacing; 
       p = exp(lambda*r).*(histog'); 
       Z(k) = sum(p*binSize); p = p/Z(k); 
       storedExpectation(k) = sum(r.*p*binSize);
    end
    storedlambda = (0:(numStoredPoint-1))*spacing; 
    storedLogZ = log(Z); 

	%{
    figure; h1=plot(storedlambda, storedExpectation);
    title('mean vs lambda'); xlabel('lambda'); ylabel('mean');
    set(gca,'FontSize',16); % adjust the display format and save the plot
    set(get(gca,'XLabel'),'FontSize',18);
    set(get(gca,'YLabel'),'FontSize',18);
    set(get(gca,'Title'),'FontSize',20);
    set(h1,'Linewidth',3);
    set(gcf, 'PaperPositionMode', 'auto');
    print('-depsc', ['output/backgroundLink' num2str(s) '.eps']);

    figure; h1=plot(storedlambda, storedLogZ);
    title('logZ vs lambda'); xlabel('lambda'); ylabel('logZ');
    set(gca,'FontSize',16); % adjust the display format and save the plot
    set(get(gca,'XLabel'),'FontSize',18);
    set(get(gca,'YLabel'),'FontSize',18);
    set(get(gca,'Title'),'FontSize',20);
    set(h1,'Linewidth',3);
    set(gcf, 'PaperPositionMode', 'auto');
    print('-depsc', ['output/backgroundLogZ' num2str(s) '.eps']);
    %}
    
    %% store filter properties and exponential model
    Correlation = CorrFilter(allFilter, epsilon);  % correlation between filters 
     
    save(storeExponentialModelName, 'storedlambda', 'storedExpectation', 'storedLogZ', ...
                    'scaleFilter', 'halfFilterSize', 'allFilter', 'allSymbol', 'Correlation', ...
                    'localHalfx', 'localHalfy','histog');
end


