%% Compute SUM1 and MAX1 maps
close all;
disp('===============> Getting the SUM1 and MAX1 maps for testing images');
sizeTemplatex = outerBBsize;
sizeTemplatey = outerBBsize;
for img = 1:numImage
    multipleResolutionImageName = ['working/multipleResolutionImage' num2str(img)];
    load(multipleResolutionImageName,'J'); 
    for s = 1 : numScale
        storeExponentialModelName = ['storedExponentialModel' num2str(s)];   
        load(storeExponentialModelName);
        %% Prepare spaces for variables
        allSizex = zeros(1, numResolution); allSizey = zeros(1, numResolution);
        trackMap = cell(numResolution, numOrient);
        SUM2map = cell(1, numResolution);
        
        MAX2score = single(zeros(1, numResolution));  % maximum log-likelihood score at each resolution
        Fx = zeros(1, numResolution); Fy = zeros(1, numResolution); % maximum likelihood location at each resolution
        %% Compute SUM1 maps
        % use non-inscribed images to compute SUM1 maps, to avoid boundary effects 
        disp(['start filtering image ' num2str(img) ' at Gabor length ' num2str(halfFilterSize*2+1) ' at all resolutions']); tic
        SUM1map = ApplyFilterfftSame(J, allFilter, localOrNot, localHalfx, localHalfy, doubleOrNot, thresholdFactor);  % filter testing image at all resolutions
        disp(['filtering time: ' num2str(toc) ' seconds']);
        
        %% inscribe the SUM1 maps into a square bounding box
        % change allSizex and allSizey
        for r = 1:size(SUM1map,1)
            for j = 1:size(SUM1map,2)
                currentOuterBBsize = max(size(SUM1map{r,j}));
                SUM1map{r,j} = single(inscribe([currentOuterBBsize currentOuterBBsize], SUM1map{r,j}, 0));
            end
            allSizex(r) = currentOuterBBsize;
            allSizey(r) = currentOuterBBsize;
        end
        %% also inscribe the source images
        for r = 1:size(SUM1map,1)
            currentOuterBBsize = max(size(J{r}));
            J{r} = single(inscribe([currentOuterBBsize currentOuterBBsize],J{r},255));
        end
        % don't forget to save
        multipleResolutionImageName = ['working/multipleResolutionImage' num2str(img)]; 
        save(multipleResolutionImageName, 'J');
        
        %% Still, prepare spaces for variables
        for r=1:numResolution
            [sizex, sizey] = size(SUM1map{r,1}); 
            allSizex(r) = sizex; allSizey(r) = sizey; % sizes of images at multiple resolutions
            sizexSubsample = floor(sizex/subsample); sizeySubsample = floor(sizey/subsample); 
            SUM2map{r} = single(zeros(floor((sizex+sizeTemplatex)/subsample), floor((sizey+sizeTemplatey)/subsample)));  % log-likelihood score maps for testing images
            averageMap{r} = single(zeros(sizex, sizey));
            for (orient=1:numOrient)
                MAX1map{r, orient} = single(zeros(sizexSubsample, sizeySubsample)); 
                trackMap{r, orient} = single(zeros(sizexSubsample, sizeySubsample));
            end
            translatedTemplate{r} = single(zeros(sizex, sizey));  % symbolic sketch on testing image
        end

        
        %% Compute MAX1 maps and track maps
        disp(['start maxing image ' num2str(img) ' at Gabor length ' num2str(halfFilterSize*2+1) ' at all resolutions']); tic
        subsampleM1 = 1;  % to be safe, please refrain from setting it to be larger than 1 
		[MAX1map ARGMAX1map M1RowShift M1ColShift M1OriShifted] = ...
			mexc_ComputeMAX1( numOrient, SUM1map, locationShiftLimit,... %single(locationShiftLimit)/(halfFilterSize*2+1),...
            orientShiftLimit, subsampleM1 );
        
        for ii = 1:numel(MAX1map)
            MAX1map{ii} = single( saturation*(2./(1.+exp(-2.*MAX1map{ii}/saturation))-1.) );
            SUM1map{ii} = single( saturation*(2./(1.+exp(-2.*SUM1map{ii}/saturation))-1.) );
        end
        
        disp(['maxing time: ' num2str(toc) ' seconds']);
        %% Save the maps and spaces
        SUM1MAX1mapName = ['working/SUM1MAX1map' 'image' num2str(img) 'scale' num2str(s)];   
        save(SUM1MAX1mapName, 'SUM1map', 'MAX1map', 'M1RowShift', 'M1ColShift',...
			'M1OriShifted', 'ARGMAX1map', 'averageMap', 'SUM2map', 'translatedTemplate',...
            'MAX2score', 'Fx', 'Fy', 'allSizex', 'allSizey', 'J');
    end
end
