clear
close all

%% mex-C compilation
mex mexc_ComputeMAX1.cpp
mex mexc_ComputeSUM2.cpp
mex mexc_Histogram.cpp	% pool histogram from negative images
mex mexc_LocalNormalize.cpp	% local normalization of type single (float)
mex mexc_SharedSketch.cpp	% learning by shared sketch algorithm (with data weights)
mex mexc_Sigmoid.cpp	% sigmoid transformation
mex mexc_ComputeMAX2.cpp
mex mexc_ComputeMAX2MP.cpp
mex mexc_TemplateAffineTransform.cpp
mex mexc_CropInstanceNew.cpp

%% preparation

ParameterCodeImage;
ExponentialModel; close all
storeExponentialModelName = ['storedExponentialModel' num2str(1)];   
load(storeExponentialModelName);
storedlambda = single(storedlambda);
storedExpectation = single(storedExpectation);
storedLogZ = single(storedLogZ);
% SUM1MAX1;

Correlation = CorrFilter(allFilter, epsilon);  % correlation between filters 
for j = 1:numel(Correlation)
    Correlation{j} = single(Correlation{j});
end
for j = 1:numel(allSymbol)
    allSymbol{j} = single(allSymbol{j});
end



%% begin EM clustering with window scanning in E step

% initialize: generate random SUM2 maps and thus random cluster members
mixing = zeros(numCluster,1); % number of examples in each cluster
aveLogL = zeros(numCluster,1); % average log likelihood in each cluster
activations = []; % 3 by N matrix, where N is an unknown large number
for i = 1:numImage
	% compute SUM2
	SUM1MAX1mapName = ['working/SUM1MAX1map' 'image' num2str(i) 'scale' num2str(1)];
	load(SUM1MAX1mapName, 'SUM1map');
	width = floor(size(SUM1map{1},1)/subsampleS2);
	height = floor(size(SUM1map{1},2)/subsampleS2);
	SUM2map = cell(nTransform*numCluster,1);
	for j = 1:nTransform*numCluster
		map = single( rand( height, width ) );
		SUM2map{j} = single( rand( height, width ) );
	end
	
	% compute MAX2, perform surround supression and get activations
	if strcmp('MatchingPursuit',supressionModeInEStep) == 1
		tmpActivations = mexc_ComputeMAX2MP( SUM2map, int32(locationPerturbationFraction*partSize/subsampleS2), -1000 );
	elseif strcmp('LocalSurroundSurpression',supressionModeInEStep) == 1
		subsampleM2 = 1;
		[MAX2map M2LocationTrace M2TemplateTrace M2RowColShift tmpActivations] = ...
			mexc_ComputeMAX2( templateAffinityMatrix, SUM2map, locationPerturbationFraction, ...
			int32(partSize/subsampleS2*ones(numCluster*nTransform,1)), subsampleM2 );
		tmpActivations = tmpActivations( :,tmpActivations(4,:) > -1000 );
	end
	activations = [activations,[tmpActivations;single(i*ones(1,size(tmpActivations,2)))]];
end
activations(1:2,:) = activations(1:2,:) * subsampleS2;
activatedCluster = ceil( ( activations(3,:) + 1 ) / nTransform );
activatedTransform = activations(3,:) + 1 - (activatedCluster-1) * nTransform;
activatedImg = activations(5,:);
disp(sprintf('on average %.2f activations per image',size(activations,2)/numImage));


for iter = 1:numIter

	%% M step
	syms = cell(numCluster,1);
    buf_length = 0;
	for cc = 1:numCluster
        for k = 1:buf_length
            fprintf(1,'\b');
        end
        str = sprintf('learning iteration %d for cluster %d',iter,cc);
        fprintf(1,str);
        buf_length = length(str);
        
		selectedOrient = zeros(1, numElement, 'single');  % orientation and location of selected Gabors
		selectedx = zeros(1, numElement, 'single'); 
		selectedy = zeros(1, numElement, 'single'); 
		selectedlambda = zeros(1, numElement, 'single'); % weighting parameter for scoring template matching
		selectedLogZ = zeros(1, numElement, 'single'); % normalizing constant
		commonTemplate = single(zeros(sizeTemplatex, sizeTemplatey)); % template of active basis 
		% =====================================
		% crop back and relearn
		% =====================================
		ind = find(activatedCluster == cc);
		mixing(cc) = length(ind);
		aveLogL(cc) = mean(activations(4,ind));
		% sample a subset of training postitives, if necessary
		if length(ind) > maxNumClusterMember
			idx = randperm(length(ind));
			ind = ind(idx(1:maxNumClusterMember));
			ind = sort(ind,'ascend');
		end
		nMember = length(ind);
		SUM1mapLearn = cell(nMember,numOrient);
		MAX1mapLearn = cell(nMember,numOrient);
		ARGMAX1mapLearn = cell(nMember,numOrient);
		cropped = cell(nMember,1);
		currentImg = -1;
		for iMember = 1:length(ind)
			if activatedImg(ind(iMember)) ~= currentImg
				currentImg = activatedImg(ind(iMember));
				SUM1MAX1mapName = ['working/SUM1MAX1map' 'image' num2str(currentImg) 'scale' num2str(1)];
				load(SUM1MAX1mapName, 'SUM1map', 'J' );
			end
			% use mex-C code instead: crop S1 map
			tScale = 0; destHeight = templateSize(1); destWidth = templateSize(2); nScale = 1; reflection = 1;
			SUM1mapLearn(iMember,:) = mexc_CropInstanceNew( SUM1map,...
				activations(1,ind(iMember))-1,...
				activations(2,ind(iMember))-1,...
				rotationRange(activatedTransform(ind(iMember))),tScale,reflection,destWidth,destHeight,numOrient,nScale );

			% Crop detected image patch for visualization
			srcIm = J{1};
			cropped(iMember) = mexc_CropInstanceNew( {single(srcIm)},...
				activations(1,ind(iMember))-1,...
				activations(2,ind(iMember))-1,...
				rotationRange(activatedTransform(ind(iMember))),0,reflection,destWidth,destHeight,1,1 );
			
			% local max
			subsampleM1 = 1;
			[M1 ARGMAX1 M1RowShift M1ColShift M1OriShifted] = ...
				mexc_ComputeMAX1( 16, SUM1mapLearn(iMember,:), locationShiftLimit,...
					orientShiftLimit, subsampleM1 );
			MAX1mapLearn(iMember,:) = M1;
			ARGMAX1mapLearn(iMember,:) = ARGMAX1;
        end
        im = displayImages(cropped,10,60,60);
        if ~isempty(im)
            imwrite(im,sprintf('output/cluter%d_iter%d.png',cc,iter));
        end
			
        % now start re-learning
        commonTemplate = single(zeros(templateSize(1), templateSize(2)));  
        deformedTemplate = cell(1, nMember); % templates for training images 
        for ii = 1 : nMember
            deformedTemplate{ii} = single(zeros(templateSize(1), templateSize(2)));  
        end
        mexc_SharedSketch(numOrient, locationShiftLimit, orientShiftLimit, subsampleM1, ... % about active basis  
           numElement, nMember, templateSize(1), templateSize(2), ...
           SUM1mapLearn, MAX1mapLearn, ARGMAX1mapLearn, ... % about training images
           halfFilterSize, Correlation, allSymbol(1, :), ... % about filters
           numStoredPoint, storedlambda, storedExpectation, storedLogZ, ... % about exponential model 
           selectedOrient, selectedx, selectedy, selectedlambda, selectedLogZ, ... % learned parameters
           commonTemplate, deformedTemplate, ... % learned templates 
           M1RowShift, M1ColShift, M1OriShifted); % local shift parameters

		save(sprintf('working/learnedmodel%d_iter%d.mat',cc,iter), 'numElement', 'selectedOrient',...
            'selectedx', 'selectedy', 'selectedlambda', 'selectedLogZ',...
            'commonTemplate'...
        );

        syms{cc} = -commonTemplate;
	end
	
	towrite = displayImages(syms,10,templateSize(1),templateSize(2));
	imwrite(towrite,sprintf('output/template_iter%d.png',iter));
    fprintf(1,'\n');
	
	% ==============================================
	%% E step
	% ==============================================
	
	% transform the templates
	S2Templates = cell(numCluster,1);
	for cc = 1:numCluster
		load(sprintf('working/learnedmodel%d_iter%d.mat',cc,iter), 'numElement', 'selectedOrient', 'selectedx', 'selectedy', 'selectedlambda', 'selectedLogZ');
		S2Templates{cc} = struct( 'selectedRow', single(selectedx -1 - floor(templateSize(1)/2)),...
			'selectedCol', single(selectedy -1 - floor(templateSize(2)/2)), ...
			'selectedOri', single(selectedOrient), 'selectedScale', zeros(length(selectedx),1,'single'), ...
			'selectedLambda', single(selectedlambda), 'selectedLogZ', single(selectedLogZ) );
	end
	TransformedTemplate = cell(nTransform,numCluster);
	selectedScale = zeros(1,length(selectedx),'single');
	for cc = 1:numCluster
		for iT = 1:nTransform
			templateScaleInd = templateTransform{iT}(1);
			rowScale = templateTransform{iT}(2);
			colScale = templateTransform{iT}(3);
			rotation = templateTransform{iT}(4);
			[tmpSelectedRow tmpSelectedCol tmpSelectedOri tmpSelectedScale] = ...
				mexc_TemplateAffineTransform( templateScaleInd, rowScale,...
				colScale, rotation, S2Templates{cc}.selectedRow, S2Templates{cc}.selectedCol,...
				S2Templates{cc}.selectedOri, selectedScale, numOrient );
			TransformedTemplate{iT,cc}.selectedRow = tmpSelectedRow;
			TransformedTemplate{iT,cc}.selectedCol = tmpSelectedCol;
			TransformedTemplate{iT,cc}.selectedOri = tmpSelectedOri;
			TransformedTemplate{iT,cc}.selectedScale = tmpSelectedScale;
			TransformedTemplate{iT,cc}.selectedLambda = S2Templates{cc}.selectedLambda;
			TransformedTemplate{iT,cc}.selectedLogZ = S2Templates{cc}.selectedLogZ;
		end
	end

	% prepare for affinity matrix
	templateAffinityMatrix = cell( numCluster * nTransform, 1 );
	for cc = 1:numCluster
		% from = (cc-1)*nTransform + 1;
		% to = cc * nTransform;
        from = 1; to = nTransform*numCluster;
		for jj = 1:nTransform
			templateAffinityMatrix{jj+(cc-1)*nTransform} = int32((from:to)-1);
		end
	end
	
	activations = []; % 3 by N matrix, where N is an unknown large number
	for i = 1:numImage
		% compute SUM2
		SUM1MAX1mapName = ['working/SUM1MAX1map' 'image' num2str(i) 'scale' num2str(1)];
		load(SUM1MAX1mapName, 'SUM1map', 'MAX1map', 'M1RowShift', 'M1ColShift',...
					'M1OriShifted', 'J');
		SUM2map = mexc_ComputeSUM2( numOrient, MAX1map, TransformedTemplate, subsampleS2 );
        % random perturbation (to break ties arbitrarily for MAX2)
        for ii = 1:numel(SUM2map)
            SUM2map{ii}(:) = SUM2map{ii}(:) + 1e-3 * ( rand(numel(SUM2map{ii}),1) - .5 );
        end
		
		% compute MAX2, perform surround supression and get activations
        if iter == 1
			if strcmp('MatchingPursuit',supressionModeInEStep) == 1
				tmpActivations = mexc_ComputeMAX2MP( SUM2map, int32(locationPerturbationFraction*partSize/subsampleS2), -1000 );
			elseif strcmp('LocalSurroundSurpression',supressionModeInEStep) == 1
				subsampleM2 = 1;
				[MAX2map M2LocationTrace M2TemplateTrace M2RowColShift tmpActivations] = ...
					mexc_ComputeMAX2( templateAffinityMatrix, SUM2map, locationPerturbationFraction, ...
					int32(partSize/subsampleS2*ones(numCluster*nTransform,1)), subsampleM2 );
				tmpActivations = tmpActivations( :,tmpActivations(4,:) > -1000 );
			end
        else
            % discard the activated instances that have a low S2 score
            if iter > floor(numIter/2) % for the later iterations, increase the sparsity
            	locationPerturbationFraction = .5;
            end
			if strcmp('MatchingPursuit',supressionModeInEStep) == 1
				tmpActivations = mexc_ComputeMAX2MP( SUM2map, int32(locationPerturbationFraction*partSize/subsampleS2), S2Thres );
			elseif strcmp('LocalSurroundSurpression',supressionModeInEStep) == 1
				subsampleM2 = 1;
				[MAX2map M2LocationTrace M2TemplateTrace M2RowColShift tmpActivations] = ...
					mexc_ComputeMAX2( templateAffinityMatrix, SUM2map, locationPerturbationFraction, ...
					int32(partSize/subsampleS2*ones(numCluster*nTransform,1)), subsampleM2 );
				tmpActivations = tmpActivations( :,tmpActivations(4,:) > S2Thres );
			end
        end
		activations = [activations,[tmpActivations;single(i*ones(1,size(tmpActivations,2)))]];
        % keyboard
	end
	activations(1:2,:) = activations(1:2,:) * subsampleS2;
	activatedCluster = ceil( ( activations(3,:) + 1 ) / nTransform );
	activatedTransform = activations(3,:) + 1 - (activatedCluster-1) * nTransform;
	activatedImg = activations(5,:);
    disp(sprintf('on average %.2f activations per image',size(activations,2)/numImage));
end

save( 'activations.mat', 'activations' );
disp('\n');


% rank the learned templates
% mixing = mixing/sum(mixing); % no need to normalize

[sorted idx] = sort( sqrt(mixing) .* aveLogL, 'descend' );
for i = 1:numel(syms)
	towrite = syms{i};
	if range(towrite) < 1
		towrite = 255;
	else
		towrite = uint8(255 * (towrite-min(towrite(:)))/(max(towrite(:))-min(towrite(:))));
		towrite = double(towrite) - 50;
	end
	syms{i} = towrite;
end
towrite = displayImages( syms(idx([1:end])), 10, templateSize(1), templateSize(2), false );
imwrite(towrite,sprintf('template_sorted.png'));



