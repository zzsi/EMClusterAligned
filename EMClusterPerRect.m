% - EMClusterPerRect: learns mixture of active basis models for the given rectangular region.
% 
%	The templates are allowed to transform locally.

% need the variable input <rect>: [top left height width]

numRandomStart = 1;
bestOverallScore = -inf;

buf_length = 0;

%% JS - for empty clusters
emptyCluster = zeros(1, numCluster);

for iRS = 1:numRandomStart

	mixing = zeros(numCluster,1); % number of examples in each cluster
	aveLogL = zeros(numCluster,1); % average log likelihood in each cluster

	%% Initial E step: random assignment
	activations = zeros(5,numImage,'single'); % one activation is a vector: [imgNo row col templateNo score]
	activations(1,:) = 1:numImage;
	activatedImg = activations(1,:);
	activations(2,:) = top + floor(height/2) - 1; % starts from 0
	activations(3,:) = left + floor(width/2) - 1; % starts from 0

    activatedCluster = ceil( numCluster * rand(1,numImage) ); % starts from 1
    activatedCluster = imageClass;
    
	activations(4,:) = (activatedCluster-1) * nTransform + NO_TRANSFORM; % starts from 0
	activatedTransform = activations(4,:) + 1 - (activatedCluster-1) * nTransform; % starts from 1
    initialClusters = activations;

	for iter = 1:numEMIter
		% ==============================================================================
		%% M step
		% ==============================================================================
		% crop back SUM1 maps
		
		for cc = 1:numCluster
            if emptyCluster(cc)>0, continue, end;

% 			for k = 1:buf_length
% 				fprintf(1,'\b');
% 			end
			str = sprintf('run %d: learning iteration %d for cluster %d\n',iRS,iter,cc);
			fprintf(1,str);
			drawnow;
			buf_length = length(str);
			
			selectedOrient = zeros(1, numElement, 'single');  % orientation and location of selected Gabors
			selectedx = zeros(1, numElement, 'single'); 
			selectedy = zeros(1, numElement, 'single'); 
			selectedlambda = zeros(1, numElement, 'single'); % weighting parameter for scoring template matching
			selectedLogZ = zeros(1, numElement, 'single'); % normalizing constant
			commonTemplate = single(zeros(templateSize)); % template of active basis 
			
			ind = find(activatedCluster == cc);
			mixing(cc) = length(ind);
			aveLogL(cc) = mean(activations(4,ind));
			
			% sample a subset of training postitives, if necessary
			if length(ind) > maxNumClusterMember
				idx = randperm(length(ind));
				ind = ind(idx(1:maxNumClusterMember));
				ind = sort(ind,'ascend'); % make sure the imageNo is still in ascending order
			end
			nMember = length(ind);
			
			if nMember == 0
                emptyCluster(cc) = 1;
				disp('empty cluster\n');
				continue;
			end
			
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
					activations(2,ind(iMember))-1,...
					activations(3,ind(iMember))-1,...
					rotationRange(activatedTransform(ind(iMember))),tScale,reflection,destWidth,destHeight,numOrient,nScale );

				% Crop detected image patch for visualization
				srcIm = J{1};
				cropped(iMember) = mexc_CropInstanceNew( {single(srcIm)},...
					activations(2,ind(iMember))-1,...
					activations(3,ind(iMember))-1,...
					rotationRange(activatedTransform(ind(iMember))),0,reflection,destWidth,destHeight,1,1 );
				
				% local max
				subsampleM1 = 1;
				[M1 ARGMAX1 M1RowShift M1ColShift M1OriShifted] = ...
					mexc_ComputeMAX1( 16, SUM1mapLearn(iMember,:), locationShiftLimit,...
						orientShiftLimit, subsampleM1 );
				MAX1mapLearn(iMember,:) = M1;
				ARGMAX1mapLearn(iMember,:) = ARGMAX1;
			end
			im = displayImages(cropped,10,templateSize(1),templateSize(2));

            imwrite(im,sprintf('output/cluter%d_iter%d.png',cc,iter));
% 			if ~isempty(im)
% 				imwrite(im,sprintf('output/cluter%d_iter%d.png',cc,iter));
% 			else
% 				continue;
% 			end
			
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
			   numStoredPoint, single(storedlambda), single(storedExpectation), single(storedLogZ), ... % about exponential model 
			   selectedOrient, selectedx, selectedy, selectedlambda, selectedLogZ, ... % learned parameters
			   commonTemplate, deformedTemplate, ... % learned templates 
			   M1RowShift, M1ColShift, M1OriShifted); % local shift parameters
            
            nn = sum(selectedlambda > S1Thres);
            nn = max(nn,1);
            selectedx = selectedx(1:nn);
            selectedy = selectedy(1:nn);
            selectedOrient = selectedOrient(1:nn);
            selectedlambda = selectedlambda(1:nn);
            selectedLogZ = selectedLogZ(1:nn);
            commonTemplate = displayMatchedTemplate(templateSize,...
                selectedx,selectedy,selectedOrient,zeros(nn,1),...
                selectedlambda,allSymbol,numOrient);

           
			save(sprintf('working/learnedmodel%d_iter%d.mat',cc,iter), 'numElement', 'selectedOrient',...
				'selectedx', 'selectedy', 'selectedlambda', 'selectedLogZ',...
				'commonTemplate'...
			);

			syms{cc} = -single(commonTemplate);
		end
		towrite = displayImages(syms,10,templateSize(1),templateSize(2));
		imwrite(towrite,sprintf('output/template_iter%d.png',iter));
		
		% ==============================================================================
		%% E step
		% ==============================================================================
		% transform the templates
		S2Templates = cell(numCluster,1);
		for cc = 1:numCluster
            if emptyCluster(cc)>0, continue, end;
            
			load(sprintf('working/learnedmodel%d_iter%d.mat',cc,iter), 'numElement', 'selectedOrient', 'selectedx', 'selectedy', 'selectedlambda', 'selectedLogZ', 'commonTemplate');
			S2Templates{cc} = struct( 'selectedRow', single(selectedx -1 - floor(templateSize(1)/2)),...
				'selectedCol', single(selectedy -1 - floor(templateSize(2)/2)), ...
				'selectedOri', single(selectedOrient), 'selectedScale', zeros(length(selectedx),1,'single'), ...
				'selectedLambda', single(selectedlambda), 'selectedLogZ', single(selectedLogZ), 'commonTemplate', commonTemplate );
		end
		TransformedTemplate = cell(nTransform,numCluster);
		
		for cc = 1:numCluster
            %% JS - for empty clusters
            if emptyCluster(cc)>0
                TransformedTemplate{iT,cc} = struct( 'selectedRow', [],...
                    'selectedCol', [], ...
                    'selectedOri', [], 'selectedScale', [], ...
                    'selectedLambda', [], 'selectedLogZ', [], 'commonTemplate', [] );
                continue;
            end;
            
            selectedScale = zeros(1,length(S2Templates{cc}.selectedRow),'single');
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
		
		activations = []; % 5 by N matrix
		for i = 1:numImage
			% compute SUM2 and find local maximum (MAX2)
			SUM1MAX1mapName = ['working/SUM1MAX1map' 'image' num2str(i) 'scale' num2str(1)];
			load(SUM1MAX1mapName, 'MAX1map');
			activation = mexc_ComputeSUMMAX2FixCluster( numOrient, MAX1map, TransformedTemplate, subsampleS2, ...
				locationPerturbationFraction, ...
				int32(templateSize), int32([top+height/2,left+width/2]), imageClass(i) );
			% discard the activated instances that have a low S2 score
			activations = [activations,[i;activation]];
		end
		activatedImg = activations(1,:);
		activatedCluster = ceil( ( activations(4,:) + 1 ) / nTransform ); % starts from 1
		activatedTransform = activations(4,:) + 1 - (activatedCluster-1) * nTransform; % starts from 1
	end
	
	% compute overall Score
	scores = activations(5,:);
	scores = scores(scores>S2Thres);
	overallScore = mean(scores);
	
	if overallScore > bestOverallScore
		bestOverallScore = overallScore;
		bestS2Templates = S2Templates;
		bestInitialClusters = initialClusters;
		bestActivations = activations;
	end
end

% -- now we have selected the best random starting point --
%% display the templates and cluster members for the best random starting point
activations = bestActivations;
S2Templates = bestS2Templates;
activatedImg = activations(1,:);
activatedCluster = ceil( ( activations(4,:) + 1 ) / nTransform ); % starts from 1
activatedTransform = activations(4,:) + 1 - (activatedCluster-1) * nTransform; % starts from 1
for cc = 1:numCluster
    if emptyCluster(cc)>0, continue, end;
    
	ind = find(activatedCluster == cc);
	% sample a subset of training postitives, if necessary
	if length(ind) > maxNumClusterMember
		idx = randperm(length(ind));
		ind = ind(idx(1:maxNumClusterMember));
		ind = sort(ind,'ascend'); % make sure the imageNo is still in ascending order
	end
	
	nMember = length(ind);
	cropped = cell(nMember,1);
	currentImg = -1;
	for iMember = 1:length(ind)
		if activatedImg(ind(iMember)) ~= currentImg
			currentImg = activatedImg(ind(iMember));
			SUM1MAX1mapName = ['working/SUM1MAX1map' 'image' num2str(currentImg) 'scale' num2str(1)];
			load(SUM1MAX1mapName, 'J' );
		end
		% use mex-C code instead: crop S1 map
		tScale = 0; destHeight = templateSize(1); destWidth = templateSize(2); nScale = 1; reflection = 1;

		% Crop detected image patch for visualization
		srcIm = J{1};
		cropped(iMember) = mexc_CropInstanceNew( {single(srcIm)},...
			activations(2,ind(iMember))-1,...
			activations(3,ind(iMember))-1,...
			rotationRange(activatedTransform(ind(iMember))),0,reflection,destWidth,destHeight,1,1 );		
	end
	im = displayImages(cropped,10,templateSize(1),templateSize(2));
	if ~isempty(im)
		imwrite(im,sprintf('output/%d_%d_%d_%d_cluster%d.png',top,left,height,width,cc));
	end
	
	syms{cc} = -single(S2Templates{cc}.commonTemplate);
end
towrite = displayImages(syms,10,templateSize(1),templateSize(2));
imwrite(towrite,sprintf('output/%d_%d_%d_%d_template.png',top,left,height,width));
save(sprintf('output/learning_result%d_%d_%d_%d.mat',top,left,height,width),'bestActivations','bestS2Templates','bestOverallScore','bestInitialClusters');


