% - EMClusterPerRect: learns mixture of active basis models for the given rectangular region.
% 
%	The templates are allowed to transform locally.

% need the variable input <rect>: [top left height width]

numRandomStart = 1;
bestOverallScore = -inf;
for iRS = 1:numRandomStart

	%% Initial E step: random assignment
	activations = zeros(5,numImage,'single'); % one activation is a vector: [imgNo row col templateNo score]
	activations(1,:) = 1:numImage;
	activaitons(2,:) = top + floor(height/2) - 1;
	activations(3,:) = left + floor(width/2) - 1;
	activatedClusters = floor( numCluster * rand(1,numImage) ); % starts from 0
	activations(4,:) = activatedClusters * nTransform + NO_TRANSFORM;

	for it = 1:numEMIter

		%% M step
		% crop back SUM1 maps
		buf_length = 0;
		for cc = 1:numCluster
			for k = 1:buf_length
				fprintf(1,'\b');
			end
			str = sprintf('learning iteration %d for cluster %d',iter,cc);
			fprintf(1,str);
			buf_length = length(str);
		end
		
		%% E step

	end
	
	if overallScore > bestOverallScore
		bestOverallScore = overallScore;
		bestTemplates = S2Template;
		bestInitialClusters = activations;
	end
end
