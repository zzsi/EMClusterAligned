clear
close all

%% mex-C compilation
mex mexc_ComputeMAX1.cpp
mex mexc_ComputeSUMMAX2.cpp
mex mexc_Histogram.cpp	% pool histogram from negative images
mex mexc_LocalNormalize.cpp	% local normalization of type single (float)
mex mexc_SharedSketch.cpp	% learning by shared sketch algorithm (with data weights)
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
SUM1MAX1;

Correlation = CorrFilter(allFilter, epsilon);  % correlation between filters 
for j = 1:numel(Correlation)
    Correlation{j} = single(Correlation{j});
end
for j = 1:numel(allSymbol)
    allSymbol{j} = single(allSymbol{j});
end


% go over all the rectangles:
for top = 0:rectStep:outerBBsize-rectStep
	for left = 0:rectStep:outerBBsize-rectStep
		for bottom = top+rectStep:rectStep:outerBBsize
			for right = left+rectStep:rectStep:outerBBsize
				height = bottom-top;
				width = right-left;
				templateSize = [height width];
				disp(sprintf('\nrect: top=%d left=%d height=%d width=%d',top,left,height,width));
				EMClusterPerRect;
			end
		end
	end
end





