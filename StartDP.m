clear
close all
% 
% %% mex-C compilation
% mex mexc_ComputeMAX1.cpp
% mex mexc_ComputeSUMMAX2.cpp
% mex mexc_Histogram.cpp	% pool histogram from negative images
% mex mexc_LocalNormalize.cpp	% local normalization of type single (float)
% mex mexc_SharedSketch.cpp	% learning by shared sketch algorithm (with data weights)
% mex mexc_TemplateAffineTransform.cpp
% mex mexc_CropInstanceNew.cpp

%% preparation

ParameterCodeImage;
% 
% ExponentialModel; close all
% storeExponentialModelName = ['storedExponentialModel' num2str(1)];   
% load(storeExponentialModelName);
% storedlambda = single(storedlambda);
% storedExpectation = single(storedExpectation);
% storedLogZ = single(storedLogZ);
% SUM1MAX1;
% 
% Correlation = CorrFilter(allFilter, epsilon);  % correlation between filters 
% for j = 1:numel(Correlation)
%     Correlation{j} = single(Correlation{j});
% end
% for j = 1:numel(allSymbol)
%     allSymbol{j} = single(allSymbol{j});
% end

% 
% % go over all the rectangles:
% for top = 0:rectStep:outerBBsize-rectStep
% 	for left = 0:rectStep:outerBBsize-rectStep
% 		for bottom = top+rectStep:rectStep:outerBBsize
% 			for right = left+rectStep:rectStep:outerBBsize
% 				height = bottom-top;
% 				width = right-left;
% 				templateSize = [height width];
% 				disp(sprintf('\nrect: top=%d left=%d height=%d width=%d',top,left,height,width));
% 				EMClusterPerRect;
% 			end
% 		end
% 	end
% end



% go over all the rectangles:
BestParse = [];
scoremap = zeros(5,5);
for top = 0:rectStep:outerBBsize-rectStep
	for left = 0:rectStep:outerBBsize-rectStep
		for bottom = top+rectStep:rectStep:outerBBsize
			for right = left+rectStep:rectStep:outerBBsize
				height = bottom-top;
				width = right-left;
				templateSize = [height width];
                model_name = sprintf('./output/learning_result%d_%d_%d_%d.mat', top, left, height, width);
                load(model_name, 'bestOverallScore');
                score = bestOverallScore;
%                 if score > 100000000
%                     123123123123
% %                     score = 0;
%                 end
                BestParse = [BestParse struct('top', top, 'left', left, 'height', height, 'width', width, ...
                    'score', score, 'ch1', 0, 'ch2', 0)];
                
                if height == rectStep && width == rectStep && score > 0
                    score;
%                     scoremap(top/rectStep+1, left/rectStep+1) = score
                end
			end
		end
	end
end

root = [0 0 200 200];

BestParse_org = BestParse;
% findRect(BestParse, root);

nCases = 20;
img_whole = [];
splitPenalty = -20.0;
tmp_whole = [];

for n=1:nCases
    BestParse = BestParse_org;

    for height = rectStep:rectStep:outerBBsize
        for width = rectStep:rectStep:outerBBsize
            if height == rectStep && width == rectStep, continue; end;

            templateSize = [height width];

            for top = 0:rectStep:outerBBsize-height
                for left = 0:rectStep:outerBBsize-width
                    [id, score] = findRect(BestParse, [top,left,height,width]);

                    split_list1 = [];
                    split_list2 = [];

                    % possible vertical splits
                    for split_width=rectStep:rectStep:width-rectStep
                        split_list1 = [split_list1 ; top left height split_width];
                        split_list2 = [split_list2 ; top left+split_width height width-split_width];
                    end

                    % possible horizontal splits
                    for split_height=rectStep:rectStep:height-rectStep
                        split_list1 = [split_list1 ; top left split_height width];
                        split_list2 = [split_list2 ; top+split_height left height-split_height width];
                    end

                    for k = 1:size(split_list1, 1)
                        [child_id1, score1] = findRect(BestParse, split_list1(k,:));
                        [child_id2, score2] = findRect(BestParse, split_list2(k,:));
                        if (score1 + score2 + splitPenalty > score)
                            score = score1 + score2;
                            BestParse(id).score = score;
                            BestParse(id).ch1 = child_id1;
                            BestParse(id).ch2 = child_id2;
                        end
                    end
                end
            end
        end
    end

    [root_id, score] = findRect(BestParse, root)

    selected_patches = [root_id];
    final_patches = {};
    while ~isempty(selected_patches)
        id = selected_patches(1);
        selected_patches = selected_patches(2:end);

        if BestParse(id).ch1 > 0
            selected_patches = [selected_patches BestParse(id).ch1 BestParse(id).ch2];
        else
            final_patches = [final_patches [id BestParse(id).top BestParse(id).left BestParse(id).height BestParse(id).width]];
        end
    end

    img = zeros(200, 200, 3);
    template_img = zeros(200, 200, 3);

    for i=1:length(final_patches)
        id = final_patches{i}(1);
        rr = rand;
        gg = rand;
        bb = rand;
        img(BestParse(id).top+1:BestParse(id).top+BestParse(id).height, BestParse(id).left+1:BestParse(id).left+BestParse(id).width, 1) = rr;
        img(BestParse(id).top+1:BestParse(id).top+BestParse(id).height, BestParse(id).left+1:BestParse(id).left+BestParse(id).width, 2) = gg;
        img(BestParse(id).top+1:BestParse(id).top+BestParse(id).height, BestParse(id).left+1:BestParse(id).left+BestParse(id).width, 3) = bb;

        model_name = sprintf('./output/learning_result%d_%d_%d_%d.mat', BestParse(id).top, BestParse(id).left, BestParse(id).height, BestParse(id).width);
        load(model_name);

        q = bestActivations(4,:);
        w = [];
        for j=1:length(bestS2Templates)
            w = [w sum(q==j-1)];
        end
        [q tid] = max(w);
        
        template_img(BestParse(id).top+1:BestParse(id).top+BestParse(id).height, BestParse(id).left+1:BestParse(id).left+BestParse(id).width, 1) = single(bestS2Templates{tid}.commonTemplate) / 256.0 * rr;
        template_img(BestParse(id).top+1:BestParse(id).top+BestParse(id).height, BestParse(id).left+1:BestParse(id).left+BestParse(id).width, 2) = single(bestS2Templates{tid}.commonTemplate) / 256.0 * gg;
        template_img(BestParse(id).top+1:BestParse(id).top+BestParse(id).height, BestParse(id).left+1:BestParse(id).left+BestParse(id).width, 3) = single(bestS2Templates{tid}.commonTemplate) / 256.0 * bb;
        
    end
        
    best_conf_file = sprintf('./bestConfigPenalty%f.mat', splitPenalty);
    save(best_conf_file, 'final_patches');
        
    img_whole = [img_whole zeros(200, 200, 3) img];
    tmp_whole = [tmp_whole zeros(200, 200, 3) template_img];
    
    splitPenalty = splitPenalty - 2.0;
end

imshow([img_whole ; tmp_whole]);
imwrite([img_whole ; tmp_whole], 'partition.png');

