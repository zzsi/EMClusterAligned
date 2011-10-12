function [id, score] = findRect(list, top, left, height, width)
id = 0;
score = 0;

if nargin < 3
    width = top(4);
    height = top(3);
    left = top(2);
    top = top(1);
end

for i=1:length(list)
    if list(i).top == top && list(i).left == left && list(i).height == height && list(i).width == width
        id = i;
        score = list(i).score;
        return;
    end
end

end
