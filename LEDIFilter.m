function F = LEDIFilter(I, r)
    %---------------------------------------------------------------------%
    % Input:
    %           I: input image
    %           r: determing size of local filter and radius of structuring element
    % Output:
    %           F: filtered image
    %---------------------------------------------------------------------%
    
    % local filter template
    sigma = 2 * r + 1;
    h = ones(sigma);
    h(2 : end - 1, 2 : end - 1) = 0;
    h = double(h / sum(h(:)));
    
    % structuring element
    SE = strel('disk', r);
    
    % filtering image locally
    F1 = imerode(I, SE);
    F2 = imfilter(F1, h, 'symmetric');
    F3 = imdilate(F2, SE);
    F = imfilter(F3, h, 'symmetric');