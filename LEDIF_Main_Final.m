clc, clear, close all;

% dataset ID:  1. PET-MRI, 2. SPECT-MRI, 3. CT-MRI, 4. Infrared-Visual, 5. Multi-focus images
dataID = 4;     % select dataset

% set number of images in each dataset
if dataID <= 3
    ImgNum = 10;
elseif dataID == 6
    ImgNum = 8;
elseif dataID == 7
    ImgNum = 5;
else
    ImgNum = 20;
end


% Parameter settings
scale = 6;      % scale number
alfa = 0.35;    % parameter for balancing the ssim-based weight and the intensity-based weight

for ii = 1 : ImgNum    
    
    if dataID == 1
        img1 = imread(strcat('.\Datasets\PET-MRI\MRI\', num2str(ii), '.png'));
        img2 = imread(strcat('.\Datasets\PET-MRI\PET\', num2str(ii), '.png'));
        fileName = ['.\Results\LEDIF-PET-MRI-', num2str(ii), '.png'];
    elseif dataID == 2
        img1 = imread(strcat('.\Datasets\SPECT-MRI\MRI\', num2str(ii), '.png'));
        img2 = imread(strcat('.\Datasets\SPECT-MRI\SPECT\', num2str(ii), '.png'));
        fileName = ['.\Results\LEDIF-SPECT-MRI-', num2str(ii), '.png'];
    elseif dataID == 3
        img1 = imread(strcat('.\Datasets\CT-MRI\MRI\', num2str(ii), '.png'));
        img2 = imread(strcat('.\Datasets\CT-MRI\CT\', num2str(ii), '.png'));
        fileName = ['.\Results\LEDIF-CT-MRI-', num2str(ii), '.png'];
    elseif dataID == 4
        img1 = imread(strcat('.\Datasets\IV2Dataset\IR', num2str(ii), '.png'));
        img2 = imread(strcat('.\Datasets\IV2Dataset\VIS', num2str(ii), '.png'));
        fileName = ['.\Results\LEDIF-IV2-', num2str(ii), '.png'];
    elseif dataID == 5
        img2 = imread(strcat('.\Datasets\Lytro\lytro-', num2str(ii, '%02d'), '-A.jpg'));
        img1 = imread(strcat('.\Datasets\Lytro\lytro-', num2str(ii, '%02d'), '-B.jpg'));
        fileName = ['.\Results\LEDIF-Lytro-lytro-', num2str(ii, '%02d'), '.png'];
    elseif dataID == 6
        img1 = imread(strcat('.\Datasets\MDDataset\c0', num2str(ii), '_1.tif'));
        img2 = imread(strcat('.\Datasets\MDDataset\c0', num2str(ii), '_2.tif'));
        fileName = ['.\Results\LEDIF-MD-', num2str(ii, '%02d'), '.png'];
    elseif dataID == 7
        img1 = imread(strcat('.\Datasets\OUDataset\u', num2str(ii), '.jpg'));
        img2 = imread(strcat('.\Datasets\OUDataset\o', num2str(ii), '.jpg'));
        fileName = ['.\Results\LEDIF-OU-', num2str(ii), '.png'];
        fileName0 = ['.\Results\LEDIF0-OU-', num2str(ii), '.png'];
    end

    tic
    % transform color images from RGB color space to YCbCr color space if there exists color input image
    is_color = 0;
    if size(img2,3) == 3 || size(img1,3) == 3
        is_color = 1;
        if size(img2,3) == 3
            LAB2 = rgb2ycbcr(img2);
        else
            img2 = cat(3, img2, img2, img2);
            LAB2 = rgb2ycbcr(img2);
        end

        if size(img1,3) == 3
            LAB1 = rgb2ycbcr(img1);
        else
            img1 = cat(3, img1, img1, img1);
            LAB1 = rgb2ycbcr(img1);
        end
        img1 = LAB1(:,:,1);
        img2 = LAB2(:,:,1);
    end

    % ----------------------- Proposed method -------------------------%
    L1 = double(img1(:,:,1));
    L2 = double(img2(:,:,1));

    BFeats1 = {};
    BFeats2 = {};

    DFeats1 = {};
    DFeats2 = {};

    BFeats = {};
    DFeats = {};

    SumBFeat = 0;
    SumDFeat = 0;

    % multi-scale feature extraction with our LEDIFilter
    for jj = 1 : scale      
        L12 = LEDIFilter(L1, jj);
        L22 = LEDIFilter(L2, jj);

        feats1{jj} = L1 - L12;
        feats2{jj} = L2 - L22;

        L1 = L12;
        L2 = L22;
    end

    % ------------------- Entropy based feature fusion ------------------%
    for jj = 1 : scale
        BFeats1{jj} = max(feats1{jj}, 0);
        DFeats1{jj} = min(feats1{jj}, 0);

        BFeats2{jj} = max(feats2{jj}, 0);
        DFeats2{jj} = min(feats2{jj}, 0);

        BFeats{jj} = max(BFeats1{jj}, BFeats2{jj});
        DFeats{jj} = min(DFeats1{jj}, DFeats2{jj});

        en1(jj) = entropy(uint8(BFeats{jj}));
        en2(jj) = entropy(uint8(-DFeats{jj}));
    end

    w1 = en1 / sum(en1) * scale;
    w2 = en2 / sum(en2) * scale;

    for jj = 1 : scale
        SumBFeat = SumBFeat + w1(jj) * BFeats{jj};
        SumDFeat = SumDFeat + w2(jj) * DFeats{jj};
    end


    % ------------------------ Base image fusion -----------------------%
    FL = (L2 + L1)/2 + SumBFeat + SumDFeat;

    [ssimval1, ssimmap1] = ssim(uint8(FL), uint8(L1), 'Exponents', [1 1 1]);
    [ssimval2, ssimmap2] = ssim(uint8(FL), uint8(L2), 'Exponents', [1 1 1]);

    weights1 = (abs(ssimmap1) + eps) ./ (abs(ssimmap1) + abs(ssimmap2) +  eps);    % SSIM based weight map
    weights2 = exp((L1 + eps) ./ (L1 + L2 + eps));                                 % brightness based weight map
    
    weights = weights1 .* weights2 .^ alfa;     % alfa should be adjusted for other types of datasets
    weights = imgaussfilt(weights, 5);

    % base image fusion with similarity and brightness based weighting average
    base = L1 .* weights + L2 .* (1 - weights);


    % ------------------------ Final image fusion -----------------------%
    FL2 = base + SumBFeat + SumDFeat;     % fuse L channel (LAB color space) for color images


    % transform the fusion image from YCbYr color space back to RGB color space
    if is_color
        LAB = LAB1;
        A1 = double(LAB1(:,:,2)); B1 = double(LAB1(:,:,3));
        A2 = double(LAB2(:,:,2)); B2 = double(LAB2(:,:,3));
        Aup = A1 .* abs(A1-127.5) + A2 .* abs(A2-127.5);
        Adw = abs(A1-127.5) + abs(A2-127.5);
        FA = Aup ./ Adw;

        Bup = B1 .* abs(B1-127.5) + B2 .* abs(B2-127.5);
        Bdw = abs(B1-127.5) + abs(B2-127.5);
        FB = Bup ./ Bdw;

        LAB(:,:,1) = uint8(FL2);
        LAB(:,:,2) = uint8(FA);
        LAB(:,:,3) = uint8(FB);
        result = ycbcr2rgb(LAB);
    else
        result = uint8(FL2);    
    end

    % ----------------- Store and visualize fusion image ----------------%
    imwrite(uint8(result), fileName)
    figure, imshow(uint8(result))
end