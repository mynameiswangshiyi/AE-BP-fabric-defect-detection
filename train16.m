function Solar_SAE
n = 750;
m=20;
train_x = [];
test_x = [];
for i = 1:n
    %filename = strcat(['E:\Graduation design\matlab programming\Solar_SAE\64_64_3train\' num2str(i,'%03d') '.bmp']);
    filename = strcat(['E:\Graduation design\matlab programming\64乘64的图像\正常\' num2str(i) '.bmp']);
    b = imread(filename);
    c = rgb2gray(b);
    [ImageRow ImageCol] = size(c);
    c = reshape(c,[1,ImageRow*ImageCol]);
    train_x = [train_x;c];
end
for i = 1:m
    %filename = strcat(['E:\Graduation design\matlab programming\Solar_SAE\64_64_3test\' num2str(i,'%03d') '.bmp']);
    filename = strcat(['E:\Graduation design\matlab programming\64乘64的图像\纬疵\' num2str(i) '.bmp']);
    %filename = strcat(['E:\科研\数据库\缺陷图\c9\' num2str(i,'%02d') '.bmp']);   
    b = imread(filename);
    c = rgb2gray(b);
    [ImageRow ImageCol] = size(c);
    c = reshape(c,[1,ImageRow*ImageCol]);
    test_x = [test_x;c];
end
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
%train_y = double(train_y);
%test_y  = double(test_y);

%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
%sae = saesetup([4096 800 200 50]);
sae = saesetup([4096 1500 500 200 50]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.5;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 0.5
sae.ae{2}.inputZeroMaskedFraction   = 0.3;

sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 0.5;
sae.ae{3}.inputZeroMaskedFraction   = 0.3;

sae.ae{4}.activation_function       = 'sigm';
sae.ae{4}.learningRate              = 0.5;
sae.ae{4}.inputZeroMaskedFraction   = 0.3;
tic;

opts.numepochs = 10;
opts.batchsize = 50;
sae = saetrain(sae, train_x, opts);
save sae;
%visualize(sae.ae{1}.W{1}(:,2:end)');
toc;
% Use the SDAE to initialize a FFNN
%nn = nnsetup([4096 800 200 50 200 800 4096]);
nn = nnsetup([4096 1500 500 200 50 200 500 1500 4096]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.2;
nn.output                           = 'linear';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    
%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.W{3} = sae.ae{3}.W{1};
nn.W{4} = sae.ae{4}.W{1};
nn.W{5} = sae.ae{4}.W{2};
nn.W{6} = sae.ae{3}.W{2};
nn.W{7} = sae.ae{2}.W{2};
nn.W{8} = sae.ae{1}.W{2};

% Train the FFNN
opts.numepochs =   10;
opts.batchsize = 50;
tx = test_x(16,:);
nn1 = nnff(nn,tx,tx);
ty1 = reshape(nn1.a{9},64,64);
tic;
nn = nntrain(nn, train_x, train_x, opts);
toc;
tic;
nn2 = nnff(nn,tx,tx);
toc;
save SEA_BP nn;
ty2 = reshape(nn2.a{9},64,64);
tx = reshape(tx,64,64);
tz = tx - ty2;
%tz = ty2 - tx;
tz = im2bw(tz,0.05);
%imshow(tx);
%figure,imshow(ty2);
%figure,imshow(tz);
ty = cat(2,tx,ty2,tz);%把数组链接，2表示按行链接
imwrite(ty2,'F:\织物缺陷检测\图\1.jpg')
imwrite(tx,'F:\织物缺陷检测\图\2.jpg')
montage(ty);
%visualize(ty);
%[er, bad] = nntest(nn, tx, tx);
%assert(er < 0.1, 'Too big error');