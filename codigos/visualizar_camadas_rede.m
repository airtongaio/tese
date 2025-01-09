% Carregar a rede ResNet-18 pré-treinada
clc;
rede = net_cocao;
outputDir = 'D:\bkp_gaio_alienware_m15_r7_1jun2024\doutorado\experimentos_finais\resultado_final\vizualizacoes_net\tucuma\';

% Carregar uma imagem de teste
img = imread('./tucuma/tucuma.tif'); % Carregar a imagem de teste
img = imresize(img, [512, 512]); % Redimensionar a imagem para o tamanho esperado pela rede

rot = imread('./tucuma/rot_tucuma.tif'); % Carregar a imagem de teste
rot = imresize(rot, [512, 512]); % Redimensionar a imagem para o tamanho esperado pela rede

% Visualizar a imagem original
figure;
imshow(img);
title('Imagem de Teste');
%saveas(gcf, fullfile(outputDir, 'img_Teste.png')); % Salvar a imagem original

imgSize = size(img);
imgSize = imgSize(1:2);

% Iterar sobre todas as camadas da rede
layerNames = {rede.Layers.Name};
for i = 1:length(layerNames)
    layerName = layerNames{i};
    if isa(rede.Layers(i), 'nnet.cnn.layer.Convolution2DLayer')
        % Obter as ativações da camada atual
        act = activations(rede, img, layerName, 'OutputAs', 'channels');

        % Visualizar as ativações
        figure;
        montage(mat2gray(act), 'Size', [8 8]);
        title(['Ativações da camada ', layerName]);
        
        % Salvar a figura como PNG
        %SSsaveas(gcf, fullfile(outputDir, ['Ativacoes_Camada_' layerName '.png']));
    
        sz = size(act);
        act1 = reshape(act,[sz(1) sz(2) 1 sz(3)]);

        [maxValue,maxValueIndex] = max(max(max(act1)));
        act_Max = act1(:,:,:,maxValueIndex);
        act_Max = mat2gray(act_Max);
        act_Max = imresize(act_Max,imgSize);

        figure;
        %A = imtile({img,act_Max});
        %imshow(A);
        imshow(act_Max);
        title(['Ativação mais Representativa ', layerName]);

        % Salvar a figura como PNG
        %saveas(gcf, fullfile(outputDir, ['Ativacoes_Mais_Representativa_' layerName '.png']));
        
        % Calcular a média das ativações ao longo do eixo do canal (terceiro eixo)
        meanAct = mean(act, 3);

        % Normalizar a média das ativações
        normMeanAct = mat2gray(meanAct);
        
        % Visualizar a média das ativações
        figure;
        imagesc(normMeanAct);
        colormap('jet');
        colorbar;
        axis off;
        title(['Média das ativações da camada ', layerName]);
        
        % Salvar a figura como PNG
        %saveas(gcf, fullfile(outputDir, ['Media_ativacoes_' layerName '.png']));
    end
end
