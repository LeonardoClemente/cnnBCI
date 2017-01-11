
close all
clear all
%pseudo EEG plot
load('/Users/leonardo/Downloads/A-Generalizable-BCI-using-Machine-Learning-for-Feature-Discovery-master/data/EEG_ParticipantD_01_10_2013.mat')

nChannels=46;
nSamples=100;
nEpochs=size(data_epochs_D,2);
epoch=95;
rows=size(data_epochs_D,1);


%moving average
mask = ones(1,10)/10;

epochSample=zeros(nChannels,100);

%New data holders
decimatedMatrix=zeros(nChannels,nChannels,nEpochs);
decimatedMatrixTest=zeros(nChannels,nChannels,nEpochs);
dataVector=zeros(1,100);
decimatedVector=zeros(1,50);
decimatedVectors=zeros(nEpochs,nChannels*nChannels);
dmVectors=zeros(nEpochs,nChannels*nChannels);
icaVectors=zeros(nEpochs,nChannels*nChannels);
ibmDataPreprocessed = zeros(nEpochs, nSamples*nChannels);

% PRE PROCESSING FOR IBMCNN

for i=1:nEpochs
    for j=1:nChannels
    dataVector=data_epochs_D((j-1)*nSamples+1:(j)*nSamples,i);
    dataVector=dataVector - mean(dataVector);
    dataVector=dataVector/std(dataVector);
    % dataVector=conv(dataVector,mask,'same'); %FILTERING STEP
    ibmDataPreprocessed(i,(j-1)*nSamples+1:j*nSamples)=dataVector;
    end
end


%Averaging over all channels of each class % Data doesn't show any
%discernible feature, only difference between the signal magnitude.
% DATA WAS RECORDED 100 ms before each onset and 300 ms after each onset (4 ms per value)
ibmDataPreprocessed = ibmDataPreprocessed';
sampleAverage1 = zeros(1,4600);
sampleAverage1 = zeros(1,4600);
sampleAverage1 = zeros(1,4600);
counter1 = 0;
counter2 = 0;
counter3 = 0;

indices1=find(data_key_D(data_key_D == 1));
indices2=find(data_key_D(data_key_D == 2));
indices3=find(data_key_D(data_key_D == 3));

average1 = reshape(mean(ibmDataPreprocessed(:,indices1),2),[100,46])';
average2 = reshape(mean(ibmDataPreprocessed(:,indices2),2),[100, 46])';
average3 = reshape(mean(ibmDataPreprocessed(:,indices3),2),[100, 46])';

averageMat = [mean(ibmDataPreprocessed(:,indices1),2)'
mean(ibmDataPreprocessed(:,indices2),2)'
mean(ibmDataPreprocessed(:,indices3),2)'];

colors = {[1 0 0],[0 1 0],[0 0 1]};
plots = cell(1,3);
fftPlots = cell(1,3);
PP1 = cell(1,3);
legendMessage={'No move','Left Squeeze','Right Squeeze'};
x =(1:100)*4

% FFT info
Fs = 250;
T = 1/Fs;
L = 100;
t = (0:L-1)*T;
f = Fs*(0:(L/2))/L;



for channel = 1:4
    % EEG PLOT
    figure(channel)
    hold on
    for i = 1:3
       ft = fft(averageMat(i,1+nSamples*(channel-1):channel*nSamples));
       P2 = abs(ft/L);
       P1 = P2(1:L/2+1);
       P1(2:end-1) = 2*P1(2:end-1);
       PP1{i}=P1;
       plots{i} = plot(x,averageMat(i,1+nSamples*(channel-1):channel*nSamples),'Color',colors{i},'LineWidth',1.5);
 
    end
    xlabel('Time (ms)','FontSize',20);
    ylabel('V','FontSize',20);
    legend([plots{:}],legendMessage{:});
    hold off
    
    % Fourier TransformPlot
    figure(channel+4)
    hold on
    for i = 1:3
       fftPlots{i} =plot(f,PP1{i},'Color',colors{i},'LineWidth',1.5) ;
    end
    hold off
    xlabel('fs (hz)','FontSize',20);
    ylabel('V/s','FontSize',20);
    display(fftPlots)
    legend([fftPlots{:}],legendMessage{:});
end


%     
% for i = 1:nChannels
%     epochSample(i,:)=data_epochs_D((i-1)*100+1:(i)*100,epoch);
%     if i < 23
%         epochSample(i,:) = epochSample(i,:);
%     end
% end
% 
% yTranslation=max(epochSample(1,:))*4;
% x=1:100;
% 
% figure(1)
% hold on
%     for i=1:1
%     plot(x(:),epochSample(i,:)+yTranslation*(i-1));
%     end
% hold off
% 
% 
%         
% %Data selection and plotting.
% for i = 1:nChannels
%     epochSample(i,:)=data_epochs_D((i-1)*100+1:(i)*100,epoch);
%     if i < 23
%         epochSample(i,:) = epochSample(i,:);
%     end
% end
% 
% yTranslation=max(epochSample(1,:))*4;
% x=1:100;
% 
% figure(1)
% hold on
%     for i=1:1
%     plot(x(:),epochSample(i,:)+yTranslation*(i-1));
%     end
% hold off
% 
% 
% %Down sampling using decimation
% %Averaging one dimensional convolution
% 
% 
% % 
% % for i=1:nEpochs
% %     for j=1:nChannels
% %     dataVector=data_epochs_D((j-1)*100+1:(j)*100,i);
% %     movAv= conv(dataVector,mask,'same');
% %     decimatedVector=decimate(movAv,2);
% %     decimatedMatrix(j,:,i)=decimatedVector(3:48);
% %     decimatedVectors(i,(j-1)*46+1:j*46)=decimatedVector(3:48);
% %     end
% % end
% 

% % 
% % 
% % figure(1)
% % plot(decimatedVector)
% % figure(2)
% % plot(movAv)
% 
% 
% 
% 
% % 
% % % ICA DATASET
% % 
% % 
%TEST DATASET
v =1:10;
seno=sin((2*pi/10)*v);
k=repmat(seno,[46,1]);
elvector=zeros(1,46*46);
for i=1:1868
    matrix=decimatedMatrix(:,:,i);
    icaMatrix = myICA(matrix,nChannels);
    decimatedMatrixTest(:,:,i)=icaMatrix;
    elvector=reshape(decimatedMatrixTest(:,:,i)',[1,46*46]);
    icaVectors(i,:)=elvector;
end
        

   




%TEST DATASET
v =1:10;
seno=sin((2*pi/10)*v);
k=repmat(seno,[46,1]);

elvector=zeros(1,46*46);
for i=1:1868
    if data_key_D(i)==2
        matrix=decimatedMatrix(:,:,i);
        means=max(matrix')';
        means=repmat(means,[1,10]);
        sink=k.*means*2;
        matrix(:,23:32)=matrix(:,23:32)+sink;
        decimatedMatrixTest(:,:,i)=matrix;
    else
        decimatedMatrixTest(:,:,i)=decimatedMatrix(:,:,i);
    end
    elvector=reshape(decimatedMatrixTest(:,:,i)',[1,46*46]);
    dmVectors(i,:)=elvector;
end
        
% 
%  

matrix = matrixorig;
v =1:20;
seno=sin((2*pi/20)*v);
k=repmat(seno,[46,1]);
matrix(:,41:60) = matrix(:,41:60)+k;

figure(10)
hold on
f=plot(matrix(1,:),'Color',[1 0 0])
g=plot(matrixorig(1,:),'Color',[0 1 0])
hold off
xlabel('Time (ms)','FontSize',20);
ylabel('V','FontSize',20);
legend([f,g],{'Artificial','Original'});