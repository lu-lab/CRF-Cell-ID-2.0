%close all
%visualize the halves
load(segmentationPath);
zstart1=1;
zend1= round(max(mu_r(:,3))/2);%size(inputImage,3)/2; %specify the dividing z zend1
zend1_saved= size(inputImage,3)/2;
zstart2=zend1+1;
zstart2_saved=zend1_saved+1;
zend2=max(mu_r(:,3))+1;%size(img_1,3);
zend2_saved= size(img_1,3);
%First half
figure, imshow(max(mat2gray(img_1(:,:,zstart1:zend1_saved)),[],3),[],'border','tight');
set(gcf,'Position',[200 100 800 800])
colormap([(0:0.001:1)' zeros(1001,1) zeros(1001,1)]);
caxis([0,0.3])
hold on
mu_r1= mu_r(find(mu_r(:,3)>=zstart1 & mu_r(:,3)<=zend1)',:);
plot(mu_r1(:,1),mu_r1(:,2),'.b')
hold on
count=0;
submur=find(mu_r(:,3)>=zstart1 & mu_r(:,3)<=zend1)';
[~,w]=sort(mu_r(submur,2));
for i=w'
    count= count+1;
remembercolor=[randi([30 70])/100 randi([30 100])/100 randi([30 100])/100];
h=text(mu_r(submur(i),1)+1,mu_r(submur(i),2)+1, string(count),'Color',remembercolor,'FontWeight','Bold','FontSize',10);
set(h, 'Rotation',0);
 text(350,count*12,[char(string(count)) ' -  ' char(top1prediction(submur(i))) '    GT:' char(neuronkey(submur(i)))],'Color',remembercolor,'FontWeight','Bold','FontSize',15)
%text(220,count*12,[char(string(count)) ' -  ' char(top1prediction(submur(i))) ' ' char(string(IDpercent(submur(i),end))) '%  ' char(top2prediction(submur(i))) ' ' char(string(IDpercent(submur(i),end-1))) '%  ' char(top3prediction(submur(i))) ' ' char(string(IDpercent(submur(i),end-2))) '%  ' '(' char(neuronkey(submur(i))) ')'],'Color',remembercolor,'FontWeight','Bold','FontSize',15)
end
%Second half
figure, imshow(max(mat2gray(img_1(:,:,zstart2_saved:zend2_saved)),[],3),[],'border','tight');
set(gcf,'Position',[1200 100 800 800])
colormap([(0:0.001:1)' zeros(1001,1) zeros(1001,1)]);
caxis([0,0.3])
hold on
mu_r2= mu_r(find(mu_r(:,3)>=zstart2 & mu_r(:,3)<=zend2)',:);
plot(mu_r2(:,1),mu_r2(:,2),'.b')
hold on
count=0;
submur=find(mu_r(:,3)>=zstart2 & mu_r(:,3)<=zend2)';
[~,w]=sort(mu_r(submur,2));
for i=w' %find(mu_r(:,3)>=zstart2 & mu_r(:,3)<=zend2)'
    count=count+1;
remembercolor=[randi([20 100])/100 randi([20 100])/100 randi([20 100])/100];
h=text(mu_r(submur(i),1)+1,mu_r(submur(i),2)+1, string(count),'Color',remembercolor,'FontWeight','Bold','FontSize',10);
set(h, 'Rotation',0);
 text(350,count*12,[char(string(count)) ' -  ' char(top1prediction(submur(i))) '    GT:'  char(neuronkey(submur(i)))],'Color',remembercolor,'FontWeight','Bold','FontSize',15)
%text(220,count*12,[char(string(count)) ' -  ' char(top1prediction(submur(i))) ' ' char(string(IDpercent(submur(i),end))) '%  ' char(top2prediction(submur(i))) ' ' char(string(IDpercent(submur(i),end-1))) '%  ' char(top3prediction(submur(i))) ' ' char(string(IDpercent(submur(i),end-2))) '%  ' '(' char(neuronkey(submur(i))) ')'],'Color',remembercolor,'FontWeight','Bold','FontSize',14)
end
