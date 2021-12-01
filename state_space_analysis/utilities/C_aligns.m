%calculating the cluster index of neural alignments.
%%
load('C_aligns.mat');
%CoM is the postion of each neuron in this demo FOV.
%tot_template is the averaged image of this demo FOV under 2PE.
%L_norm is the normalized subspace contribution(i.e. square of neural alignment) of each neuron in the demo
%FOV.The normalization is basically divide the maximum neural contribution
%across all recorded neurons throughout all recording sessions. For the
%neurons below the contribution threshold, their L_norm values were set to 0.
%The contribution thresholds were based on normalized PR.  
%%
Tuning = L_norm;
[NeuronNum,rankNum] = size(Tuning');


weights_pool0 = Tuning';
%%

for count = 1:rankNum
    figure;
    imshow(tot_template,[000 3000]);hold on
ranki = count;

weights_modeframe=weights_pool0(:,count);
cmap = [0.00,0.45,0.74;
    0.85,0.33,0.10;
    0.93,0.69,0.13
   ];



scatter(CoM(:,2),CoM(:,1),weights_modeframe*200+1,cmap(count,:),'filled','MarkerEdgeColor','k');axis([0 512 0 512]);axis off





end
%%
figure;
    imshow(imresize(tot_template,1),[000 3000]);hold on
    scatter(linspace(10,100,10),linspace(10,100,10),linspace(0.1,1,10)*200);
%%
threshold = 0;
for countrank = 1:rankNum

weights_pool_tmp = weights_pool0(:,countrank);
idx = find(sum(abs(weights_pool_tmp)>threshold,2)>=1);
Tuning =weights_pool_tmp(idx,:);
% for kk = 1:size(Tuning,1)
%     rng shuffle
%     Tuning(kk,:) = Tuning(kk,randperm(6,6));
% end
D1 = pdist(CoM(idx,:));
% D2 = pdist(Tuning,'euclidean');
ii=1;D2 = [];D2tmp1 = [];
for i = 1:length(Tuning)
    for j = 1:length(Tuning)
        if i<j
            
        D2(ii) = abs((Tuning(i)-Tuning(j))/(Tuning(i)+Tuning(j)));
%         D2tmp1(ii) = abs(Tuning(i)-Tuning(j));
        ii = ii+1;
        end
    end
end
% figure;histogram(D2,linspace(0,2,21),'Normalization','probability');
% figure;scatter(D1,D2,1,'k','filled');savefig('tuningvsdistance');saveas(gcf,'tuningvsdistance','pdf')
%
% figure;scatter(CoM(idx,2),CoM(idx,1),weights_modeframe1(idx,count)*5000)
[r p]=corrcoef(D1,D2)
%
[D1_sort,vi] = sort(D1);
D2_sort = D2(vi);
corrbin = linspace(0,550,12);
%
D2_bin_mean = zeros(length(corrbin)-1,1);
D2_bin_sem = D2_bin_mean;
for count = 2:length(corrbin)
   idx =  find(D1_sort>=corrbin(count-1) & D1_sort<=corrbin(count));
   idx_pool(count-1) = length(idx);
   D2_bin_mean(count-1) = mean(D2_sort(idx));
   D2_bin_sem(count-1) = std(D2_sort(idx))./sqrt(length(idx));
    
end
%
% figure;h1 = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,D2_bin_mean,D2_bin_sem);hold on
%
    D2_bin_mean_shuffle = zeros(length(corrbin)-1,1000);
for countj = 1:1000
    rng('shuffle');
    D2_sort = D2(randperm(length(D2)));


for count = 2:length(corrbin)
   idx =  find(D1_sort>=corrbin(count-1) & D1_sort<=corrbin(count));
   D2_bin_mean_shuffle(count-1,countj) = mean(D2_sort(idx));

end

end
cluster_index_pool = bsxfun(@rdivide,D2_bin_mean_shuffle,D2_bin_mean);
cluster_index_pool1(:,:,countrank) = cluster_index_pool;
%
cluster_index_mean1(:,countrank) = mean(cluster_index_pool,2);
cluster_index_std1(:,countrank) = std(cluster_index_pool,0,2);


% figure;h0 = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,cluster_index_mean(:,countrank),3*cluster_index_std(countrank));hold on;
% xlimit = get(gca,'xlim');
% plot(xlimit,[1 1],'--')
end

%
x=corrbin(2:end)-(corrbin(2)-corrbin(1))/2;x=x';h=[];
figure;hold on
% patch([x;flipud(x)],[cluster_index_mean1+cluster_index_std1;flipud(cluster_index_mean1-cluster_index_std1)],ones(length(x)*2,1),[0 0 1],'EdgeColor','none','facealpha',0.5);
% patch([x;flipud(x)],[cluster_index_mean2+cluster_index_std2;flipud(cluster_index_mean2-cluster_index_std2)],ones(length(x)*2,1),[1 0 0],'EdgeColor','none','facealpha',0.5);
for countrank = 1:rankNum
h(countrank) = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,cluster_index_mean1(:,countrank),2*cluster_index_std1(:,countrank));
% h2 = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,cluster_index_mean2,3*cluster_index_std2);
leng{countrank} = ['rank-',num2str(countrank)];
end
xlimit = get(gca,'xlim');
plot(xlimit,[1 1],'k--');
% ylim([0.5 1.2]);
ylabel('Cluster index');
legend(h,leng);
% savefig('cluster_D_NSA_euclidean');


%%


cluster_index_pool1_shuffle=[];
cluster_index_mean2=[];
cluster_index_std2=[];
for countrank = 1:rankNum
weights_pool_tmp = weights_pool0(:,countrank);
idx = find(sum(abs(weights_pool_tmp)>threshold,2)>=1);
Tuning =weights_pool_tmp(idx,:);
rng shuffle
CoMtmp = CoM(randperm(NeuronNum),:);
D1 = pdist(CoMtmp(idx,:));
% D2 = pdist(Tuning,'euclidean');
ii=1;D2 = [];D2tmp1 = [];
for i = 1:length(Tuning)
    for j = 1:length(Tuning)
        if i<j
            
        D2(ii) = abs((Tuning(i)-Tuning(j))/(Tuning(i)+Tuning(j)));
%         D2tmp1(ii) = abs(Tuning(i)-Tuning(j));
        ii = ii+1;
        end
    end
end
% figure;histogram(D2,linspace(0,2,21),'Normalization','probability');
% figure;scatter(D1,D2,1,'k','filled');savefig('tuningvsdistance');saveas(gcf,'tuningvsdistance','pdf')
%
% figure;scatter(CoM(idx,2),CoM(idx,1),weights_modeframe1(idx,count)*5000)
[r p]=corrcoef(D1,D2)
%
[D1_sort,vi] = sort(D1);
D2_sort = D2(vi);
corrbin = linspace(0,550,12);
%
D2_bin_mean = zeros(length(corrbin)-1,1);
D2_bin_sem = D2_bin_mean;
for count = 2:length(corrbin)
   idx =  find(D1_sort>=corrbin(count-1) & D1_sort<=corrbin(count));
   idx_pool(count-1) = length(idx);
   D2_bin_mean(count-1) = mean(D2_sort(idx));
   D2_bin_sem(count-1) = std(D2_sort(idx))./sqrt(length(idx));
    
end
%
% figure;h1 = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,D2_bin_mean,D2_bin_sem);hold on
%
    D2_bin_mean_shuffle = zeros(length(corrbin)-1,1000);
for countj = 1:1000
    rng('shuffle');
    D2_sort = D2(randperm(length(D2)));


for count = 2:length(corrbin)
   idx =  find(D1_sort>=corrbin(count-1) & D1_sort<=corrbin(count));
   D2_bin_mean_shuffle(count-1,countj) = mean(D2_sort(idx));

end

end
cluster_index_pool = bsxfun(@rdivide,D2_bin_mean_shuffle,D2_bin_mean);
cluster_index_pool2_shuffle(:,:,countrank) = cluster_index_pool;
%
cluster_index_mean2(:,countrank) = mean(cluster_index_pool,2);
cluster_index_std2(:,countrank) = std(cluster_index_pool,0,2);

end
%
x=corrbin(2:end)-(corrbin(2)-corrbin(1))/2;x=x';h=[];
figure;hold on
% patch([x;flipud(x)],[cluster_index_mean1+cluster_index_std1;flipud(cluster_index_mean1-cluster_index_std1)],ones(length(x)*2,1),[0 0 1],'EdgeColor','none','facealpha',0.5);
% patch([x;flipud(x)],[cluster_index_mean2+cluster_index_std2;flipud(cluster_index_mean2-cluster_index_std2)],ones(length(x)*2,1),[1 0 0],'EdgeColor','none','facealpha',0.5);
for countrank = 1:rankNum
h(countrank) = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,cluster_index_mean2(:,countrank),3*cluster_index_std2(:,countrank));
% h2 = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,cluster_index_mean2,3*cluster_index_std2);
leng{countrank} = ['rank-',num2str(countrank)];
end
xlimit = get(gca,'xlim');
plot(xlimit,[1 1],'k--');
% ylim([0.5 1.2]);
ylabel('Cluster index');
legend(h,leng);
% savefig('cluster_D_eucldiean_shuffle');
%%
close all
save('Cluster_D_v5.mat','corrbin','cluster_index_pool1','cluster_index_pool2_shuffle');
%%
load('Cluster_D_v5.mat');
cluster_index_mean1 = squeeze(mean(cluster_index_pool1,2));

cluster_index_mean2_shuff = squeeze(mean(cluster_index_pool2_shuffle,2));

cluster_index_std1 = squeeze(std(cluster_index_pool1,[],2))*3;

cluster_index_std2_shuff = squeeze(std(cluster_index_pool2_shuffle,[],2))*3;
%
x=corrbin(2:end)-(corrbin(2)-corrbin(1))/2;x=x';
figure('position',[200 300 1400 600]);
for i = 1:rankNum
    h=[];
    subplot(2,3,i);hold on;title(['rank ',num2str(i)])
    patch([x;flipud(x)],[cluster_index_mean2_shuff(:,i)+cluster_index_std2_shuff(:,i);flipud(cluster_index_mean2_shuff(:,i)-cluster_index_std2_shuff(:,i))],ones(length(x)*2,1),[.6 0.6 0.6],'EdgeColor','none','facealpha',0.5);
    h(1)=plot(x,cluster_index_mean2_shuff(:,i),'k');
    patch([x;flipud(x)],[cluster_index_mean1(:,i)+cluster_index_std1(:,i);flipud(cluster_index_mean1(:,i)-cluster_index_std1(:,i))],ones(length(x)*2,1),[0.00,0.45,0.74],'EdgeColor','none','facealpha',0.5);
    h(2)=plot(x,cluster_index_mean1(:,i));
    xlim([0 600]);ylim([0.3 1.7]);
%     legend(h,{'cluster euclidean','shuff'});
end
