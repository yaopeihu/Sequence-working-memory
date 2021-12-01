load('C_tuning.mat');

%CoM is the postion of each neuron in this demo FOV.
%tot_template is the averaged image of this demo FOV under 2PE.
%weights_pool is the linear regression coefficients on each rank's item.




%%
[NeuronNum,rl_num] = size(weights_pool);
rankNum = rl_num/6;
weights_pool0 = reshape(weights_pool,NeuronNum,6,rankNum);
%% plot neurons' spatial preferences on rank 1


for count = 1%:rankNum
    figure('position',[200 200 400 400]);
%     imshow(imresize(tot_template,0.5),[000 1]);hold on
ranki = count;

weights_modeframe=weights_pool0(:,:,ranki);
cmap =  [10 108 176;
    239 124 33;
    52 153 57;
    202 42 40;
    139 100 168;
    138 85 74]/255;


    [~,vi] = max(weights_modeframe,[],2);
weights_mode = vecnorm(weights_modeframe,2,2).^2;
weights_mode = weights_mode./max(weights_mode);
scatter(CoM(:,2),513-CoM(:,1),weights_mode(:,ranki)*200,cmap(vi,:),'filled','MarkerEdgeColor','k');axis([0 512 0 512]);axis off

end


%% correlation of tuning curve in .

threshold = 0;


cluster_index_pool1=[];
cluster_index_mean1=[];
cluster_index_std1=[];

for countrank = 1:rankNum
weights_pool_tmp = reshape(weights_pool,[],6,rankNum);
weights_pool_tmp = weights_pool_tmp(:,:,countrank);
idx = find(sum(abs(weights_pool_tmp)>=threshold,2)>=1);
Tuning =weights_pool_tmp(idx,:);
% for kk = 1:size(Tuning,1)
%     rng shuffle
%     Tuning(kk,:) = Tuning(kk,randperm(6,6));
% end
D1 = pdist(CoM(idx,:));
D2 = pdist(Tuning,'correlation');% figure;histogram(D2,linspace(0,2,21),'Normalization','probability');
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




%%
threshold = 0;
CoM = CoM(randperm(NeuronNum,NeuronNum),:);

cluster_index_pool1_shuffle=[];
cluster_index_mean1=[];
cluster_index_std1=[];
% weights_modeframe = Tuning();
for countrank = 1:rankNum
weights_pool_tmp = reshape(weights_pool,[],6,rankNum);
weights_pool_tmp = weights_pool_tmp(:,:,countrank);
idx = find(sum(abs(weights_pool_tmp)>=threshold,2)>=1);
Tuning =weights_pool_tmp(idx,:);
for kk = 1:size(Tuning,1)
    rng shuffle
    Tuning(kk,:) = Tuning(kk,randperm(6,6));
end
D1 = pdist(CoM(idx,:));
D2 = pdist(Tuning,'correlation');% figure;histogram(D2,linspace(0,2,21),'Normalization','probability');
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
cluster_index_pool1_shuffle(:,:,countrank) = cluster_index_pool;
%
cluster_index_mean1(:,countrank) = mean(cluster_index_pool,2);
cluster_index_std1(:,countrank) = std(cluster_index_pool,0,2);


% figure;h0 = errorbar(corrbin(2:end)-(corrbin(2)-corrbin(1))/2,cluster_index_mean(:,countrank),3*cluster_index_std(countrank));hold on;
% xlimit = get(gca,'xlim');
% plot(xlimit,[1 1],'--')
end

save('Cluster_D_v4.mat','corrbin','cluster_index_pool1','cluster_index_pool1_shuffle');
%%

load('Cluster_D_v4.mat');
cluster_index_mean1 = squeeze(mean(cluster_index_pool1,2));


cluster_index_mean1_shuff = squeeze(mean(cluster_index_pool1_shuffle,2));


cluster_index_std1 = squeeze(std(cluster_index_pool1,[],2))*3;


cluster_index_std1_shuff = squeeze(std(cluster_index_pool1_shuffle,[],2))*3;


x=corrbin(2:end)-(corrbin(2)-corrbin(1))/2;x=x';
figure('position',[200 300 1400 600]);
for i = 1:rankNum
    h=[];
    subplot(2,3,i);hold on
    patch([x;flipud(x)],[cluster_index_mean1_shuff(:,i)+cluster_index_std1_shuff(:,i);flipud(cluster_index_mean1_shuff(:,i)-cluster_index_std1_shuff(:,i))],ones(length(x)*2,1),[.6 0.6 0.6],'EdgeColor','none','facealpha',0.5);
    h(1)=plot(x,cluster_index_mean1_shuff(:,i),'k');
    patch([x;flipud(x)],[cluster_index_mean1(:,i)+cluster_index_std1(:,i);flipud(cluster_index_mean1(:,i)-cluster_index_std1(:,i))],ones(length(x)*2,1),[0.00,0.45,0.74],'EdgeColor','none','facealpha',0.5);
    h(2)=plot(x,cluster_index_mean1(:,i));
    xlim([0 600]);ylim([0.7 1.7]);
%     legend(h,{'cluster corr','shuff'});

end


