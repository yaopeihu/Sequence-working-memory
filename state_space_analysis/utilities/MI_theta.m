%%
function [shift_Q12_boot1,shift_Q12_boot2,lock_Q12_boot1,lock_Q12_boot2]=MI_theta(theta_diff1,Q1,Q2,iv12,RA)
%%
diff_tmp = theta_diff1(iv12);
Q_tmp1 = Q1(iv12); 
Q_tmp2 = Q2(iv12);


Im1_boot1=[];
Im2_boot1 =[];
Im1_boot2=[];
Im2_boot2=[];
for booti = 1:100
    rng shuffle
    ivtmp = randi([1,length(Q_tmp1)],[1,length(Q_tmp1)]);
   
    diff_boot1 = diff_tmp(ivtmp);
    Q1_boot = Q_tmp1(ivtmp);
    Q2_boot = Q_tmp2(ivtmp);
    iv_theta = find(diff_boot1>30);
    Im1_boot1(booti)=sum(Q1_boot(iv_theta))/sum(Q1_boot);
    Im2_boot1(booti)=sum(Q2_boot(iv_theta))/sum(Q2_boot);
    
    iv_theta2 = find(diff_boot1<=30);
    Im1_boot2(booti)=sum(Q1_boot(iv_theta2))/sum(Q1_boot);
    Im2_boot2(booti)=sum(Q2_boot(iv_theta2))/sum(Q2_boot);
end

%%
shift_Q12_boot1 = Im1_boot1;
shift_Q12_boot2 = Im2_boot1;
% 
lock_Q12_boot1 = Im1_boot2;
lock_Q12_boot2 = Im2_boot2;



%%

Im1_boot1=shift_Q12_boot1;
Im2_boot1 = shift_Q12_boot2;

Im1_boot2 = lock_Q12_boot1;
Im2_boot2 = lock_Q12_boot2;


i =1;j =2;
colorstr = [0.00,0.45,0.74;
    0.85,0.33,0.10;
    0.93,0.69,0.13];

figure('position',[100 200 400 400]);hold on;h = [];
% h(1)=errorbar([1 2],[mean(Im1_boot2) mean(Im1_boot1)],[std(Im1_boot2),std(Im1_boot1)],'Color',colorstr(i,:),'LineWidth',0.75);
% h(2) = errorbar([1 2],[mean(Im2_boot2) mean(Im2_boot1)],[std(Im2_boot2),std(Im2_boot1)],'Color',colorstr(j,:),'LineWidth',0.75);

boxplot([Im1_boot2' Im2_boot2' zeros(size(Im1_boot2')) zeros(size(Im1_boot2')) Im1_boot1'  Im2_boot1'],'Colors',colorstr([i j i i i j ],:));
% boxplot([Im2_boot2';Im2_boot1'],reshape(ones(100,2)*diag([1 2]),[],1),'PlotStyle','compact','Colors',[.7 .7 .7;.8 0.1 .8]);

 set(gca,'XTickLabel',{' '});


h(1)=plot([1 5],[median(Im1_boot2) median(Im1_boot1)],':','Color',[.6 .6 .6],'LineWidth',0.75);
h(2) = plot([2 6],[median(Im2_boot2) median(Im2_boot1)],':','Color',[.6 .6 .6],'LineWidth',0.75);
axis([0 7 0.15 .85]);xticks([1.5 5.5]);yticks([.2 .5 .8]);box off
xticklabels({'Phase lock','Phase shift'});
ylabel('Relative importance');
legend(h,{['Rank',RA{1}],['Rank',RA{2}]})

end