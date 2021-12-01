CR1 = load('Monkey1_acc_perday.mat');% get monkey 1's correct rates across seq lengths and ranks.
CR2 = load('Monkey2_acc_perday.mat');% get monkey 2's correct rates across seq lengths and ranks.
itemerror_len3 = load('itemerror_len3.mat');% get both monkeys' response rates on items.[20 monkey1 sessions, 13 monkey2 sessions]
ordererror_len3 = load('ordererror_len3.mat');% get both monkeys' response rates on ranks.[20 monkey1 sessions, 13 monkey2 sessions]




%% plot correct rates on each rank.
CR_pool = CR1.CR_pool;
mCR = cell(length(CR_pool),1);
semCR =mCR;
for countf = 1:length(CR_pool)

mCR{countf} = nanmean(CR_pool{countf},1);semCR{countf} = nanstd(CR_pool{countf},[],1)/sqrt(length(CR_pool{countf}));
end
cmap = [0.00,0.45,0.74;
    0.85,0.33,0.10;
    0.93,0.69,0.13];
figure;
for count = 1:length(mCR)-1
    

hold on; h = [];
    setsize = count+1;
    h(count) = errorbar(1:setsize,mCR{count},semCR{count},'Color',cmap(count,:),'LineWidth',2); 

    leng_CR{count} = ['setsize: ',num2str(setsize)];
    axis([0 4 0 1])
xlabel('Serial Order #'); ylabel('Correct Rate'); 
set(gca,'XTick',1:setsize,'FontSize',14,'FontName','Arial','FontWeight','bold');
end
plot([0 4],[1/6 1/6],'k--');title('Monkey 1');
%% plot item error
acc_mat_pool = itemerror_len3.acc_mat_pool;
acc_mat_mean = nanmean(acc_mat_pool,3);
acc_mat_sem = nanstd(acc_mat_pool,0,3)./sqrt(size(acc_mat_pool,3));
%
colorstring = [10 108 176;
    239 124 33;
    52 153 57;
    202 42 40;
    139 100 168;
    138 85 74]/255;
h = [];
figure;hold on
for count = 1:6
    h(count)=errorbar(acc_mat_mean(count,:),acc_mat_sem(count,:),'CapSize',12,'LineWidth',0.75,'MarkerSize',15,'color', colorstring(count,:));
   
end
set(gca, 'XTick', 1:6);
set(gca, 'YTick', 0:0.25:1);

set(gca, 'TickLength', [0.02 0.025]);
set(gca, 'Tickdir', 'in');

xlim([0.5 6+0.5]);
ylim([0 1])
%% plot rank error
order_mat_pool = ordererror_len3.order_mat_pool;
order_mat_mean = nanmean(order_mat_pool,3);
order_mat_sem =nanstd(order_mat_pool,0,3)./sqrt(size(order_mat_pool,3));
figure('position',[400 300 350 400]);hold on
for count = 1:setsize
    errorbar(order_mat_mean(count,:),order_mat_sem(count,:),'-','LineWidth',0.75,'MarkerSize',15,'CapSize',10)
end
set(gca, 'XTick',1:setsize);%set(gca,'XTickLabel',{'absent','1','2','3'});
set(gca, 'YTick', 0:0.25:1);
set(gca, 'LineWidth', 1, 'FontSize', 15, 'FontName', 'Arial');
set(gca, 'TickLength', [0.02 0.025]);
set(gca, 'Tickdir', 'out');
xlabel('Rank', 'FontName', 'Arial', 'FontSize', 15);
xlim([0.5 setsize+0.5]);
ylim([0 1])
