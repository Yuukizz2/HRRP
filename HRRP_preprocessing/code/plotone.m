clc
clear
close all;
dirs = 'D:\博士研究课题\hrrp\datasets\rawHRRP0725\new\';
filename = '1740_BY737_0322_4.mat';
data = load([dirs, filename]);
res0 = data.newcell;
%[r,c] = size(res0);
%res1 = res0(1, :);
%x = 1:c;
plot(res0{1000,5})
xlim([0 512])
xlabel('Range Cell')
ylabel('Normalized Amplitude')
%title(extractBefore(filename, '_'), 'FontWeight', 'bold')
title(['B737, ', 'azimuth:', num2str(round(res0{1000,1}, 1)), '°, elevation:', num2str(round(res0{1000,2}, 1)), '°'], 'FontWeight', 'bold')
%saveas(gcf, [dirs,'plot\', extractBefore(filename, '_'),'.png']);
saveas(gcf, [dirs,'plot\', filename,'.png']);


%%%%%%%%%%%%多类%%%%%%%%%%%%%%
% files = dir([dirs,'*.mat']);
% for i = 1:length(files)
%     data = load(files(i).name);
%     res0 = data.res;
%     [r,c] = size(res0);
%     %res1 = res0(1, :);
%     %x = 1:c;
%     subplot(2,4,i)
%     plot(res0(200,:))
%     xlim([0 c])
%     xlabel('Range Cell')
%     ylabel('Normalized Amplitude')
%     title(extractBefore(files(i).name, '_'), 'FontWeight', 'bold')
% end
% %saveas(gcf, [dirs,'plot\', 'allclass.png']);