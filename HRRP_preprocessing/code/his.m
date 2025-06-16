clc
clear
close all;
dirs = 'D:\博士研究课题\hrrp\datasets\rawHRRP0725\';
files = dir([dirs, '*.mat']);
for i = 1:length(files)
    data = load(files(i).name);
    cell = data.HRRP_Fragment;
    fircol = round(cell2mat(cell(:,1)));
    seccol = round(cell2mat(cell(:,2)));
    %h = histogram2(fircol,seccol);
    histogram2(fircol, seccol, 'DisplayStyle', 'tile');
    %h.FaceColor = 'flat';
    %h.DisplayStyle = 'tile';
    %view(2)
    title([files(i).name(1:end-4), '的hrrp方位角和俯仰角分布'],'Interpreter','none');
    xlabel('方位角');
    ylabel('俯仰角');
    colorbar;
    filename = [dirs,'his\',files(i).name(1:end-4),'.fig'];
    savefig(filename);
    close gcf
end





