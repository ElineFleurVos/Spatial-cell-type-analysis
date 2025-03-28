% Step 1: Load the CSV file
single_cell_dir = 'C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables\P37_S29-CRC01.csv';
data = readtable(single_cell_dir);

%%
% Step 2: Extract the column of marker intensities
markersample = data.Pan_CK;  % Replace 'marker_column' with the actual column name

% Step 3: Define the parameters
minval = prctile(markersample,1);  % Minimum marker intensity value
FDR = 0.05;  % False Discovery Rate

% Step 4: Run the findcutoff function
% [cutoff, poscells, GMMmodel] = findcutoff(markersample, minval, FDR);

tic
%options = statset('MaxIter', 1000);
GMMmodel = fitgmdist(markersample(markersample>minval),2,'replicates',10);
%GMMmodel = fitgmdist(markersample,2,'replicates',10);
toc
    
[~,minid] = min(GMMmodel.mu);
[~,maxid] = max(GMMmodel.mu);
    
peak_min = normpdf(GMMmodel.mu(minid),GMMmodel.mu(minid),sqrt(GMMmodel.Sigma(minid)));
peak_plus = normpdf(GMMmodel.mu(maxid),GMMmodel.mu(maxid),sqrt(GMMmodel.Sigma(maxid)));

stepsize=(prctile(markersample,70)-prctile(markersample,2))/1000;
if GMMmodel.ComponentProportion(maxid)*peak_plus>GMMmodel.ComponentProportion(minid)*peak_min
    searchrange = prctile(markersample,2):stepsize:prctile(markersample,70);
else
    searchrange = prctile(markersample,10):stepsize:prctile(markersample,98);
end
  
objfunc = searchrange;

tic
for searchid = 1:size(searchrange,2)
    objfunc(searchid) = (falserate(searchrange(searchid),GMMmodel)-FDR)^2;
end
toc    
    
%figure,plot(searchrange,log(objfunc)) %this is a diagnostic line
    
[~,minobj] = min(objfunc);
cutoff = searchrange(minobj);
    
%cutoff = fmincon(@(x) (falserate(x,GMMmodel)-FDR).^2,mean(markersample),[1 -1]',[prctile(markersample,99) -min(GMMmodel.mu)]);
%cutoff = fzero(@(x) (falserate(x,GMMmodel)-FDR), mean(markersample));
    
poscells = markersample>cutoff;

%% I'm plotting the results here; not sure how you want to incorporate this into your code
    
%figure()
    
histogram(markersample,'Normalization','pdf','EdgeAlpha',0.5,'FaceAlpha',0.3,'FaceColor','k')
ylimits = ylim;
hold on
x = linspace(minval,max(markersample),1000);
%plot(x,pdf(GMMmodel,x'))
    
plot(x,GMMmodel.ComponentProportion(maxid)*normpdf(x,GMMmodel.mu(maxid),sqrt(GMMmodel.Sigma(maxid))),'-b','linewidth',2)
plot(x,GMMmodel.ComponentProportion(minid)*normpdf(x,GMMmodel.mu(minid),sqrt(GMMmodel.Sigma(minid))),'-g','linewidth',2)
    
line([cutoff cutoff], ylimits, 'Color', 'r','linewidth',2);
ylim(ylimits);
%text(cutoff+5e3,mean(ylimits)+10,['cutoff=' num2str(cutoff,2), newline, '%pos=' num2str(100*sum(poscells)/length(poscells),'%2.1f'), newline, 'FDR=' num2str(FDR+sqrt(min(objfunc)),2)],'fontsize',12)
text(1000,0.005,['cutoff=' num2str(cutoff,2), newline, '%pos=' num2str(100*sum(poscells)/length(poscells),'%2.1f'), newline, 'FDR=' num2str(FDR+sqrt(min(objfunc)),2)],'fontsize',12)
legend ('Ori.data','1st Dist.','2nd Dist.','Gate');
xlim([0 3000])
ylim([0 10e-3])
%figure,plot(searchrange,log(objfunc)) %this is a diagnostic line
    