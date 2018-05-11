function [areaSRKDA,areaSRDA,areaDT,areaRidge,areaSVM]=run_all_inferences(model,settings,options)
    areaSRKDA=srkdaInference(model.K,model.X,model.Y,settings.XTest,settings.YTest,options);
    areaSRDA=srdaInference(model.K,model.X,model.Y,settings.XTest,settings.YTest,options);
    areaDT=trainDT(model.X,model.Y,settings.XTest,settings.YTest,options.classes,options.positiveClass);
    areaSVM=lssvmInference(model.X,model.Y,settings.XTest,settings.YTest,options);
    areaRidge=trainRidge(model.X,model.Y,settings.XTest,settings.YTest,options.classes,options.positiveClass);
    areaSRKDA=max(areaSRKDA,1-areaSRKDA);
    areaSRDA=max(areaSRDA,1-areaSRDA);
    areaDT=max(areaDT,1-areaDT);
    areaSVM=max(areaSVM,1-areaSVM);
    areaRidge=max(areaRidge,1-areaRidge);