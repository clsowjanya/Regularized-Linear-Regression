clc;
clear;
load patients;
% Pushing attributes in a table called 'patients'
patients = table(Age, Diastolic, Gender, Height, LastName, Location, SelfAssessedHealthStatus, Smoker, Systolic, Weight);

% For Categorical variables
Gender = nominal(Gender);
Location = nominal(Location);
SelfAssessedHealthStatus = nominal(SelfAssessedHealthStatus);

% Creating Dummy variables for the Categorical variables
Gender_dv = dummyvar(Gender);
Location_dv = dummyvar(Location);
Self_dv = dummyvar(SelfAssessedHealthStatus);

% Creating Reference group of all dummy variables
% Gender = Female - Male
Gender_dv_refer = Gender_dv(:, 1);        % Reference group = Male 

% Location = County - StMary - VA
Location_dv_refer = Location_dv(:,1:2);   % Reference group = VA Hospital

% SelfAssess = Excelent - Fasir - Good - Poor
Self_dv_refer = Self_dv(:, 1:3);          % Reference group = Poor

% Standardization of numerical predictors
Age_zs = zscore(Age);
Height_zs = zscore(Height);
Weight_zs = zscore(Weight);

X = [Age_zs Gender_dv_refer Height_zs  Weight_zs Smoker Location_dv_refer Self_dv_refer];
Y= Systolic;
predictor_names={'Age';'GenderFemale'; 'Height';'Weight';'Smoker';'Loc1County';'Loc2StMary';'SE1Excellent';'SE2Fair';'SE3Good'};


% Building model
fprintf('\n\n Building model with all predictors');
final_model = fitlm(X,Y)

[b fitinfo] = lasso(X, Y, 'CV',10, 'Alpha', 1,'PredictorNames',predictor_names);

% Trace Plot of coefficients  fit by Lasso
lassoPlot(b,fitinfo,'PlotType', 'Lambda','PredictorNames',predictor_names, 'XScale','log');
legend('show');

% Cross-validated MSE of Lasso fit
lassoPlot(b,fitinfo,'PlotType','CV');
legend('show');


%Building model using predictors as Smoker and SelfAssessedHealthStauts:Fair
fprintf('\n\n Building model using predictors as Smoker and SelfAssessedHealthStauts:Fair');
X_after_Lasso = [Smoker Self_dv_refer(:,2)];
Y = Systolic;
[b_new fitinfo_new] = lasso(X_after_Lasso, Y, 'CV',10, 'Alpha', 1);
afterLasso = fitlm(X_after_Lasso,Y)
