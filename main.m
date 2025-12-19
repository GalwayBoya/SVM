clear all;
clc;

% 参数设置
K = 5; % K折交叉验证

%%% 数据导入与预处理
[X, Y, Xpreprocess, Y_all] = Preprocess();

%% === SVM 模型建立与预测 ===

% 1. 交叉验证 (Cross Validation)
fprintf('\n正在进行 %d 折交叉验证...\n', K);
% 训练 SVM 模型，使用线性核函数，标准化数据
SVMModel = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [1, 7]);
CVSVMModel = crossval(SVMModel, 'KFold', K);
classLoss = kfoldLoss(CVSVMModel);
fprintf('交叉验证分类误差: %.4f\n', classLoss);
fprintf('交叉验证准确率: %.2f%%\n', (1-classLoss)*100);

% 2. 训练最终模型 (使用平衡后的数据)
fprintf('\n正在训练最终 SVM 模型...\n');
FinalModel = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [1, 7]);

% 3. 预测 (使用所有原始数据 Xpreprocess)
fprintf('正在对所有数据进行预测...\n');
[label_pred, score] = predict(FinalModel, Xpreprocess);

% 计算性能指标
% 混淆矩阵
C = confusionmat(Y_all, label_pred);
% 正常果(1)是第一类，水脱果(7)是第二类
TP = C(1,1); % 真实1，预测1
FN = C(1,2); % 真实1，预测7
FP = C(2,1); % 真实7，预测1
TN = C(2,2); % 真实7，预测7

Sensitivity = TP / (TP + FN) * 100; % 正常果识别率
Specificity = TN / (TN + FP) * 100; % 水脱果识别率
Accuracy = (TP + TN) / sum(C(:)) * 100; % 总准确率

fprintf('\n==================== SVM 模型结果报告 ====================\n');
fprintf('正常果识别率 (Sensitivity): %.2f%%\n', Sensitivity);
fprintf('水脱果识别率 (Specificity): %.2f%%\n', Specificity);
fprintf('总准确率 (Accuracy)       : %.2f%%\n', Accuracy);
fprintf('==========================================================\n');

%% === 结果展示部分 ===

% 图1：预测集分类效果散点图
idx_normal = find(Y_all == 1);
idx_water = find(Y_all == 7);

% SVM score(:,2) 表示属于第二类(7)的得分/距离
% 为了可视化，我们可以画出得分
% 这里的score是距离超平面的距离，正值偏向类7，负值偏向类1 (取决于ClassNames顺序)
% ClassNames是[1, 7]，所以score(:,2)越大越可能是7

figure('Name', 'Prediction Results', 'Color', 'w');
hold on;
% 绘制正常果（真实值为1）的预测得分
plot(idx_normal, score(idx_normal, 2), 'g.', 'MarkerSize', 10, 'DisplayName', '真实: 正常果');
% 绘制水脱果（真实值为7）的预测得分
plot(idx_water, score(idx_water, 2), 'r.', 'MarkerSize', 10, 'DisplayName', '真实: 水脱果');

yline(0, 'k--', 'LineWidth', 1.5, 'DisplayName', '分类阈值 (0)');

xlabel('样本编号');
ylabel('SVM 得分 (距离)');
title(['全样本预测结果 (准确率: ' num2str(Accuracy, '%.1f') '%)']);
legend('Location', 'best');
grid on;
box on;

% 图2：混淆矩阵
figure('Name', 'Confusion Matrix', 'Color', 'w');
cm = confusionchart(Y_all, label_pred);
cm.Title = 'SVM 混淆矩阵';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
