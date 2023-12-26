# LLM4HPCTransCompile

## Issue
### 数据集
- [-] 将所有算子名加上shape，保证算子名各不相同
### 训练
- [ ] Prompt输入长度太长，模型上下文窗口无法满足记忆
- [ ] 对于长文本代码翻译end-to-end效果差，需要考虑使用step-by-step方法

## Plan

### Model
- [ ] 设计cpu2cuda、中间IR转换的prompt，进一步丰富数据集
- [ ] 对模型进行全量微调的bug修复
- [ ] 按算子进行划分重新训练一版模型

### Idea
- [ ] 其他垂类任务的fine-tune调研
- [ ] prompt部分的创新点

### Paper
- [ ] Grammar Prompting for Domain-Specific Language Generation with Large LanguageModels
